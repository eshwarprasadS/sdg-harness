from __future__ import annotations

import asyncio
import time
from statistics import mean

import litellm
import structlog

from sdg_harness.eval.cost import CostTracker
from sdg_harness.eval.models import (
    FailureCategory,
    L1BatchResult,
    L1Config,
    SampleResult,
)


class WeakStrongEvaluator:
    def __init__(
        self,
        config: L1Config,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        self._config = config
        self._cost_tracker = cost_tracker or CostTracker(max_cost=config.max_cost)
        self._sem = asyncio.Semaphore(10)
        self._log = structlog.get_logger().bind(component="l1_eval")

    async def evaluate_batch(
        self, samples: list[dict[str, str]]
    ) -> L1BatchResult:
        start = time.monotonic()
        self._log.info(
            "eval_batch_start",
            num_samples=len(samples),
            weak_model=self._config.weak_model,
            strong_model=self._config.strong_model,
        )

        tasks = [self._evaluate_sample(s) for s in samples]
        results: list[SampleResult] = await asyncio.gather(*tasks)

        elapsed = time.monotonic() - start
        passed = [r for r in results if r.passed]
        failure_counts: dict[str, int] = {}
        for r in results:
            if r.failure_category is not None:
                key = r.failure_category.value
                failure_counts[key] = failure_counts.get(key, 0) + 1

        batch = L1BatchResult(
            samples=results,
            weak_avg=mean(r.weak_acc for r in results) if results else 0.0,
            strong_avg=mean(r.strong_acc for r in results) if results else 0.0,
            gap_avg=mean(r.gap for r in results) if results else 0.0,
            pass_rate=len(passed) / len(results) if results else 0.0,
            failure_counts=failure_counts,
            total_cost=self._cost_tracker.total,
            duration_seconds=elapsed,
        )

        self._log.info(
            "eval_batch_complete",
            num_samples=len(results),
            pass_rate=batch.pass_rate,
            weak_avg=batch.weak_avg,
            strong_avg=batch.strong_avg,
            gap_avg=batch.gap_avg,
            total_cost=batch.total_cost,
            duration_seconds=batch.duration_seconds,
        )
        return batch

    async def _evaluate_sample(self, sample: dict[str, str]) -> SampleResult:
        sample_id = sample.get("sample_id", sample.get("prompt", "")[:32])
        prompt = sample["prompt"]

        weak_scores = await self._run_rollouts(
            sample, self._config.weak_model, self._config.num_rollouts
        )
        strong_scores = await self._run_rollouts(
            sample, self._config.strong_model, self._config.num_rollouts
        )

        weak_acc = mean(weak_scores) if weak_scores else 0.0
        strong_acc = mean(strong_scores) if strong_scores else 0.0
        gap = strong_acc - weak_acc

        passed = self._check_pass(weak_acc, strong_acc, gap)
        failure_category = (
            None if passed else self._classify_failure(weak_acc, strong_acc, gap)
        )

        result = SampleResult(
            sample_id=sample_id,
            prompt=prompt,
            weak_scores=weak_scores,
            strong_scores=strong_scores,
            weak_acc=weak_acc,
            strong_acc=strong_acc,
            gap=gap,
            passed=passed,
            failure_category=failure_category,
        )

        self._log.info(
            "sample_evaluated",
            sample_id=sample_id,
            weak_acc=weak_acc,
            strong_acc=strong_acc,
            gap=gap,
            passed=passed,
        )
        return result

    async def _run_rollouts(
        self, sample: dict[str, str], model: str, n: int
    ) -> list[float]:
        scores: list[float] = []
        expected = sample["expected_answer"]
        messages = [{"role": "user", "content": sample["prompt"]}]

        for _ in range(n):
            async with self._sem:
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    temperature=self._config.temperature,
                )

            content = response.choices[0].message.content or ""
            score = 1.0 if expected.lower() in content.lower() else 0.0
            scores.append(score)

            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens or 0,
                    "completion_tokens": response.usage.completion_tokens or 0,
                }
                self._cost_tracker.track(model, usage)

                spent = self._cost_tracker.total
                limit = self._cost_tracker._max_cost
                if limit > 0 and spent >= limit * 0.8 and spent < limit:
                    self._log.warning(
                        "budget_warning",
                        spent=spent,
                        limit=limit,
                        pct=spent / limit * 100,
                    )

                self._cost_tracker.check_budget()

        return scores

    def _classify_failure(
        self, weak_acc: float, strong_acc: float, gap: float
    ) -> FailureCategory:
        thresholds = self._config.thresholds
        if weak_acc > thresholds.weak_avg_max:
            return FailureCategory.TOO_EASY
        if strong_acc < thresholds.strong_avg_min:
            return FailureCategory.FAILED_ON_STRONG
        return FailureCategory.QUALITY_FAIL

    def _check_pass(
        self, weak_acc: float, strong_acc: float, gap: float
    ) -> bool:
        thresholds = self._config.thresholds
        return (
            weak_acc <= thresholds.weak_avg_max
            and strong_acc >= thresholds.strong_avg_min
            and gap >= thresholds.gap_min
        )
