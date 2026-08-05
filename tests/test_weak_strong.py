from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sdg_harness.eval.cost import BudgetExceededError, CostTracker
from sdg_harness.eval.models import FailureCategory, L1Config
from sdg_harness.eval.weak_strong import WeakStrongEvaluator


def _make_config(**overrides: object) -> L1Config:
    defaults: dict[str, object] = {
        "weak_model": "gpt-3.5-turbo",
        "strong_model": "gpt-4o",
        "num_rollouts": 3,
        "temperature": 1.0,
        "max_cost": 100.0,
    }
    defaults.update(overrides)
    return L1Config(**defaults)  # type: ignore[arg-type]


def _make_response(
    content: str,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    resp.usage = usage
    return resp


def _make_sample(
    prompt: str = "What is 2+2?",
    expected: str = "4",
    sample_id: str = "s1",
) -> dict[str, str]:
    return {"sample_id": sample_id, "prompt": prompt, "expected_answer": expected}


class TestClassifyFailure:
    def test_too_easy(self) -> None:
        ev = WeakStrongEvaluator(_make_config())
        cat = ev._classify_failure(weak_acc=0.80, strong_acc=0.90, gap=0.10)
        assert cat == FailureCategory.TOO_EASY

    def test_failed_on_strong(self) -> None:
        ev = WeakStrongEvaluator(_make_config())
        cat = ev._classify_failure(weak_acc=0.30, strong_acc=0.40, gap=0.10)
        assert cat == FailureCategory.FAILED_ON_STRONG

    def test_quality_fail(self) -> None:
        ev = WeakStrongEvaluator(_make_config())
        cat = ev._classify_failure(weak_acc=0.50, strong_acc=0.60, gap=0.10)
        assert cat == FailureCategory.QUALITY_FAIL

    def test_passing_conditions_not_classified(self) -> None:
        ev = WeakStrongEvaluator(_make_config())
        passed = ev._check_pass(weak_acc=0.30, strong_acc=0.70, gap=0.40)
        assert passed is True


class TestCheckPass:
    def test_all_met(self) -> None:
        ev = WeakStrongEvaluator(_make_config())
        assert ev._check_pass(0.50, 0.80, 0.30) is True

    def test_weak_too_high(self) -> None:
        ev = WeakStrongEvaluator(_make_config())
        assert ev._check_pass(0.66, 0.90, 0.24) is False

    def test_strong_too_low(self) -> None:
        ev = WeakStrongEvaluator(_make_config())
        assert ev._check_pass(0.30, 0.59, 0.29) is False

    def test_gap_too_small(self) -> None:
        ev = WeakStrongEvaluator(_make_config())
        assert ev._check_pass(0.50, 0.69, 0.19) is False

    def test_at_boundary_weak(self) -> None:
        ev = WeakStrongEvaluator(_make_config())
        assert ev._check_pass(0.65, 0.90, 0.25) is True

    def test_at_boundary_strong(self) -> None:
        ev = WeakStrongEvaluator(_make_config())
        assert ev._check_pass(0.40, 0.60, 0.20) is True

    def test_at_boundary_gap(self) -> None:
        ev = WeakStrongEvaluator(_make_config())
        assert ev._check_pass(0.40, 0.60, 0.20) is True


class TestEvaluateBatch:
    @pytest.mark.asyncio
    async def test_end_to_end_passing(self) -> None:
        config = _make_config(num_rollouts=2)
        ev = WeakStrongEvaluator(config)

        weak_resp = _make_response("I think the answer is 5")
        strong_resp = _make_response("The answer is 4")

        with patch("sdg_harness.eval.weak_strong.litellm") as mock_llm:
            async def mock_acompletion(model: str, **kwargs: object) -> MagicMock:
                if model == "gpt-3.5-turbo":
                    return weak_resp
                return strong_resp

            mock_llm.acompletion = AsyncMock(side_effect=mock_acompletion)

            samples = [_make_sample(), _make_sample(sample_id="s2")]
            result = await ev.evaluate_batch(samples)

        assert len(result.samples) == 2
        for sr in result.samples:
            assert sr.weak_acc == 0.0
            assert sr.strong_acc == 1.0
            assert sr.gap == 1.0
            assert sr.passed is True

        assert result.pass_rate == 1.0
        assert result.failure_counts == {}

    @pytest.mark.asyncio
    async def test_end_to_end_failing(self) -> None:
        config = _make_config(num_rollouts=1)
        ev = WeakStrongEvaluator(config)

        both_correct = _make_response("The answer is 4")

        with patch("sdg_harness.eval.weak_strong.litellm") as mock_llm:
            mock_llm.acompletion = AsyncMock(return_value=both_correct)

            samples = [_make_sample()]
            result = await ev.evaluate_batch(samples)

        sr = result.samples[0]
        assert sr.weak_acc == 1.0
        assert sr.strong_acc == 1.0
        assert sr.passed is False
        assert sr.failure_category == FailureCategory.TOO_EASY
        assert result.failure_counts["TOO_EASY"] == 1

    @pytest.mark.asyncio
    async def test_budget_exceeded_propagates(self) -> None:
        config = _make_config(num_rollouts=1, max_cost=0.0001)
        cost_tracker = CostTracker(max_cost=0.0001)
        ev = WeakStrongEvaluator(config, cost_tracker=cost_tracker)

        resp = _make_response(
            "answer 4",
            prompt_tokens=1_000_000,
            completion_tokens=500_000,
        )

        with patch("sdg_harness.eval.weak_strong.litellm") as mock_llm:
            mock_llm.acompletion = AsyncMock(return_value=resp)

            with pytest.raises(BudgetExceededError):
                await ev.evaluate_batch([_make_sample()])

    @pytest.mark.asyncio
    async def test_failure_count_aggregation(self) -> None:
        config = _make_config(num_rollouts=1)
        ev = WeakStrongEvaluator(config)

        both_correct = _make_response("The answer is 4")
        both_wrong = _make_response("I don't know")

        call_count = 0

        with patch("sdg_harness.eval.weak_strong.litellm") as mock_llm:
            async def mock_acompletion(**kwargs: object) -> MagicMock:
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    return both_correct
                return both_wrong

            mock_llm.acompletion = AsyncMock(side_effect=mock_acompletion)

            samples = [
                _make_sample(sample_id="easy"),
                _make_sample(sample_id="hard"),
            ]
            result = await ev.evaluate_batch(samples)

        assert len(result.failure_counts) > 0
        total_failures = sum(result.failure_counts.values())
        total_failed = sum(1 for s in result.samples if not s.passed)
        assert total_failures == total_failed

    @pytest.mark.asyncio
    async def test_empty_batch(self) -> None:
        config = _make_config()
        ev = WeakStrongEvaluator(config)

        result = await ev.evaluate_batch([])
        assert len(result.samples) == 0
        assert result.pass_rate == 0.0
        assert result.total_cost == 0.0
