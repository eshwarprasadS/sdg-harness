from __future__ import annotations

import random
import time
from typing import Any

import structlog
from pydantic import BaseModel

from sdg_harness.core.config import IterationConfig
from sdg_harness.core.result import IterationResult
from sdg_harness.inner_loop.base import InnerLoopRunner

logger = structlog.get_logger()


class MathQuestion(BaseModel):
    question: str
    expected_answer: float


class MathQAOutput(BaseModel):
    questions: list[MathQuestion]
    num_correct: int
    num_total: int
    accuracy: float


class ToyTaskRunner(InnerLoopRunner):

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._metrics: dict[str, float] = {}

    def setup(self) -> None:
        logger.info("toy_task_setup")
        self._metrics = {}

    def run_iteration(self, config: IterationConfig) -> MathQAOutput:
        num_samples = int(config.config.get("num_samples", 10))
        difficulty = float(config.config.get("difficulty", 1.0))

        questions: list[MathQuestion] = []
        for _ in range(num_samples):
            a = self._rng.randint(1, int(10 * difficulty))
            b = self._rng.randint(1, int(10 * difficulty))
            op = self._rng.choice(["+", "-", "*"])
            if op == "+":
                answer = float(a + b)
            elif op == "-":
                answer = float(a - b)
            else:
                answer = float(a * b)
            questions.append(
                MathQuestion(question=f"{a} {op} {b}", expected_answer=answer)
            )

        noise = float(config.config.get("noise", 0.1))
        num_correct = sum(
            1 for _ in questions if self._rng.random() > noise
        )
        accuracy = num_correct / max(num_total := len(questions), 1)

        return MathQAOutput(
            questions=questions,
            num_correct=num_correct,
            num_total=num_total,
            accuracy=accuracy,
        )

    def teardown(self) -> None:
        logger.info("toy_task_teardown")

    def get_metrics(self) -> dict[str, float]:
        return dict(self._metrics)

    def run(self, config: IterationConfig) -> IterationResult:
        logger.info("toy_task_run_started")
        self.setup()
        start = time.monotonic()

        output = self.run_iteration(config)

        elapsed = time.monotonic() - start
        self._metrics = {
            "accuracy": output.accuracy,
            "num_correct": float(output.num_correct),
            "num_total": float(output.num_total),
        }

        samples: list[dict[str, Any]] = [
            {"question": q.question, "expected_answer": q.expected_answer}
            for q in output.questions[:5]
        ]

        self.teardown()

        logger.info(
            "toy_task_run_completed",
            accuracy=output.accuracy,
            duration=elapsed,
        )

        return IterationResult(
            metrics=self._metrics,
            artifacts={},
            data_samples=samples,
            training_signals={"accuracy": output.accuracy},
            cost=0.0,
            duration_seconds=elapsed,
        )

    def validate_config(self, config: IterationConfig) -> list[str]:
        errors: list[str] = []
        num_samples = config.config.get("num_samples")
        if num_samples is not None:
            try:
                val = int(num_samples)
                if val <= 0:
                    errors.append("num_samples must be positive")
            except (TypeError, ValueError):
                errors.append("num_samples must be an integer")

        difficulty = config.config.get("difficulty")
        if difficulty is not None:
            try:
                val_f = float(difficulty)
                if val_f <= 0:
                    errors.append("difficulty must be positive")
            except (TypeError, ValueError):
                errors.append("difficulty must be a number")

        return errors
