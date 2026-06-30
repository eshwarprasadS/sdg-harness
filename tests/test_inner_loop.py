from __future__ import annotations

import pytest

from sdg_harness.core.config import IterationConfig
from sdg_harness.core.result import IterationResult
from sdg_harness.inner_loop.base import InnerLoopRunner


class DummyRunner(InnerLoopRunner):
    def run(self, config: IterationConfig) -> IterationResult:
        return IterationResult(
            metrics={"score": 1.0},
            artifacts={},
            data_samples=[],
            training_signals={},
            cost=0.0,
            duration_seconds=0.0,
        )

    def validate_config(self, config: IterationConfig) -> list[str]:
        errors = []
        if "model" not in config.config:
            errors.append("missing required key: model")
        return errors


class TestInnerLoopRunner:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            InnerLoopRunner()  # type: ignore[abstract]

    def test_subclass_run(self, sample_config: IterationConfig) -> None:
        runner = DummyRunner()
        result = runner.run(sample_config)
        assert result.metrics["score"] == 1.0

    def test_subclass_validate(self, sample_config: IterationConfig) -> None:
        runner = DummyRunner()
        errors = runner.validate_config(sample_config)
        assert errors == []

    def test_validate_missing_key(self) -> None:
        runner = DummyRunner()
        config = IterationConfig(
            config={"temperature": 0.7},
            mutable_keys=["temperature"],
            fixed_keys=[],
        )
        errors = runner.validate_config(config)
        assert "missing required key: model" in errors
