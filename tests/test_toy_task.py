from __future__ import annotations

from sdg_harness.core.config import IterationConfig
from sdg_harness.inner_loop.toy_task import ToyTaskRunner


class TestToyTaskRunner:
    def test_run_returns_result(self) -> None:
        runner = ToyTaskRunner(seed=42)
        config = IterationConfig(
            config={"num_samples": 5, "difficulty": 1.0, "noise": 0.1},
            mutable_keys=["num_samples", "difficulty", "noise"],
            fixed_keys=[],
        )
        result = runner.run(config)
        assert "accuracy" in result.metrics
        assert "num_correct" in result.metrics
        assert "num_total" in result.metrics
        assert result.metrics["num_total"] == 5.0
        assert result.duration_seconds >= 0
        assert result.cost == 0.0

    def test_run_deterministic_with_seed(self) -> None:
        config = IterationConfig(
            config={"num_samples": 10, "difficulty": 1.0},
            mutable_keys=["num_samples"],
            fixed_keys=[],
        )
        r1 = ToyTaskRunner(seed=99).run(config)
        r2 = ToyTaskRunner(seed=99).run(config)
        assert r1.metrics == r2.metrics
        assert r1.data_samples == r2.data_samples

    def test_run_iteration_output(self) -> None:
        runner = ToyTaskRunner(seed=42)
        config = IterationConfig(
            config={"num_samples": 3, "difficulty": 2.0},
            mutable_keys=["num_samples"],
            fixed_keys=[],
        )
        output = runner.run_iteration(config)
        assert output.num_total == 3
        assert len(output.questions) == 3
        assert 0.0 <= output.accuracy <= 1.0

    def test_get_metrics_after_run(self) -> None:
        runner = ToyTaskRunner(seed=42)
        config = IterationConfig(
            config={"num_samples": 5},
            mutable_keys=["num_samples"],
            fixed_keys=[],
        )
        runner.run(config)
        metrics = runner.get_metrics()
        assert "accuracy" in metrics

    def test_validate_config_valid(self) -> None:
        runner = ToyTaskRunner()
        config = IterationConfig(
            config={"num_samples": 10, "difficulty": 1.5},
            mutable_keys=["num_samples"],
            fixed_keys=[],
        )
        errors = runner.validate_config(config)
        assert errors == []

    def test_validate_config_negative_samples(self) -> None:
        runner = ToyTaskRunner()
        config = IterationConfig(
            config={"num_samples": -1},
            mutable_keys=["num_samples"],
            fixed_keys=[],
        )
        errors = runner.validate_config(config)
        assert any("num_samples" in e for e in errors)

    def test_validate_config_invalid_difficulty(self) -> None:
        runner = ToyTaskRunner()
        config = IterationConfig(
            config={"difficulty": "bad"},
            mutable_keys=[],
            fixed_keys=[],
        )
        errors = runner.validate_config(config)
        assert any("difficulty" in e for e in errors)

    def test_data_samples_capped(self) -> None:
        runner = ToyTaskRunner(seed=42)
        config = IterationConfig(
            config={"num_samples": 20},
            mutable_keys=["num_samples"],
            fixed_keys=[],
        )
        result = runner.run(config)
        assert len(result.data_samples) <= 5
