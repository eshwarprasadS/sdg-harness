from __future__ import annotations

import sys

from sdg_harness.core.config import IterationConfig
from sdg_harness.inner_loop.generic import GenericRunner


class TestGenericRunner:
    def test_run_echo(self) -> None:
        runner = GenericRunner(command=[sys.executable, "-c", "print('hello')"])
        config = IterationConfig(
            config={},
            mutable_keys=[],
            fixed_keys=[],
        )
        result = runner.run(config)
        assert result.metrics["exit_code"] == 0.0
        assert "hello" in result.artifacts["stdout"]

    def test_run_captures_stderr(self) -> None:
        runner = GenericRunner(
            command=[sys.executable, "-c", "import sys; sys.stderr.write('err\\n')"]
        )
        config = IterationConfig(
            config={},
            mutable_keys=[],
            fixed_keys=[],
        )
        result = runner.run(config)
        assert "err" in result.artifacts["stderr"]

    def test_run_nonzero_exit(self) -> None:
        runner = GenericRunner(command=[sys.executable, "-c", "raise SystemExit(1)"])
        config = IterationConfig(
            config={},
            mutable_keys=[],
            fixed_keys=[],
        )
        result = runner.run(config)
        assert result.metrics["exit_code"] == 1.0

    def test_run_parses_json_metrics(self) -> None:
        script = 'import json; print(json.dumps({"score": 0.95, "count": 10}))'
        runner = GenericRunner(command=[sys.executable, "-c", script])
        config = IterationConfig(
            config={},
            mutable_keys=[],
            fixed_keys=[],
        )
        result = runner.run(config)
        assert result.metrics.get("score") == 0.95
        assert result.metrics.get("count") == 10.0

    def test_get_metrics_after_run(self) -> None:
        runner = GenericRunner(command=[sys.executable, "-c", "pass"])
        config = IterationConfig(
            config={},
            mutable_keys=[],
            fixed_keys=[],
        )
        runner.run(config)
        metrics = runner.get_metrics()
        assert "exit_code" in metrics
        assert "duration" in metrics

    def test_validate_config_valid(self) -> None:
        runner = GenericRunner(command=["echo", "hi"])
        config = IterationConfig(
            config={},
            mutable_keys=[],
            fixed_keys=[],
        )
        errors = runner.validate_config(config)
        assert errors == []

    def test_validate_config_empty_command(self) -> None:
        runner = GenericRunner(command=[])
        config = IterationConfig(
            config={},
            mutable_keys=[],
            fixed_keys=[],
        )
        errors = runner.validate_config(config)
        assert "command must not be empty" in errors

    def test_run_iteration_returns_completed_process(self) -> None:
        runner = GenericRunner(command=[sys.executable, "-c", "print('ok')"])
        config = IterationConfig(
            config={"key": "val"},
            mutable_keys=["key"],
            fixed_keys=[],
        )
        proc = runner.run_iteration(config)
        assert proc.returncode == 0
        assert "ok" in proc.stdout
