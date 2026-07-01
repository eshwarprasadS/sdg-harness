from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any

import structlog

from sdg_harness.core.config import IterationConfig
from sdg_harness.core.result import IterationResult
from sdg_harness.inner_loop.base import InnerLoopRunner

logger = structlog.get_logger()


class GenericRunner(InnerLoopRunner):

    def __init__(self, command: list[str], timeout: int = 300) -> None:
        self._command = command
        self._timeout = timeout
        self._metrics: dict[str, float] = {}

    def run_iteration(
        self, config: IterationConfig
    ) -> subprocess.CompletedProcess[str]:
        env_vars = {f"SDG_{k.upper()}": str(v) for k, v in config.config.items()}
        logger.info(
            "generic_run_iteration",
            command=self._command,
            env_vars=list(env_vars.keys()),
        )
        result = subprocess.run(
            self._command,
            capture_output=True,
            text=True,
            timeout=self._timeout,
            env={**os.environ, **env_vars},
            check=False,
        )
        return result

    def get_metrics(self) -> dict[str, float]:
        return dict(self._metrics)

    def run(self, config: IterationConfig) -> IterationResult:
        logger.info("generic_run_started", command=self._command)
        start = time.monotonic()

        try:
            proc = self.run_iteration(config)
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            logger.warning(
                "generic_run_timed_out",
                command=self._command,
                timeout=self._timeout,
                duration=elapsed,
            )
            self._metrics = {"exit_code": -1.0, "duration": elapsed}
            return IterationResult(
                metrics=dict(self._metrics),
                artifacts={"timed_out": f"timeout after {self._timeout}s"},
                data_samples=[],
                training_signals={},
                cost=0.0,
                duration_seconds=elapsed,
            )

        elapsed = time.monotonic() - start
        self._metrics = {
            "exit_code": float(proc.returncode),
            "duration": elapsed,
        }

        metrics = dict(self._metrics)
        artifacts: dict[str, str] = {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }

        stdout_metrics = self._parse_stdout_metrics(proc.stdout)
        metrics.update(stdout_metrics)

        logger.info(
            "generic_run_completed",
            exit_code=proc.returncode,
            duration=elapsed,
        )

        return IterationResult(
            metrics=metrics,
            artifacts=artifacts,
            data_samples=[],
            training_signals={},
            cost=0.0,
            duration_seconds=elapsed,
        )

    def validate_config(self, config: IterationConfig) -> list[str]:
        errors: list[str] = []
        if not self._command:
            errors.append("command must not be empty")
        return errors

    @staticmethod
    def _parse_stdout_metrics(stdout: str) -> dict[str, float]:
        try:
            data: Any = json.loads(stdout.strip().split("\n")[-1])
            if isinstance(data, dict):
                return {
                    k: float(v) for k, v in data.items() if isinstance(v, (int, float))
                }
        except (json.JSONDecodeError, ValueError, IndexError):
            pass
        return {}
