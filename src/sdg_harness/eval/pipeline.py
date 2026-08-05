from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel


def _compute_checksum(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


class PipelineArtifact(BaseModel):
    type: str
    path: Path
    metadata: dict[str, Any] = {}
    checksum: str

    @classmethod
    def from_path(
        cls,
        type: str,
        path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> PipelineArtifact:
        return cls(
            type=type,
            path=path,
            metadata=metadata or {},
            checksum=_compute_checksum(path),
        )

    def verify_checksum(self) -> bool:
        return _compute_checksum(self.path) == self.checksum


class StageStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class StageResult(BaseModel):
    stage_name: str
    status: StageStatus
    artifact: PipelineArtifact | None = None
    duration_seconds: float
    error: str | None = None


class Stage:
    def __init__(
        self,
        name: str,
        run_fn: Callable[[dict[str, StageResult]], PipelineArtifact],
        requires: list[str] | None = None,
        condition: Callable[[dict[str, StageResult]], bool] | None = None,
    ) -> None:
        self.name = name
        self.run_fn = run_fn
        self.requires = requires or []
        self.condition = condition


class Pipeline:
    def __init__(self, stages: list[Stage]) -> None:
        self._stages = stages
        self._log = structlog.get_logger().bind(component="pipeline")
        self._validate_topology()

    def _validate_topology(self) -> None:
        seen: set[str] = set()
        for stage in self._stages:
            for req in stage.requires:
                if req not in seen:
                    raise ValueError(
                        f"Stage '{stage.name}' requires '{req}' which is "
                        f"not defined before it"
                    )
            if stage.name in seen:
                raise ValueError(f"Duplicate stage name: '{stage.name}'")
            seen.add(stage.name)

    def run(
        self, initial_inputs: dict[str, PipelineArtifact]
    ) -> dict[str, StageResult]:
        context: dict[str, StageResult] = {}
        for name, artifact in initial_inputs.items():
            context[name] = StageResult(
                stage_name=name,
                status=StageStatus.SUCCEEDED,
                artifact=artifact,
                duration_seconds=0.0,
            )

        self._log.info("pipeline_start", num_stages=len(self._stages))

        stage_results: dict[str, StageResult] = {}

        for stage in self._stages:
            skip_reason: str | None = None
            for req in stage.requires:
                if (
                    req not in context
                    or context[req].status != StageStatus.SUCCEEDED
                ):
                    skip_reason = f"required stage '{req}' did not succeed"
                    break

            if skip_reason:
                result = StageResult(
                    stage_name=stage.name,
                    status=StageStatus.SKIPPED,
                    duration_seconds=0.0,
                    error=skip_reason,
                )
                self._log.info(
                    "stage_skipped", stage=stage.name, reason=skip_reason
                )
                context[stage.name] = result
                stage_results[stage.name] = result
                continue

            if stage.condition is not None and not stage.condition(context):
                reason = "condition returned False"
                result = StageResult(
                    stage_name=stage.name,
                    status=StageStatus.SKIPPED,
                    duration_seconds=0.0,
                    error=reason,
                )
                self._log.info(
                    "stage_skipped", stage=stage.name, reason=reason
                )
                context[stage.name] = result
                stage_results[stage.name] = result
                continue

            self._log.info("stage_start", stage=stage.name)
            start = time.monotonic()

            try:
                produced = stage.run_fn(context)
                elapsed = time.monotonic() - start
                result = StageResult(
                    stage_name=stage.name,
                    status=StageStatus.SUCCEEDED,
                    artifact=produced,
                    duration_seconds=elapsed,
                )
                self._log.info(
                    "stage_complete",
                    stage=stage.name,
                    status="SUCCEEDED",
                    duration_seconds=elapsed,
                )
            except Exception as exc:
                elapsed = time.monotonic() - start
                result = StageResult(
                    stage_name=stage.name,
                    status=StageStatus.FAILED,
                    duration_seconds=elapsed,
                    error=str(exc),
                )
                self._log.info(
                    "stage_complete",
                    stage=stage.name,
                    status="FAILED",
                    duration_seconds=elapsed,
                    error=str(exc),
                )

            context[stage.name] = result
            stage_results[stage.name] = result

        counts = {s.value: 0 for s in StageStatus}
        for r in stage_results.values():
            counts[r.status.value] += 1

        self._log.info("pipeline_complete", **counts)

        return stage_results
