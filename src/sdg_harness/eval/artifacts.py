from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel


class SDGArtifact(BaseModel):
    dataset_path: Path
    num_samples: int
    l1_quality_score: float | None = None
    metadata: dict[str, Any] = {}

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text())


class TrainingArtifact(BaseModel):
    checkpoint_path: Path
    adapter_path: Path | None = None
    training_metrics: dict[str, float] = {}
    metadata: dict[str, Any] = {}

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text())


class EvalArtifact(BaseModel):
    results_path: Path
    composite_score: float
    task_scores: dict[str, float] = {}
    metadata: dict[str, Any] = {}

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text())
