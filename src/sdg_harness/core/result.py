from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class IterationResult(BaseModel):
    metrics: dict[str, float]
    artifacts: dict[str, str]
    data_samples: list[dict[str, Any]]
    training_signals: dict[str, Any]
    cost: float
    duration_seconds: float
