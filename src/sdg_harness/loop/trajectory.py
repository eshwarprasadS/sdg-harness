from __future__ import annotations

from pydantic import BaseModel

from sdg_harness.core.record import IterationRecord


class TrajectoryResult(BaseModel):
    iterations: list[IterationRecord]
    best_iteration: int
    best_score: float
    stop_reason: str
    total_cost: float
