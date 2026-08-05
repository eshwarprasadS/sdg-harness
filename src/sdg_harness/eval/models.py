from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class FailureCategory(StrEnum):
    TOO_EASY = "TOO_EASY"
    FAILED_ON_STRONG = "FAILED_ON_STRONG"
    QUALITY_FAIL = "QUALITY_FAIL"


class L1Thresholds(BaseModel):
    weak_avg_max: float = 0.65
    strong_avg_min: float = 0.60
    gap_min: float = 0.20


class L1Config(BaseModel):
    weak_model: str
    strong_model: str
    num_rollouts: int = 3
    temperature: float = 1.0
    thresholds: L1Thresholds = L1Thresholds()
    max_cost: float = 5.0


class SampleResult(BaseModel):
    sample_id: str
    prompt: str
    weak_scores: list[float]
    strong_scores: list[float]
    weak_acc: float
    strong_acc: float
    gap: float
    passed: bool
    failure_category: FailureCategory | None = None


class L1BatchResult(BaseModel):
    samples: list[SampleResult]
    weak_avg: float
    strong_avg: float
    gap_avg: float
    pass_rate: float
    failure_counts: dict[str, int]
    total_cost: float
    duration_seconds: float
