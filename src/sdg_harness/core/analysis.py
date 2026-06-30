from __future__ import annotations

from pydantic import BaseModel


class AnalysisReport(BaseModel):
    summary: str
    data_quality_issues: list[str]
    training_issues: list[str]
    eval_breakdown: dict[str, float]
    strengths: list[str]
    weaknesses: list[str]
    root_causes: list[str]
    suggested_directions: list[str]
