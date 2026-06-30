from __future__ import annotations

from pydantic import BaseModel

from sdg_harness.core.analysis import AnalysisReport
from sdg_harness.core.config import IterationConfig
from sdg_harness.core.proposal import Proposal
from sdg_harness.core.result import IterationResult


class IterationRecord(BaseModel):
    iteration_id: int
    config: IterationConfig
    result: IterationResult | None = None
    analysis: AnalysisReport | None = None
    proposal: Proposal | None = None
    timestamp: str
