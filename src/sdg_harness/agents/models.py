from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from sdg_harness.core.config import IterationConfig
from sdg_harness.core.result import IterationResult


class PipelineInfo(BaseModel):
    description: str
    config_schema: dict[str, Any]
    constraints: list[str] = []


class PlannerResult(BaseModel):
    mutable_parameters: list[str]
    fixed_parameters: list[str]
    search_strategy: str
    rationale: str


class AnalysisContext(BaseModel):
    config: IterationConfig
    result: IterationResult
    data_samples: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []


class ProposalContext(BaseModel):
    config: IterationConfig
    analysis: dict[str, Any]
    constraints: list[str] = []
    history: list[dict[str, Any]] = []


class LLMUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
