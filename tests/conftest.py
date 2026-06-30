from __future__ import annotations

import pytest

from sdg_harness.core.analysis import AnalysisReport
from sdg_harness.core.config import IterationConfig
from sdg_harness.core.proposal import Proposal
from sdg_harness.core.result import IterationResult


@pytest.fixture
def sample_config() -> IterationConfig:
    return IterationConfig(
        config={"temperature": 0.7, "num_samples": 100, "model": "gpt-4"},
        mutable_keys=["temperature", "num_samples"],
        fixed_keys=["model"],
        metadata={"source": "test"},
    )


@pytest.fixture
def sample_result() -> IterationResult:
    return IterationResult(
        metrics={"accuracy": 0.85, "diversity": 0.72},
        artifacts={"output_path": "/tmp/samples.jsonl"},
        data_samples=[{"input": "q1", "output": "a1"}],
        training_signals={"loss": 0.45},
        cost=1.25,
        duration_seconds=30.0,
    )


@pytest.fixture
def sample_analysis() -> AnalysisReport:
    return AnalysisReport(
        summary="Iteration produced reasonable quality data",
        data_quality_issues=["low diversity in math domain"],
        training_issues=[],
        eval_breakdown={"accuracy": 0.85, "diversity": 0.72},
        strengths=["high accuracy"],
        weaknesses=["low diversity"],
        root_causes=["narrow prompt templates"],
        suggested_directions=["broaden prompt variety"],
    )


@pytest.fixture
def sample_proposal() -> Proposal:
    return Proposal(
        changes={"temperature": 0.9},
        rationale={"temperature": "increase diversity by raising temperature"},
        expected_effect="higher output diversity at slight accuracy cost",
        risk="low",
    )
