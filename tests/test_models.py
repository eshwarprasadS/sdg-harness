from __future__ import annotations

from sdg_harness.core.analysis import AnalysisReport
from sdg_harness.core.config import IterationConfig
from sdg_harness.core.proposal import Proposal
from sdg_harness.core.record import IterationRecord
from sdg_harness.core.result import IterationResult
from sdg_harness.loop.trajectory import TrajectoryResult


class TestIterationConfig:
    def test_roundtrip(self, sample_config: IterationConfig) -> None:
        data = sample_config.model_dump()
        restored = IterationConfig.model_validate(data)
        assert restored == sample_config

    def test_default_metadata(self) -> None:
        cfg = IterationConfig(
            config={"x": 1},
            mutable_keys=["x"],
            fixed_keys=[],
        )
        assert cfg.metadata == {}


class TestIterationResult:
    def test_roundtrip(self, sample_result: IterationResult) -> None:
        data = sample_result.model_dump()
        restored = IterationResult.model_validate(data)
        assert restored == sample_result


class TestAnalysisReport:
    def test_roundtrip(self, sample_analysis: AnalysisReport) -> None:
        data = sample_analysis.model_dump()
        restored = AnalysisReport.model_validate(data)
        assert restored == sample_analysis


class TestProposal:
    def test_roundtrip(self, sample_proposal: Proposal) -> None:
        data = sample_proposal.model_dump()
        restored = Proposal.model_validate(data)
        assert restored == sample_proposal


class TestIterationRecord:
    def test_roundtrip(
        self,
        sample_config: IterationConfig,
        sample_result: IterationResult,
        sample_analysis: AnalysisReport,
        sample_proposal: Proposal,
    ) -> None:
        record = IterationRecord(
            iteration_id=0,
            config=sample_config,
            result=sample_result,
            analysis=sample_analysis,
            proposal=sample_proposal,
            timestamp="2026-06-30T00:00:00Z",
        )
        data = record.model_dump()
        restored = IterationRecord.model_validate(data)
        assert restored == record

    def test_optional_fields(self, sample_config: IterationConfig) -> None:
        record = IterationRecord(
            iteration_id=0,
            config=sample_config,
            timestamp="2026-06-30T00:00:00Z",
        )
        assert record.result is None
        assert record.analysis is None
        assert record.proposal is None
        data = record.model_dump()
        restored = IterationRecord.model_validate(data)
        assert restored == record


class TestTrajectoryResult:
    def test_roundtrip(
        self,
        sample_config: IterationConfig,
        sample_result: IterationResult,
    ) -> None:
        record = IterationRecord(
            iteration_id=0,
            config=sample_config,
            result=sample_result,
            timestamp="2026-06-30T00:00:00Z",
        )
        trajectory = TrajectoryResult(
            iterations=[record],
            best_iteration=0,
            best_score=0.85,
            stop_reason="max_iterations",
            total_cost=1.25,
        )
        data = trajectory.model_dump()
        restored = TrajectoryResult.model_validate(data)
        assert restored == trajectory

    def test_empty_iterations(self) -> None:
        trajectory = TrajectoryResult(
            iterations=[],
            best_iteration=0,
            best_score=0.0,
            stop_reason="no_iterations",
            total_cost=0.0,
        )
        data = trajectory.model_dump()
        restored = TrajectoryResult.model_validate(data)
        assert restored == trajectory
