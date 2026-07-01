from __future__ import annotations

import pytest

from sdg_harness.core.config import IterationConfig
from sdg_harness.core.record import IterationRecord
from sdg_harness.core.result import IterationResult
from sdg_harness.loop.trajectory import TrajectoryResult
from sdg_harness.tracking.storage import TrajectoryStorage


@pytest.fixture
def trajectory() -> TrajectoryResult:
    config = IterationConfig(
        config={"num_samples": 10},
        mutable_keys=["num_samples"],
        fixed_keys=[],
    )
    result = IterationResult(
        metrics={"accuracy": 0.9},
        artifacts={},
        data_samples=[],
        training_signals={},
        cost=0.5,
        duration_seconds=1.0,
    )
    record = IterationRecord(
        iteration_id=0,
        config=config,
        result=result,
        timestamp="2026-07-01T00:00:00Z",
    )
    return TrajectoryResult(
        iterations=[record],
        best_iteration=0,
        best_score=0.9,
        stop_reason="max_iterations",
        total_cost=0.5,
    )


class TestTrajectoryStorage:
    def test_save_and_load(
        self, tmp_path: object, trajectory: TrajectoryResult
    ) -> None:
        storage = TrajectoryStorage(str(tmp_path))
        storage.save("test-run", trajectory)
        loaded = storage.load("test-run")
        assert loaded.best_score == trajectory.best_score
        assert len(loaded.iterations) == 1

    def test_list_runs_empty(self, tmp_path: object) -> None:
        storage = TrajectoryStorage(str(tmp_path))
        assert storage.list_runs() == []

    def test_list_runs(self, tmp_path: object, trajectory: TrajectoryResult) -> None:
        storage = TrajectoryStorage(str(tmp_path))
        storage.save("run-a", trajectory)
        storage.save("run-b", trajectory)
        runs = storage.list_runs()
        assert runs == ["run-a", "run-b"]

    def test_load_nonexistent(self, tmp_path: object) -> None:
        storage = TrajectoryStorage(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            storage.load("nonexistent")

    def test_roundtrip_preserves_data(
        self, tmp_path: object, trajectory: TrajectoryResult
    ) -> None:
        storage = TrajectoryStorage(str(tmp_path))
        storage.save("roundtrip", trajectory)
        loaded = storage.load("roundtrip")
        assert loaded == trajectory
