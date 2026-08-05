from __future__ import annotations

import json
from pathlib import Path

import pytest

from sdg_harness.eval.pipeline import (
    Pipeline,
    PipelineArtifact,
    Stage,
    StageResult,
    StageStatus,
)


def _noop(results: dict[str, StageResult]) -> PipelineArtifact:
    raise NotImplementedError


def _make_artifact(tmp_path: Path, name: str = "test.json") -> PipelineArtifact:
    fpath = tmp_path / name
    fpath.write_text(json.dumps({"test": "data"}))
    return PipelineArtifact.from_path(type="test", path=fpath)


class TestTopologicalValidation:
    def test_valid_linear_order(self) -> None:
        stages = [
            Stage(name="a", run_fn=_noop),
            Stage(name="b", run_fn=_noop, requires=["a"]),
            Stage(name="c", run_fn=_noop, requires=["b"]),
        ]
        Pipeline(stages)

    def test_valid_diamond_dependency(self) -> None:
        stages = [
            Stage(name="a", run_fn=_noop),
            Stage(name="b", run_fn=_noop, requires=["a"]),
            Stage(name="c", run_fn=_noop, requires=["a"]),
            Stage(name="d", run_fn=_noop, requires=["b", "c"]),
        ]
        Pipeline(stages)

    def test_reject_missing_dependency(self) -> None:
        stages = [
            Stage(name="b", run_fn=_noop, requires=["a"]),
        ]
        with pytest.raises(ValueError, match="requires 'a'"):
            Pipeline(stages)

    def test_reject_reverse_order(self) -> None:
        stages = [
            Stage(name="b", run_fn=_noop, requires=["a"]),
            Stage(name="a", run_fn=_noop),
        ]
        with pytest.raises(ValueError, match="requires 'a'"):
            Pipeline(stages)

    def test_reject_circular_dependency(self) -> None:
        stages = [
            Stage(name="a", run_fn=_noop, requires=["b"]),
            Stage(name="b", run_fn=_noop, requires=["a"]),
        ]
        with pytest.raises(ValueError):
            Pipeline(stages)

    def test_reject_self_reference(self) -> None:
        stages = [
            Stage(name="a", run_fn=_noop, requires=["a"]),
        ]
        with pytest.raises(ValueError):
            Pipeline(stages)

    def test_reject_duplicate_stage_name(self) -> None:
        stages = [
            Stage(name="a", run_fn=_noop),
            Stage(name="a", run_fn=_noop),
        ]
        with pytest.raises(ValueError, match="Duplicate"):
            Pipeline(stages)


class TestStageExecution:
    def test_sequential_execution(self, tmp_path: Path) -> None:
        execution_order: list[str] = []
        fpath = tmp_path / "artifact.json"
        fpath.write_text("{}")

        def stage_a(results: dict[str, StageResult]) -> PipelineArtifact:
            execution_order.append("a")
            return PipelineArtifact.from_path(type="a", path=fpath)

        def stage_b(results: dict[str, StageResult]) -> PipelineArtifact:
            execution_order.append("b")
            return PipelineArtifact.from_path(type="b", path=fpath)

        stages = [
            Stage(name="a", run_fn=stage_a),
            Stage(name="b", run_fn=stage_b, requires=["a"]),
        ]
        pipeline = Pipeline(stages)
        results = pipeline.run({})

        assert execution_order == ["a", "b"]
        assert results["a"].status == StageStatus.SUCCEEDED
        assert results["b"].status == StageStatus.SUCCEEDED

    def test_artifact_passing_between_stages(self, tmp_path: Path) -> None:
        fpath_a = tmp_path / "a.json"
        fpath_a.write_text(json.dumps({"source": "a"}))
        fpath_b = tmp_path / "b.json"
        fpath_b.write_text(json.dumps({"source": "b"}))

        def stage_a(results: dict[str, StageResult]) -> PipelineArtifact:
            return PipelineArtifact.from_path(
                type="dataset", path=fpath_a, metadata={"key": "value_a"}
            )

        received: list[PipelineArtifact | None] = []

        def stage_b(results: dict[str, StageResult]) -> PipelineArtifact:
            received.append(results["a"].artifact)
            return PipelineArtifact.from_path(type="model", path=fpath_b)

        stages = [
            Stage(name="a", run_fn=stage_a),
            Stage(name="b", run_fn=stage_b, requires=["a"]),
        ]
        Pipeline(stages).run({})

        assert received[0] is not None
        assert received[0].type == "dataset"
        assert received[0].metadata == {"key": "value_a"}

    def test_initial_inputs_accessible_to_stages(self, tmp_path: Path) -> None:
        input_path = tmp_path / "input.json"
        input_path.write_text(json.dumps({"input": True}))
        initial = PipelineArtifact.from_path(type="dataset", path=input_path)

        seen: list[dict[str, StageResult]] = []
        fpath = tmp_path / "out.json"
        fpath.write_text("{}")

        def stage_a(results: dict[str, StageResult]) -> PipelineArtifact:
            seen.append(dict(results))
            return PipelineArtifact.from_path(type="output", path=fpath)

        Pipeline([Stage(name="a", run_fn=stage_a)]).run({"seed": initial})

        assert "seed" in seen[0]
        assert seen[0]["seed"].artifact == initial


class TestConditionalExecution:
    def test_condition_true_runs_stage(self, tmp_path: Path) -> None:
        fpath = tmp_path / "data.json"
        fpath.write_text("{}")

        stages = [
            Stage(
                name="runs",
                run_fn=lambda r: PipelineArtifact.from_path(type="x", path=fpath),
                condition=lambda r: True,
            ),
        ]
        results = Pipeline(stages).run({})
        assert results["runs"].status == StageStatus.SUCCEEDED

    def test_condition_false_skips_stage(self) -> None:
        def should_not_run(results: dict[str, StageResult]) -> PipelineArtifact:
            raise AssertionError("should not run")

        stages = [
            Stage(name="skipped", run_fn=should_not_run, condition=lambda r: False),
        ]
        results = Pipeline(stages).run({})
        assert results["skipped"].status == StageStatus.SKIPPED

    def test_low_pass_rate_skips_training(self, tmp_path: Path) -> None:
        fpath = tmp_path / "data.json"
        fpath.write_text("{}")

        def l1_eval(results: dict[str, StageResult]) -> PipelineArtifact:
            return PipelineArtifact.from_path(
                type="eval", path=fpath, metadata={"pass_rate": 0.50}
            )

        def train(results: dict[str, StageResult]) -> PipelineArtifact:
            return PipelineArtifact.from_path(type="model", path=fpath)

        def should_train(results: dict[str, StageResult]) -> bool:
            l1 = results.get("l1_eval")
            if l1 is None or l1.artifact is None:
                return False
            return l1.artifact.metadata.get("pass_rate", 0) >= 0.70

        stages = [
            Stage(name="l1_eval", run_fn=l1_eval),
            Stage(
                name="train",
                run_fn=train,
                requires=["l1_eval"],
                condition=should_train,
            ),
        ]
        results = Pipeline(stages).run({})

        assert results["l1_eval"].status == StageStatus.SUCCEEDED
        assert results["train"].status == StageStatus.SKIPPED

    def test_high_pass_rate_runs_training(self, tmp_path: Path) -> None:
        fpath = tmp_path / "data.json"
        fpath.write_text("{}")

        def l1_eval(results: dict[str, StageResult]) -> PipelineArtifact:
            return PipelineArtifact.from_path(
                type="eval", path=fpath, metadata={"pass_rate": 0.85}
            )

        def train(results: dict[str, StageResult]) -> PipelineArtifact:
            return PipelineArtifact.from_path(type="model", path=fpath)

        def should_train(results: dict[str, StageResult]) -> bool:
            l1 = results.get("l1_eval")
            if l1 is None or l1.artifact is None:
                return False
            return l1.artifact.metadata.get("pass_rate", 0) >= 0.70

        stages = [
            Stage(name="l1_eval", run_fn=l1_eval),
            Stage(
                name="train",
                run_fn=train,
                requires=["l1_eval"],
                condition=should_train,
            ),
        ]
        results = Pipeline(stages).run({})

        assert results["l1_eval"].status == StageStatus.SUCCEEDED
        assert results["train"].status == StageStatus.SUCCEEDED


class TestErrorHandling:
    def test_stage_exception_recorded_as_failed(self) -> None:
        def failing(results: dict[str, StageResult]) -> PipelineArtifact:
            raise RuntimeError("stage exploded")

        results = Pipeline([Stage(name="boom", run_fn=failing)]).run({})

        assert results["boom"].status == StageStatus.FAILED
        assert results["boom"].error == "stage exploded"
        assert results["boom"].artifact is None

    def test_failed_stage_causes_dependent_to_skip(self) -> None:
        def failing(results: dict[str, StageResult]) -> PipelineArtifact:
            raise RuntimeError("fail")

        was_called: list[bool] = []

        def dependent(results: dict[str, StageResult]) -> PipelineArtifact:
            was_called.append(True)
            raise NotImplementedError

        stages = [
            Stage(name="a", run_fn=failing),
            Stage(name="b", run_fn=dependent, requires=["a"]),
        ]
        results = Pipeline(stages).run({})

        assert results["a"].status == StageStatus.FAILED
        assert results["b"].status == StageStatus.SKIPPED
        assert was_called == []

    def test_unrelated_stage_runs_after_failure(self, tmp_path: Path) -> None:
        fpath = tmp_path / "data.json"
        fpath.write_text("{}")

        def failing(results: dict[str, StageResult]) -> PipelineArtifact:
            raise RuntimeError("fail")

        def independent(results: dict[str, StageResult]) -> PipelineArtifact:
            return PipelineArtifact.from_path(type="x", path=fpath)

        stages = [
            Stage(name="a", run_fn=failing),
            Stage(name="b", run_fn=independent),
        ]
        results = Pipeline(stages).run({})

        assert results["a"].status == StageStatus.FAILED
        assert results["b"].status == StageStatus.SUCCEEDED

    def test_transitive_skip_on_failure(self) -> None:
        def failing(results: dict[str, StageResult]) -> PipelineArtifact:
            raise RuntimeError("fail")

        stages = [
            Stage(name="a", run_fn=failing),
            Stage(name="b", run_fn=_noop, requires=["a"]),
            Stage(name="c", run_fn=_noop, requires=["b"]),
        ]
        results = Pipeline(stages).run({})

        assert results["a"].status == StageStatus.FAILED
        assert results["b"].status == StageStatus.SKIPPED
        assert results["c"].status == StageStatus.SKIPPED


class TestPipelineArtifact:
    def test_from_path_computes_checksum(self, tmp_path: Path) -> None:
        fpath = tmp_path / "test.json"
        fpath.write_text('{"hello": "world"}')

        artifact = PipelineArtifact.from_path(type="test", path=fpath)
        assert artifact.checksum != ""
        assert len(artifact.checksum) == 64

    def test_verify_checksum_valid(self, tmp_path: Path) -> None:
        fpath = tmp_path / "test.json"
        fpath.write_text('{"hello": "world"}')

        artifact = PipelineArtifact.from_path(type="test", path=fpath)
        assert artifact.verify_checksum() is True

    def test_verify_checksum_detects_modification(self, tmp_path: Path) -> None:
        fpath = tmp_path / "test.json"
        fpath.write_text('{"hello": "world"}')

        artifact = PipelineArtifact.from_path(type="test", path=fpath)
        fpath.write_text('{"hello": "changed"}')
        assert artifact.verify_checksum() is False

    def test_metadata_stored(self, tmp_path: Path) -> None:
        fpath = tmp_path / "test.json"
        fpath.write_text("{}")

        artifact = PipelineArtifact.from_path(
            type="dataset", path=fpath, metadata={"rows": 100}
        )
        assert artifact.metadata == {"rows": 100}
        assert artifact.type == "dataset"
