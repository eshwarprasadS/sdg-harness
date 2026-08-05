"""3-stage pipeline: sdg -> l1_eval -> train (conditional on l1 pass_rate)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from sdg_harness.eval.pipeline import (
    Pipeline,
    PipelineArtifact,
    Stage,
    StageResult,
)


def sdg_stage(results: dict[str, StageResult]) -> PipelineArtifact:
    samples = [
        {"sample_id": f"s{i}", "prompt": f"Question {i}", "expected_answer": str(i)}
        for i in range(10)
    ]
    tmpdir = Path(tempfile.mkdtemp())
    path = tmpdir / "samples.jsonl"
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    return PipelineArtifact.from_path(
        type="dataset", path=path, metadata={"num_samples": len(samples)}
    )


def l1_eval_stage(results: dict[str, StageResult]) -> PipelineArtifact:
    mock_pass_rate = 0.75
    eval_result = {
        "pass_rate": mock_pass_rate,
        "weak_avg": 0.40,
        "strong_avg": 0.80,
        "gap_avg": 0.40,
    }
    tmpdir = Path(tempfile.mkdtemp())
    path = tmpdir / "l1_result.json"
    path.write_text(json.dumps(eval_result, indent=2))
    return PipelineArtifact.from_path(
        type="eval_result", path=path, metadata={"pass_rate": mock_pass_rate}
    )


def should_train(results: dict[str, StageResult]) -> bool:
    l1 = results.get("l1_eval")
    if l1 is None or l1.artifact is None:
        return False
    return l1.artifact.metadata.get("pass_rate", 0) >= 0.70


def train_stage(results: dict[str, StageResult]) -> PipelineArtifact:
    checkpoint = {"model": "fine-tuned", "epochs": 3, "loss": 0.42}
    tmpdir = Path(tempfile.mkdtemp())
    path = tmpdir / "checkpoint.json"
    path.write_text(json.dumps(checkpoint, indent=2))
    return PipelineArtifact.from_path(
        type="model", path=path, metadata={"training_metrics": checkpoint}
    )


def main() -> None:
    stages = [
        Stage(name="sdg", run_fn=sdg_stage),
        Stage(name="l1_eval", run_fn=l1_eval_stage, requires=["sdg"]),
        Stage(
            name="train",
            run_fn=train_stage,
            requires=["l1_eval"],
            condition=should_train,
        ),
    ]

    pipeline = Pipeline(stages)
    results = pipeline.run({})

    for name, result in results.items():
        print(f"  {name}: {result.status.value} ({result.duration_seconds:.2f}s)")
        if result.artifact:
            print(f"    artifact: {result.artifact.path}")


if __name__ == "__main__":
    main()
