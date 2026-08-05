from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from sdg_harness.eval.cli import eval_group
from sdg_harness.eval.models import L1BatchResult


def _make_samples_file(samples: list[dict[str, str]]) -> str:
    path = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for s in samples:
        path.write(json.dumps(s) + "\n")
    path.close()
    return path.name


def _make_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    resp.usage = usage
    return resp


class TestEvalL1CLI:
    def test_basic_invocation(self, tmp_path: Path) -> None:
        samples_path = _make_samples_file([
            {"sample_id": "s1", "prompt": "What is 2+2?", "expected_answer": "4"},
        ])
        output_path = str(tmp_path / "result.json")

        weak_resp = _make_response("I think 5")
        strong_resp = _make_response("The answer is 4")

        with patch("sdg_harness.eval.weak_strong.litellm") as mock_llm:
            async def mock_acompletion(model: str, **kwargs: object) -> MagicMock:
                if model == "weak-model":
                    return weak_resp
                return strong_resp

            mock_llm.acompletion = AsyncMock(side_effect=mock_acompletion)

            runner = CliRunner()
            result = runner.invoke(eval_group, [
                "l1",
                "--samples", samples_path,
                "--weak-model", "weak-model",
                "--strong-model", "strong-model",
                "--rollouts", "1",
                "--max-cost", "10.0",
                "--output", output_path,
            ])

        assert result.exit_code == 0, result.output
        data = json.loads(Path(output_path).read_text())
        assert "samples" in data
        assert "pass_rate" in data

    def test_output_is_valid_batch_result(self, tmp_path: Path) -> None:
        samples_path = _make_samples_file([
            {"sample_id": "s1", "prompt": "What is 2+2?", "expected_answer": "4"},
        ])
        output_path = str(tmp_path / "result.json")

        resp = _make_response("answer: 4")

        with patch("sdg_harness.eval.weak_strong.litellm") as mock_llm:
            mock_llm.acompletion = AsyncMock(return_value=resp)

            runner = CliRunner()
            result = runner.invoke(eval_group, [
                "l1",
                "--samples", samples_path,
                "--weak-model", "w",
                "--strong-model", "s",
                "--rollouts", "1",
                "--max-cost", "10.0",
                "--output", output_path,
            ])

        assert result.exit_code == 0, result.output
        data = json.loads(Path(output_path).read_text())
        batch = L1BatchResult.model_validate(data)
        assert len(batch.samples) == 1

    def test_missing_samples_file(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(eval_group, [
            "l1",
            "--samples", "/nonexistent/path.jsonl",
            "--weak-model", "w",
            "--strong-model", "s",
            "--output", str(tmp_path / "out.json"),
        ])
        assert result.exit_code != 0

    def test_missing_required_options(self) -> None:
        runner = CliRunner()
        result = runner.invoke(eval_group, ["l1"])
        assert result.exit_code != 0
