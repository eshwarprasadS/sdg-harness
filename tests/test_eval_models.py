from __future__ import annotations

import pytest

from sdg_harness.eval.models import (
    FailureCategory,
    L1BatchResult,
    L1Config,
    L1Thresholds,
    SampleResult,
)


class TestL1Thresholds:
    def test_defaults(self) -> None:
        t = L1Thresholds()
        assert t.weak_avg_max == 0.65
        assert t.strong_avg_min == 0.60
        assert t.gap_min == 0.20

    def test_custom_values(self) -> None:
        t = L1Thresholds(weak_avg_max=0.70, strong_avg_min=0.50, gap_min=0.15)
        assert t.weak_avg_max == 0.70
        assert t.strong_avg_min == 0.50
        assert t.gap_min == 0.15

    def test_roundtrip(self) -> None:
        t = L1Thresholds(weak_avg_max=0.55, strong_avg_min=0.70, gap_min=0.25)
        restored = L1Thresholds.model_validate(t.model_dump())
        assert restored == t


class TestL1Config:
    def test_required_fields(self) -> None:
        cfg = L1Config(weak_model="gpt-3.5-turbo", strong_model="gpt-4o")
        assert cfg.weak_model == "gpt-3.5-turbo"
        assert cfg.strong_model == "gpt-4o"

    def test_defaults(self) -> None:
        cfg = L1Config(weak_model="weak", strong_model="strong")
        assert cfg.num_rollouts == 3
        assert cfg.temperature == 1.0
        assert cfg.thresholds == L1Thresholds()
        assert cfg.max_cost == 5.0

    def test_custom_thresholds(self) -> None:
        t = L1Thresholds(weak_avg_max=0.50, strong_avg_min=0.80, gap_min=0.30)
        cfg = L1Config(weak_model="w", strong_model="s", thresholds=t)
        assert cfg.thresholds.weak_avg_max == 0.50

    def test_roundtrip(self) -> None:
        cfg = L1Config(
            weak_model="weak",
            strong_model="strong",
            num_rollouts=5,
            temperature=0.7,
            max_cost=10.0,
        )
        restored = L1Config.model_validate(cfg.model_dump())
        assert restored == cfg

    def test_missing_required_raises(self) -> None:
        with pytest.raises(Exception):
            L1Config()  # type: ignore[call-arg]


class TestFailureCategory:
    def test_enum_values(self) -> None:
        assert FailureCategory.TOO_EASY == "TOO_EASY"
        assert FailureCategory.FAILED_ON_STRONG == "FAILED_ON_STRONG"
        assert FailureCategory.QUALITY_FAIL == "QUALITY_FAIL"

    def test_is_str(self) -> None:
        assert isinstance(FailureCategory.TOO_EASY, str)

    def test_from_value(self) -> None:
        assert FailureCategory("TOO_EASY") is FailureCategory.TOO_EASY

    def test_all_members(self) -> None:
        assert len(FailureCategory) == 3


class TestSampleResult:
    def test_passing_sample(self) -> None:
        sr = SampleResult(
            sample_id="s1",
            prompt="What is 2+2?",
            weak_scores=[0.0, 0.0, 1.0],
            strong_scores=[1.0, 1.0, 1.0],
            weak_acc=1 / 3,
            strong_acc=1.0,
            gap=2 / 3,
            passed=True,
            failure_category=None,
        )
        assert sr.passed is True
        assert sr.failure_category is None

    def test_failing_sample(self) -> None:
        sr = SampleResult(
            sample_id="s2",
            prompt="easy question",
            weak_scores=[1.0, 1.0, 1.0],
            strong_scores=[1.0, 1.0, 1.0],
            weak_acc=1.0,
            strong_acc=1.0,
            gap=0.0,
            passed=False,
            failure_category=FailureCategory.TOO_EASY,
        )
        assert sr.passed is False
        assert sr.failure_category == FailureCategory.TOO_EASY

    def test_roundtrip(self) -> None:
        sr = SampleResult(
            sample_id="s3",
            prompt="test",
            weak_scores=[0.0, 1.0],
            strong_scores=[1.0, 0.0],
            weak_acc=0.5,
            strong_acc=0.5,
            gap=0.0,
            passed=False,
            failure_category=FailureCategory.QUALITY_FAIL,
        )
        data = sr.model_dump()
        restored = SampleResult.model_validate(data)
        assert restored == sr
        assert restored.failure_category is FailureCategory.QUALITY_FAIL

    def test_serialization_with_none_category(self) -> None:
        sr = SampleResult(
            sample_id="s4",
            prompt="p",
            weak_scores=[0.0],
            strong_scores=[1.0],
            weak_acc=0.0,
            strong_acc=1.0,
            gap=1.0,
            passed=True,
        )
        data = sr.model_dump()
        assert data["failure_category"] is None
        restored = SampleResult.model_validate(data)
        assert restored.failure_category is None


class TestL1BatchResult:
    def _make_sample(
        self,
        sid: str,
        passed: bool,
        category: FailureCategory | None = None,
    ) -> SampleResult:
        return SampleResult(
            sample_id=sid,
            prompt=f"prompt_{sid}",
            weak_scores=[0.0],
            strong_scores=[1.0],
            weak_acc=0.0,
            strong_acc=1.0,
            gap=1.0,
            passed=passed,
            failure_category=category,
        )

    def test_full_batch(self) -> None:
        samples = [
            self._make_sample("a", True),
            self._make_sample("b", False, FailureCategory.TOO_EASY),
            self._make_sample("c", False, FailureCategory.FAILED_ON_STRONG),
        ]
        batch = L1BatchResult(
            samples=samples,
            weak_avg=0.3,
            strong_avg=0.7,
            gap_avg=0.4,
            pass_rate=1 / 3,
            failure_counts={"TOO_EASY": 1, "FAILED_ON_STRONG": 1},
            total_cost=0.05,
            duration_seconds=12.5,
        )
        assert len(batch.samples) == 3
        assert batch.failure_counts["TOO_EASY"] == 1

    def test_roundtrip(self) -> None:
        batch = L1BatchResult(
            samples=[self._make_sample("x", True)],
            weak_avg=0.1,
            strong_avg=0.9,
            gap_avg=0.8,
            pass_rate=1.0,
            failure_counts={},
            total_cost=0.01,
            duration_seconds=1.0,
        )
        data = batch.model_dump()
        restored = L1BatchResult.model_validate(data)
        assert restored == batch

    def test_empty_batch(self) -> None:
        batch = L1BatchResult(
            samples=[],
            weak_avg=0.0,
            strong_avg=0.0,
            gap_avg=0.0,
            pass_rate=0.0,
            failure_counts={},
            total_cost=0.0,
            duration_seconds=0.0,
        )
        assert len(batch.samples) == 0
        assert batch.pass_rate == 0.0
