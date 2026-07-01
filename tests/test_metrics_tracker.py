from __future__ import annotations

from sdg_harness.tracking.metrics import MetricsTracker


class TestMetricsTracker:
    def test_empty_summary(self) -> None:
        tracker = MetricsTracker()
        s = tracker.summary()
        assert s["total_cost"] == 0.0
        assert s["num_iterations"] == 0.0

    def test_record_and_summary(self) -> None:
        tracker = MetricsTracker()
        tracker.record(
            iteration_id=0,
            cost=1.5,
            tokens_in=100,
            tokens_out=50,
            latency_seconds=2.0,
        )
        tracker.record(
            iteration_id=1,
            cost=2.5,
            tokens_in=200,
            tokens_out=100,
            latency_seconds=3.0,
        )
        s = tracker.summary()
        assert s["total_cost"] == 4.0
        assert s["total_tokens_in"] == 300.0
        assert s["total_tokens_out"] == 150.0
        assert s["mean_latency"] == 2.5
        assert s["num_iterations"] == 2.0

    def test_to_dict(self) -> None:
        tracker = MetricsTracker()
        tracker.record(
            iteration_id=0,
            cost=1.0,
            tokens_in=10,
            tokens_out=5,
            latency_seconds=1.0,
        )
        d = tracker.to_dict()
        assert "entries" in d
        assert "summary" in d
        assert len(d["entries"]) == 1
        assert d["entries"][0]["iteration_id"] == 0

    def test_record_with_extra(self) -> None:
        tracker = MetricsTracker()
        tracker.record(
            iteration_id=0,
            cost=0.0,
            tokens_in=0,
            tokens_out=0,
            latency_seconds=0.0,
            model="gpt-4",
        )
        d = tracker.to_dict()
        assert d["entries"][0]["extra"]["model"] == "gpt-4"
