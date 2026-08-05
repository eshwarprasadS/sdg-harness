from __future__ import annotations

import json

import pytest

from sdg_harness.eval.cost import BudgetExceededError, CostTracker


class TestBudgetExceededError:
    def test_attributes(self) -> None:
        err = BudgetExceededError(spent=5.50, limit=5.00)
        assert err.spent == 5.50
        assert err.limit == 5.00

    def test_is_runtime_error(self) -> None:
        err = BudgetExceededError(spent=1.0, limit=0.5)
        assert isinstance(err, RuntimeError)

    def test_message(self) -> None:
        err = BudgetExceededError(spent=1.2345, limit=1.0000)
        assert "1.2345" in str(err)
        assert "1.0000" in str(err)


class TestCostTracker:
    def test_initial_state(self) -> None:
        ct = CostTracker(max_cost=10.0)
        assert ct.total == 0.0
        assert ct.summary() == {}

    def test_track_single_call(self) -> None:
        ct = CostTracker(max_cost=10.0)
        cost = ct.track("gpt-4o", {"prompt_tokens": 100, "completion_tokens": 50})
        assert cost > 0
        assert ct.total == cost

    def test_track_accumulates(self) -> None:
        ct = CostTracker(max_cost=100.0)
        c1 = ct.track("gpt-4o", {"prompt_tokens": 1000, "completion_tokens": 500})
        c2 = ct.track("gpt-4o", {"prompt_tokens": 2000, "completion_tokens": 1000})
        assert ct.total == pytest.approx(c1 + c2)

    def test_track_multiple_models(self) -> None:
        ct = CostTracker(max_cost=100.0)
        ct.track("gpt-4o", {"prompt_tokens": 100, "completion_tokens": 50})
        ct.track("gpt-3.5-turbo", {"prompt_tokens": 200, "completion_tokens": 100})
        summary = ct.summary()
        assert "gpt-4o" in summary
        assert "gpt-3.5-turbo" in summary
        assert summary["gpt-4o"]["call_count"] == 1
        assert summary["gpt-3.5-turbo"]["call_count"] == 1

    def test_track_missing_tokens_defaults_zero(self) -> None:
        ct = CostTracker(max_cost=10.0)
        cost = ct.track("gpt-4o", {})
        assert cost == 0.0
        assert ct.total == 0.0

    def test_check_budget_under_limit(self) -> None:
        ct = CostTracker(max_cost=10.0)
        ct.track("gpt-4o", {"prompt_tokens": 100, "completion_tokens": 50})
        ct.check_budget()

    def test_check_budget_at_limit_raises(self) -> None:
        ct = CostTracker(max_cost=0.0)
        with pytest.raises(BudgetExceededError) as exc_info:
            ct.check_budget()
        assert exc_info.value.spent == 0.0
        assert exc_info.value.limit == 0.0

    def test_check_budget_over_limit_raises(self) -> None:
        ct = CostTracker(max_cost=0.000001)
        ct.track("gpt-4o", {"prompt_tokens": 1_000_000, "completion_tokens": 500_000})
        with pytest.raises(BudgetExceededError) as exc_info:
            ct.check_budget()
        assert exc_info.value.spent > exc_info.value.limit

    def test_summary_structure(self) -> None:
        ct = CostTracker(max_cost=100.0)
        ct.track("gpt-4o", {"prompt_tokens": 500, "completion_tokens": 200})
        ct.track("gpt-4o", {"prompt_tokens": 300, "completion_tokens": 100})
        summary = ct.summary()
        model_stats = summary["gpt-4o"]
        assert model_stats["prompt_tokens"] == 800
        assert model_stats["completion_tokens"] == 300
        assert model_stats["call_count"] == 2
        assert model_stats["total_cost"] > 0

    def test_to_json_valid(self) -> None:
        ct = CostTracker(max_cost=5.0)
        ct.track("gpt-4o", {"prompt_tokens": 100, "completion_tokens": 50})
        raw = ct.to_json()
        data = json.loads(raw)
        assert "total_cost" in data
        assert "max_cost" in data
        assert "by_model" in data
        assert data["max_cost"] == 5.0
        assert data["total_cost"] == ct.total

    def test_to_json_empty(self) -> None:
        ct = CostTracker(max_cost=1.0)
        data = json.loads(ct.to_json())
        assert data["total_cost"] == 0.0
        assert data["by_model"] == {}

    def test_unknown_model_uses_default_pricing(self) -> None:
        ct = CostTracker(max_cost=100.0)
        usage = {"prompt_tokens": 1000, "completion_tokens": 500}
        cost = ct.track("some-unknown-model", usage)
        assert cost > 0
