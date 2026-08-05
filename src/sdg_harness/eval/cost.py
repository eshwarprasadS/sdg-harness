from __future__ import annotations

import json
from dataclasses import dataclass


class BudgetExceededError(RuntimeError):
    def __init__(self, spent: float, limit: float) -> None:
        self.spent = spent
        self.limit = limit
        super().__init__(f"Budget exceeded: spent ${spent:.4f} >= limit ${limit:.4f}")


@dataclass
class _ModelStats:
    total_cost: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    call_count: int = 0


# Pricing per 1M tokens (input, output) for common litellm model prefixes.
# Falls back to a conservative default for unknown models.
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "claude-3-opus": (15.00, 75.00),
    "claude-3-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku-4": (0.80, 4.00),
}

_DEFAULT_PRICING: tuple[float, float] = (10.00, 30.00)


def _lookup_pricing(model: str) -> tuple[float, float]:
    model_lower = model.lower()
    for prefix, pricing in _PRICING.items():
        if prefix in model_lower:
            return pricing
    return _DEFAULT_PRICING


class CostTracker:
    def __init__(self, max_cost: float) -> None:
        self._max_cost = max_cost
        self._total: float = 0.0
        self._by_model: dict[str, _ModelStats] = {}

    def track(self, model: str, usage: dict[str, int]) -> float:
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        input_price, output_price = _lookup_pricing(model)
        cost = (
            prompt_tokens * input_price + completion_tokens * output_price
        ) / 1_000_000

        self._total += cost

        if model not in self._by_model:
            self._by_model[model] = _ModelStats()
        stats = self._by_model[model]
        stats.total_cost += cost
        stats.prompt_tokens += prompt_tokens
        stats.completion_tokens += completion_tokens
        stats.call_count += 1

        return cost

    def check_budget(self) -> None:
        if self._total >= self._max_cost:
            raise BudgetExceededError(spent=self._total, limit=self._max_cost)

    @property
    def total(self) -> float:
        return self._total

    def summary(self) -> dict[str, dict[str, float | int]]:
        return {
            model: {
                "total_cost": stats.total_cost,
                "prompt_tokens": stats.prompt_tokens,
                "completion_tokens": stats.completion_tokens,
                "call_count": stats.call_count,
            }
            for model, stats in self._by_model.items()
        }

    def to_json(self) -> str:
        return json.dumps(
            {
                "total_cost": self._total,
                "max_cost": self._max_cost,
                "by_model": self.summary(),
            },
            indent=2,
        )
