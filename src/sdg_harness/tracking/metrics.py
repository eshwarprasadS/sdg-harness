from __future__ import annotations

from typing import Any

import structlog
from pydantic import BaseModel

logger = structlog.get_logger()


class MetricEntry(BaseModel):
    iteration_id: int
    cost: float
    tokens_in: int
    tokens_out: int
    latency_seconds: float
    extra: dict[str, Any] = {}


class MetricsTracker:

    def __init__(self) -> None:
        self._entries: list[MetricEntry] = []

    def record(
        self,
        iteration_id: int,
        cost: float,
        tokens_in: int,
        tokens_out: int,
        latency_seconds: float,
        **extra: Any,
    ) -> None:
        entry = MetricEntry(
            iteration_id=iteration_id,
            cost=cost,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_seconds=latency_seconds,
            extra=extra,
        )
        self._entries.append(entry)
        logger.info(
            "metric_recorded",
            iteration_id=iteration_id,
            cost=cost,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_seconds=latency_seconds,
        )

    def summary(self) -> dict[str, float]:
        logger.info("metrics_summary_requested", num_entries=len(self._entries))
        if not self._entries:
            return {
                "total_cost": 0.0,
                "total_tokens_in": 0.0,
                "total_tokens_out": 0.0,
                "mean_latency": 0.0,
                "num_iterations": 0.0,
            }
        total_cost = sum(e.cost for e in self._entries)
        total_tokens_in = sum(e.tokens_in for e in self._entries)
        total_tokens_out = sum(e.tokens_out for e in self._entries)
        mean_latency = sum(e.latency_seconds for e in self._entries) / len(
            self._entries
        )
        return {
            "total_cost": total_cost,
            "total_tokens_in": float(total_tokens_in),
            "total_tokens_out": float(total_tokens_out),
            "mean_latency": mean_latency,
            "num_iterations": float(len(self._entries)),
        }

    def to_dict(self) -> dict[str, Any]:
        logger.info("metrics_to_dict", num_entries=len(self._entries))
        return {
            "entries": [e.model_dump() for e in self._entries],
            "summary": self.summary(),
        }
