from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import structlog

from sdg_harness.core.analysis import AnalysisReport
from sdg_harness.core.config import IterationConfig
from sdg_harness.core.proposal import Proposal
from sdg_harness.core.record import IterationRecord
from sdg_harness.core.result import IterationResult
from sdg_harness.inner_loop.base import InnerLoopRunner
from sdg_harness.logging import bind_request_id, configure_logging
from sdg_harness.loop.trajectory import TrajectoryResult

logger = structlog.get_logger()

AnalyzerCallable = Callable[[IterationConfig, IterationResult], AnalysisReport]
ProposerCallable = Callable[[IterationConfig, AnalysisReport], Proposal]
StoppingCallable = Callable[[list[IterationRecord]], bool]


class LoopController:

    def __init__(
        self,
        runner: InnerLoopRunner,
        analyzer: AnalyzerCallable,
        proposer: ProposerCallable,
        should_stop: StoppingCallable,
        initial_config: IterationConfig,
    ) -> None:
        configure_logging()
        self.runner = runner
        self.analyzer = analyzer
        self.proposer = proposer
        self.should_stop = should_stop
        self.initial_config = initial_config

    def run(
        self,
        task_description: str,
        pipeline_info: dict[str, Any],
    ) -> TrajectoryResult:
        with bind_request_id() as req_id:
            logger.info(
                "loop_started",
                task=task_description,
                pipeline=pipeline_info,
                request_id=req_id,
            )

            records: list[IterationRecord] = []
            best_iteration = 0
            best_score = float("-inf")
            current_config = self.initial_config
            iteration_id = 0

            while not self.should_stop(records):
                logger.info("iteration_started", iteration_id=iteration_id)
                start = time.monotonic()

                result = self.runner.run(current_config)

                elapsed = time.monotonic() - start
                logger.info(
                    "iteration_completed",
                    iteration_id=iteration_id,
                    duration=elapsed,
                    metrics=result.metrics,
                )

                analysis = self.analyzer(current_config, result)
                proposal = self.proposer(current_config, analysis)

                record = IterationRecord(
                    iteration_id=iteration_id,
                    config=current_config,
                    result=result,
                    analysis=analysis,
                    proposal=proposal,
                    timestamp=time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                )
                records.append(record)

                score = sum(result.metrics.values()) / max(
                    len(result.metrics), 1
                )
                if score > best_score:
                    best_score = score
                    best_iteration = iteration_id

                current_config = self._apply_proposal(current_config, proposal)
                iteration_id += 1

            total_cost = sum(
                r.result.cost for r in records if r.result is not None
            )

            logger.info(
                "loop_finished",
                iterations=len(records),
                best_iteration=best_iteration,
                best_score=best_score,
            )

            return TrajectoryResult(
                iterations=records,
                best_iteration=best_iteration,
                best_score=best_score,
                stop_reason="stopping_condition_met",
                total_cost=total_cost,
            )

    def _apply_proposal(
        self,
        config: IterationConfig,
        proposal: Proposal,
    ) -> IterationConfig:
        new_config_dict = dict(config.config)
        for key, value in proposal.changes.items():
            if key in config.mutable_keys:
                new_config_dict[key] = value
        return config.model_copy(update={"config": new_config_dict})
