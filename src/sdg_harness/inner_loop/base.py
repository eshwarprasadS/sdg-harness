from __future__ import annotations

from abc import ABC, abstractmethod

import structlog

from sdg_harness.core.config import IterationConfig
from sdg_harness.core.result import IterationResult

logger = structlog.get_logger()


class InnerLoopRunner(ABC):

    @abstractmethod
    def run(self, config: IterationConfig) -> IterationResult:
        logger.info(
            "inner_loop_run_started",
            config_keys=list(config.config.keys()),
            mutable_keys=config.mutable_keys,
        )

    @abstractmethod
    def validate_config(self, config: IterationConfig) -> list[str]:
        logger.info(
            "inner_loop_validate_started",
            config_keys=list(config.config.keys()),
        )
