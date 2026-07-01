from __future__ import annotations

from abc import ABC, abstractmethod

import structlog

from sdg_harness.core.config import IterationConfig
from sdg_harness.core.result import IterationResult

logger = structlog.get_logger()


class InnerLoopRunner(ABC):

    @abstractmethod
    def run(self, config: IterationConfig) -> IterationResult:
        ...

    @abstractmethod
    def validate_config(self, config: IterationConfig) -> list[str]:
        ...
