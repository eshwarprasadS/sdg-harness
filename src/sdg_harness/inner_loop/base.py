from __future__ import annotations

from abc import ABC, abstractmethod

from sdg_harness.core.config import IterationConfig
from sdg_harness.core.result import IterationResult


class InnerLoopRunner(ABC):

    @abstractmethod
    def run(self, config: IterationConfig) -> IterationResult:
        ...

    @abstractmethod
    def validate_config(self, config: IterationConfig) -> list[str]:
        ...
