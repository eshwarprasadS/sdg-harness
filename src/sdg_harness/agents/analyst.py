from __future__ import annotations

import json
from typing import Any

import structlog

from sdg_harness.agents.base import Agent
from sdg_harness.agents.models import AnalysisContext, LLMUsage
from sdg_harness.core.analysis import AnalysisReport

logger = structlog.get_logger()

SYSTEM_PROMPT = """\
You are an analysis agent for synthetic data generation (SDG) optimization.

Your job is to analyze the results of an SDG iteration: examine the metrics, \
data samples, training signals, and historical context to produce a structured \
diagnosis of what went well, what went wrong, and what the root causes are.

Respond with a JSON object matching this schema:
{
    "summary": "brief overall assessment",
    "data_quality_issues": ["list of data quality problems found"],
    "training_issues": ["list of training-related problems found"],
    "eval_breakdown": {"metric_name": score, ...},
    "strengths": ["list of things that went well"],
    "weaknesses": ["list of things that went poorly"],
    "root_causes": ["list of identified root causes for problems"],
    "suggested_directions": ["list of suggested improvements"]
}
"""


class AnalystAgent(Agent):

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        logger.info("analyst_run_dispatched")
        if args:
            return await self.analyze(args[0])
        raise TypeError("analyze() requires an AnalysisContext argument")

    async def analyze(
        self, context: AnalysisContext
    ) -> tuple[AnalysisReport, LLMUsage]:
        logger.info(
            "analyst_started",
            config_keys=list(context.config.config.keys()),
            num_metrics=len(context.result.metrics),
            num_data_samples=len(context.data_samples),
            history_length=len(context.history),
        )

        user_content = self._format_input(context)
        messages = self._build_messages(SYSTEM_PROMPT, user_content)
        result, usage = await self._complete(messages, AnalysisReport)

        logger.info(
            "analyst_completed",
            num_issues=len(result.data_quality_issues) + len(result.training_issues),
            num_root_causes=len(result.root_causes),
            num_suggestions=len(result.suggested_directions),
        )

        return result, usage

    def _format_input(self, context: AnalysisContext) -> str:
        logger.debug("analyst_formatting_input")
        cfg = json.dumps(context.config.model_dump(), indent=2)
        res = json.dumps(context.result.model_dump(), indent=2)
        parts = [
            f"Current configuration: {cfg}",
            f"Iteration results: {res}",
        ]
        if context.data_samples:
            samples = json.dumps(context.data_samples[:5], indent=2)
            parts.append(f"Data samples (first 5): {samples}")
        if context.history:
            parts.append(
                f"Previous iterations: {json.dumps(context.history, indent=2)}"
            )
        return "\n\n".join(parts)
