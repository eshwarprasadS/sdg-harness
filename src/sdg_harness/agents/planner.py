from __future__ import annotations

import json
from typing import Any

import structlog

from sdg_harness.agents.base import Agent
from sdg_harness.agents.models import LLMUsage, PipelineInfo, PlannerResult

logger = structlog.get_logger()

SYSTEM_PROMPT = """\
You are a pipeline planning agent for synthetic data generation (SDG) optimization.

Your job is to analyze a pipeline description and its configuration schema \
to determine which parameters are mutable (safe to change during optimization) \
and which are fixed (should not be changed). You also recommend a search strategy.

Respond with a JSON object matching this schema:
{
    "mutable_parameters": ["list of parameter names that can be tuned"],
    "fixed_parameters": ["list of parameter names that must stay constant"],
    "search_strategy": "description of the recommended search approach",
    "rationale": "explanation of why these parameters were classified this way"
}
"""


class PlannerAgent(Agent):

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        if args:
            return await self.plan(args[0])
        raise TypeError("plan() requires a PipelineInfo argument")

    async def plan(
        self, pipeline_info: PipelineInfo
    ) -> tuple[PlannerResult, LLMUsage]:
        logger.info(
            "planner_started",
            description_length=len(pipeline_info.description),
            schema_keys=list(pipeline_info.config_schema.keys()),
            num_constraints=len(pipeline_info.constraints),
        )

        user_content = self._format_input(pipeline_info)
        messages = self._build_messages(SYSTEM_PROMPT, user_content)
        result, usage = await self._complete(messages, PlannerResult)

        logger.info(
            "planner_completed",
            num_mutable=len(result.mutable_parameters),
            num_fixed=len(result.fixed_parameters),
            search_strategy=result.search_strategy,
        )

        return result, usage

    def _format_input(self, pipeline_info: PipelineInfo) -> str:
        schema = json.dumps(pipeline_info.config_schema, indent=2)
        parts = [
            f"Pipeline description: {pipeline_info.description}",
            f"Configuration schema: {schema}",
        ]
        if pipeline_info.constraints:
            parts.append(
                f"Constraints: {json.dumps(pipeline_info.constraints)}"
            )
        return "\n\n".join(parts)
