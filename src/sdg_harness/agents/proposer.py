from __future__ import annotations

import json
from typing import Any

import structlog

from sdg_harness.agents.base import Agent
from sdg_harness.agents.models import LLMUsage, ProposalContext
from sdg_harness.core.proposal import Proposal

logger = structlog.get_logger()

SYSTEM_PROMPT = """\
You are a proposal agent for synthetic data generation (SDG) optimization.

Your job is to translate a diagnosis from the analyst into concrete \
configuration changes. You receive the current config, the analysis, \
any constraints, and historical context. Propose specific parameter \
changes with rationale and risk assessment.

Respond with a JSON object matching this schema:
{
    "changes": {"param_name": new_value, ...},
    "rationale": {"param_name": "why this change", ...},
    "expected_effect": "description of expected impact",
    "risk": "low|medium|high with explanation"
}
"""


class ProposerAgent(Agent):

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        if args:
            return await self.propose(args[0])
        raise TypeError("propose() requires a ProposalContext argument")

    async def propose(
        self, context: ProposalContext
    ) -> tuple[Proposal, LLMUsage]:
        logger.info(
            "proposer_started",
            config_keys=list(context.config.config.keys()),
            num_constraints=len(context.constraints),
            history_length=len(context.history),
        )

        user_content = self._format_input(context)
        messages = self._build_messages(SYSTEM_PROMPT, user_content)
        result, usage = await self._complete(messages, Proposal)

        logger.info(
            "proposer_completed",
            num_changes=len(result.changes),
            risk=result.risk,
        )

        return result, usage

    def _format_input(self, context: ProposalContext) -> str:
        cfg = json.dumps(context.config.model_dump(), indent=2)
        analysis = json.dumps(context.analysis, indent=2)
        parts = [
            f"Current configuration: {cfg}",
            f"Analysis: {analysis}",
        ]
        if context.constraints:
            parts.append(
                f"Constraints: {json.dumps(context.constraints)}"
            )
        if context.history:
            parts.append(
                f"Previous iterations: {json.dumps(context.history, indent=2)}"
            )
        return "\n\n".join(parts)
