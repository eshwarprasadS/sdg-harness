from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sdg_harness.agents.analyst import AnalystAgent
from sdg_harness.agents.base import Agent
from sdg_harness.agents.models import (
    AnalysisContext,
    LLMUsage,
    PipelineInfo,
    PlannerResult,
    ProposalContext,
)
from sdg_harness.agents.planner import PlannerAgent
from sdg_harness.agents.proposer import ProposerAgent
from sdg_harness.core.analysis import AnalysisReport
from sdg_harness.core.config import IterationConfig
from sdg_harness.core.proposal import Proposal
from sdg_harness.core.result import IterationResult


def _make_litellm_response(content: dict[str, Any]) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = json.dumps(content)
    response.usage = MagicMock()
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 50
    response.usage.total_tokens = 150
    return response


@pytest.fixture
def pipeline_info() -> PipelineInfo:
    return PipelineInfo(
        description="SDG pipeline for math QA generation",
        config_schema={
            "temperature": {"type": "float", "min": 0.0, "max": 2.0},
            "num_samples": {"type": "int", "min": 1},
            "model": {"type": "string"},
        },
        constraints=["model must not change"],
    )


@pytest.fixture
def analysis_context(
    sample_config: IterationConfig,
    sample_result: IterationResult,
) -> AnalysisContext:
    return AnalysisContext(
        config=sample_config,
        result=sample_result,
        data_samples=[{"input": "2+2", "output": "4"}],
        history=[],
    )


@pytest.fixture
def proposal_context(
    sample_config: IterationConfig,
    sample_analysis: AnalysisReport,
) -> ProposalContext:
    return ProposalContext(
        config=sample_config,
        analysis=sample_analysis.model_dump(),
        constraints=["temperature <= 1.5"],
        history=[],
    )


class TestAgent:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            Agent()  # type: ignore[abstract]

    def test_parse_response_valid_json(self) -> None:
        class ConcreteAgent(Agent):
            async def run(self, *args: Any, **kwargs: Any) -> Any:
                return None

        agent = ConcreteAgent()
        data = {"mutable_parameters": ["temp"], "fixed_parameters": ["model"],
                "search_strategy": "grid", "rationale": "test"}
        result = agent._parse_response(json.dumps(data), PlannerResult)
        assert result.mutable_parameters == ["temp"]

    def test_parse_response_invalid_json(self) -> None:
        class ConcreteAgent(Agent):
            async def run(self, *args: Any, **kwargs: Any) -> Any:
                return None

        agent = ConcreteAgent()
        with pytest.raises(ValueError, match="invalid JSON"):
            agent._parse_response("not json", PlannerResult)

    def test_parse_response_validation_error(self) -> None:
        class ConcreteAgent(Agent):
            async def run(self, *args: Any, **kwargs: Any) -> Any:
                return None

        agent = ConcreteAgent()
        with pytest.raises(ValueError, match="validation"):
            agent._parse_response(json.dumps({"bad": "data"}), PlannerResult)

    def test_build_messages(self) -> None:
        class ConcreteAgent(Agent):
            async def run(self, *args: Any, **kwargs: Any) -> Any:
                return None

        agent = ConcreteAgent()
        messages = agent._build_messages("system prompt", "user input")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "system prompt"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "user input"

    def test_default_model_and_temperature(self) -> None:
        class ConcreteAgent(Agent):
            async def run(self, *args: Any, **kwargs: Any) -> Any:
                return None

        agent = ConcreteAgent()
        assert agent.model == "gpt-4o"
        assert agent.temperature == 0.0

    def test_custom_model_and_temperature(self) -> None:
        class ConcreteAgent(Agent):
            async def run(self, *args: Any, **kwargs: Any) -> Any:
                return None

        agent = ConcreteAgent(model="gpt-3.5-turbo", temperature=0.7)
        assert agent.model == "gpt-3.5-turbo"
        assert agent.temperature == 0.7


class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_plan_returns_planner_result(
        self, pipeline_info: PipelineInfo
    ) -> None:
        response_data = {
            "mutable_parameters": ["temperature", "num_samples"],
            "fixed_parameters": ["model"],
            "search_strategy": "bayesian optimization",
            "rationale": "temperature and num_samples directly affect output quality",
        }
        mock_response = _make_litellm_response(response_data)
        mock_acomp = AsyncMock(return_value=mock_response)

        with patch("litellm.acompletion", mock_acomp):
            with patch("litellm.completion_cost", return_value=0.01):
                agent = PlannerAgent()
                result, usage = await agent.plan(pipeline_info)

        assert isinstance(result, PlannerResult)
        assert result.mutable_parameters == ["temperature", "num_samples"]
        assert result.fixed_parameters == ["model"]
        assert result.search_strategy == "bayesian optimization"
        assert isinstance(usage, LLMUsage)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50

    @pytest.mark.asyncio
    async def test_plan_prompt_contains_pipeline_info(
        self, pipeline_info: PipelineInfo
    ) -> None:
        response_data = {
            "mutable_parameters": [],
            "fixed_parameters": [],
            "search_strategy": "random",
            "rationale": "test",
        }
        mock_response = _make_litellm_response(response_data)
        mock_acompletion = AsyncMock(return_value=mock_response)

        with patch("litellm.acompletion", mock_acompletion):
            with patch("litellm.completion_cost", return_value=0.0):
                agent = PlannerAgent()
                await agent.plan(pipeline_info)

        call_args = mock_acompletion.call_args
        messages = call_args.kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "math QA generation" in user_msg
        assert "temperature" in user_msg
        assert "model must not change" in user_msg

    @pytest.mark.asyncio
    async def test_plan_tracks_cumulative_usage(
        self, pipeline_info: PipelineInfo
    ) -> None:
        response_data = {
            "mutable_parameters": [],
            "fixed_parameters": [],
            "search_strategy": "random",
            "rationale": "test",
        }
        mock_response = _make_litellm_response(response_data)

        mock_acomp = AsyncMock(return_value=mock_response)

        with patch("litellm.acompletion", mock_acomp):
            with patch("litellm.completion_cost", return_value=0.005):
                agent = PlannerAgent()
                await agent.plan(pipeline_info)
                await agent.plan(pipeline_info)

        assert agent.total_usage.prompt_tokens == 200
        assert agent.total_usage.completion_tokens == 100
        assert agent.total_usage.cost == pytest.approx(0.01)


class TestAnalystAgent:
    @pytest.mark.asyncio
    async def test_analyze_returns_analysis_report(
        self, analysis_context: AnalysisContext
    ) -> None:
        response_data = {
            "summary": "Decent quality with room for improvement",
            "data_quality_issues": ["low diversity"],
            "training_issues": [],
            "eval_breakdown": {"accuracy": 0.85},
            "strengths": ["high accuracy"],
            "weaknesses": ["low diversity"],
            "root_causes": ["narrow templates"],
            "suggested_directions": ["add more templates"],
        }
        mock_response = _make_litellm_response(response_data)

        mock_acomp = AsyncMock(return_value=mock_response)

        with patch("litellm.acompletion", mock_acomp):
            with patch("litellm.completion_cost", return_value=0.02):
                agent = AnalystAgent()
                result, usage = await agent.analyze(analysis_context)

        assert isinstance(result, AnalysisReport)
        assert result.summary == "Decent quality with room for improvement"
        assert "low diversity" in result.data_quality_issues
        assert result.root_causes == ["narrow templates"]

    @pytest.mark.asyncio
    async def test_analyze_prompt_contains_metrics(
        self, analysis_context: AnalysisContext
    ) -> None:
        response_data = {
            "summary": "ok",
            "data_quality_issues": [],
            "training_issues": [],
            "eval_breakdown": {},
            "strengths": [],
            "weaknesses": [],
            "root_causes": [],
            "suggested_directions": [],
        }
        mock_response = _make_litellm_response(response_data)
        mock_acompletion = AsyncMock(return_value=mock_response)

        with patch("litellm.acompletion", mock_acompletion):
            with patch("litellm.completion_cost", return_value=0.0):
                agent = AnalystAgent()
                await agent.analyze(analysis_context)

        call_args = mock_acompletion.call_args
        messages = call_args.kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "accuracy" in user_msg
        assert "0.85" in user_msg


class TestProposerAgent:
    @pytest.mark.asyncio
    async def test_propose_returns_proposal(
        self, proposal_context: ProposalContext
    ) -> None:
        response_data = {
            "changes": {"temperature": 0.9},
            "rationale": {"temperature": "increase diversity"},
            "expected_effect": "higher diversity at slight accuracy cost",
            "risk": "low",
        }
        mock_response = _make_litellm_response(response_data)

        mock_acomp = AsyncMock(return_value=mock_response)

        with patch("litellm.acompletion", mock_acomp):
            with patch("litellm.completion_cost", return_value=0.015):
                agent = ProposerAgent()
                result, usage = await agent.propose(proposal_context)

        assert isinstance(result, Proposal)
        assert result.changes == {"temperature": 0.9}
        assert result.risk == "low"

    @pytest.mark.asyncio
    async def test_propose_prompt_contains_constraints(
        self, proposal_context: ProposalContext
    ) -> None:
        response_data = {
            "changes": {},
            "rationale": {},
            "expected_effect": "none",
            "risk": "low",
        }
        mock_response = _make_litellm_response(response_data)
        mock_acompletion = AsyncMock(return_value=mock_response)

        with patch("litellm.acompletion", mock_acompletion):
            with patch("litellm.completion_cost", return_value=0.0):
                agent = ProposerAgent()
                await agent.propose(proposal_context)

        call_args = mock_acompletion.call_args
        messages = call_args.kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "temperature <= 1.5" in user_msg


class TestAgentModels:
    def test_pipeline_info_roundtrip(self, pipeline_info: PipelineInfo) -> None:
        data = pipeline_info.model_dump()
        restored = PipelineInfo.model_validate(data)
        assert restored == pipeline_info

    def test_planner_result_roundtrip(self) -> None:
        result = PlannerResult(
            mutable_parameters=["temperature"],
            fixed_parameters=["model"],
            search_strategy="grid search",
            rationale="test",
        )
        data = result.model_dump()
        restored = PlannerResult.model_validate(data)
        assert restored == result

    def test_analysis_context_roundtrip(
        self, analysis_context: AnalysisContext
    ) -> None:
        data = analysis_context.model_dump()
        restored = AnalysisContext.model_validate(data)
        assert restored == analysis_context

    def test_proposal_context_roundtrip(
        self, proposal_context: ProposalContext
    ) -> None:
        data = proposal_context.model_dump()
        restored = ProposalContext.model_validate(data)
        assert restored == proposal_context

    def test_llm_usage_defaults(self) -> None:
        usage = LLMUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.cost == 0.0

    def test_pipeline_info_default_constraints(self) -> None:
        info = PipelineInfo(
            description="test",
            config_schema={"key": "value"},
        )
        assert info.constraints == []

    def test_analysis_context_defaults(
        self, sample_config: IterationConfig, sample_result: IterationResult
    ) -> None:
        ctx = AnalysisContext(config=sample_config, result=sample_result)
        assert ctx.data_samples == []
        assert ctx.history == []

    def test_proposal_context_defaults(
        self, sample_config: IterationConfig
    ) -> None:
        ctx = ProposalContext(
            config=sample_config,
            analysis={"summary": "test"},
        )
        assert ctx.constraints == []
        assert ctx.history == []
