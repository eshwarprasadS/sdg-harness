from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from typing import Any, TypeVar

import litellm
import structlog
from pydantic import BaseModel, ValidationError

from sdg_harness.agents.models import LLMUsage

logger = structlog.get_logger()

T = TypeVar("T", bound=BaseModel)


class Agent(ABC):

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.total_usage = LLMUsage()
        logger.info(
            "agent_initialized",
            agent=self.__class__.__name__,
            model=model,
            temperature=temperature,
        )

    @abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> Any:
        logger.info("agent_run_started", agent=self.__class__.__name__)

    async def _complete(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
    ) -> tuple[T, LLMUsage]:
        prompt_hash = hashlib.sha256(
            json.dumps(messages, sort_keys=True).encode()
        ).hexdigest()[:12]

        logger.info(
            "llm_call_started",
            agent=self.__class__.__name__,
            model=self.model,
            prompt_hash=prompt_hash,
        )

        start = time.monotonic()

        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )

        latency = time.monotonic() - start
        content = response.choices[0].message.content or ""

        usage_data = response.usage
        prompt_tokens = getattr(usage_data, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage_data, "completion_tokens", 0) or 0
        total_tokens = prompt_tokens + completion_tokens

        cost = 0.0
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            pass

        call_usage = LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
        )

        self.total_usage = LLMUsage(
            prompt_tokens=self.total_usage.prompt_tokens + prompt_tokens,
            completion_tokens=self.total_usage.completion_tokens + completion_tokens,
            total_tokens=self.total_usage.total_tokens + total_tokens,
            cost=self.total_usage.cost + cost,
        )

        logger.info(
            "llm_call_completed",
            agent=self.__class__.__name__,
            model=self.model,
            prompt_hash=prompt_hash,
            response_length=len(content),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
            latency_seconds=round(latency, 3),
        )

        parsed = self._parse_response(content, response_model)
        return parsed, call_usage

    def _parse_response(self, content: str, response_model: type[T]) -> T:
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.error(
                "llm_response_parse_error",
                agent=self.__class__.__name__,
                error="invalid_json",
                content_preview=content[:200],
            )
            raise ValueError(
                f"LLM returned invalid JSON: {exc}"
            ) from exc

        try:
            return response_model.model_validate(data)
        except ValidationError as exc:
            logger.error(
                "llm_response_validation_error",
                agent=self.__class__.__name__,
                error="validation_failed",
                content_preview=content[:200],
            )
            raise ValueError(
                f"LLM response failed validation: {exc}"
            ) from exc

    def _build_messages(
        self,
        system_prompt: str,
        user_content: str,
    ) -> list[dict[str, str]]:
        logger.debug(
            "building_messages",
            agent=self.__class__.__name__,
            system_prompt_length=len(system_prompt),
            user_content_length=len(user_content),
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
