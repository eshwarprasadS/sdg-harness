from __future__ import annotations

import uuid
from contextvars import ContextVar

import structlog

logger = structlog.get_logger()

run_id_var: ContextVar[str | None] = ContextVar("run_id", default=None)
iteration_id_var: ContextVar[int | None] = ContextVar("iteration_id", default=None)


def generate_run_id() -> str:
    run_id = str(uuid.uuid4())
    logger.debug("run_id_generated", run_id=run_id)
    return run_id


def bind_run_context(run_id: str) -> None:
    run_id_var.set(run_id)
    structlog.contextvars.bind_contextvars(run_id=run_id)
    logger.debug("run_context_bound", run_id=run_id)


def bind_iteration_context(iteration_id: int) -> None:
    iteration_id_var.set(iteration_id)
    structlog.contextvars.bind_contextvars(iteration_id=iteration_id)
    logger.debug("iteration_context_bound", iteration_id=iteration_id)
