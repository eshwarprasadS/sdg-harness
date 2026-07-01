from __future__ import annotations

import contextvars
import logging
import sys
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from typing import Literal

import structlog

_request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)


def _add_request_id(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    request_id = _request_id_var.get()
    if request_id is not None:
        event_dict["request_id"] = request_id
    return event_dict


def configure_logging(
    level: str = "INFO",
    output_format: Literal["json", "console"] = "json",
) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
        force=True,
    )

    renderer: structlog.types.Processor
    if output_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _add_request_id,
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )


@contextmanager
def bind_request_id(request_id: str | None = None) -> Generator[str, None, None]:
    if request_id is None:
        request_id = uuid.uuid4().hex[:12]
    token = _request_id_var.set(request_id)
    try:
        yield request_id
    finally:
        _request_id_var.reset(token)
