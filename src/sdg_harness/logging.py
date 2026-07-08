from __future__ import annotations

import logging
from typing import Any

import structlog


def _make_level_filter(level: str) -> structlog.types.Processor:
    min_level = getattr(logging, level.upper(), logging.INFO)

    def level_filter(
        logger: Any, method_name: str, event_dict: dict[str, Any]
    ) -> dict[str, Any]:
        if getattr(logging, method_name.upper(), 0) < min_level:
            raise structlog.DropEvent
        return event_dict

    return level_filter


def configure_logging(
    level: str = "INFO",
    output_format: str = "json",
) -> None:
    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        _make_level_filter(level),
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if output_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )

    logger = structlog.get_logger()
    logger.debug(
        "logging_configured", level=level, output_format=output_format
    )
