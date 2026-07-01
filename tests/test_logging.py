from __future__ import annotations

import json

import structlog

from sdg_harness.logging import bind_request_id, configure_logging


def test_configure_logging_json(capsys: object) -> None:
    configure_logging(level="INFO", output_format="json")
    log = structlog.get_logger()
    log.info("test_event", key="value")
    captured = capsys.readouterr()  # type: ignore[union-attr]
    data = json.loads(captured.out.strip())
    assert data["event"] == "test_event"
    assert data["key"] == "value"
    assert "timestamp" in data
    assert "level" in data


def test_configure_logging_console(capsys: object) -> None:
    configure_logging(level="INFO", output_format="console")
    log = structlog.get_logger()
    log.info("console_event")
    captured = capsys.readouterr()  # type: ignore[union-attr]
    assert "console_event" in captured.out


def test_bind_request_id_generates_id() -> None:
    with bind_request_id() as req_id:
        assert isinstance(req_id, str)
        assert len(req_id) == 12


def test_bind_request_id_uses_provided_id() -> None:
    with bind_request_id("my-request-123") as req_id:
        assert req_id == "my-request-123"


def test_bind_request_id_appears_in_log(capsys: object) -> None:
    configure_logging(level="INFO", output_format="json")
    with bind_request_id("trace-abc"):
        log = structlog.get_logger()
        log.info("traced_event")
    captured = capsys.readouterr()  # type: ignore[union-attr]
    data = json.loads(captured.out.strip())
    assert data["request_id"] == "trace-abc"


def test_bind_request_id_cleans_up(capsys: object) -> None:
    configure_logging(level="INFO", output_format="json")
    with bind_request_id("inside-id"):
        pass
    log = structlog.get_logger()
    log.info("after_context")
    captured = capsys.readouterr()  # type: ignore[union-attr]
    data = json.loads(captured.out.strip())
    assert "request_id" not in data
