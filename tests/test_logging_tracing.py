from __future__ import annotations

import io
import json

import structlog

from sdg_harness.logging import configure_logging
from sdg_harness.tracing import (
    bind_iteration_context,
    bind_run_context,
    generate_run_id,
    iteration_id_var,
    run_id_var,
)


class TestConfigureLogging:

    def test_configures_json_renderer(self) -> None:
        configure_logging(level="INFO", output_format="json")
        config = structlog.get_config()
        processor_types = [type(p).__name__ for p in config["processors"]]
        assert "JSONRenderer" in processor_types

    def test_configures_console_renderer(self) -> None:
        configure_logging(level="DEBUG", output_format="console")
        config = structlog.get_config()
        processor_types = [type(p).__name__ for p in config["processors"]]
        assert "ConsoleRenderer" in processor_types

    def test_configures_merge_contextvars(self) -> None:
        configure_logging()
        config = structlog.get_config()
        assert structlog.contextvars.merge_contextvars in config["processors"]

    def test_configures_timestamper(self) -> None:
        configure_logging()
        config = structlog.get_config()
        processor_types = [type(p).__name__ for p in config["processors"]]
        assert "TimeStamper" in processor_types

    def test_configures_add_log_level(self) -> None:
        configure_logging()
        config = structlog.get_config()
        assert structlog.stdlib.add_log_level in config["processors"]

    def test_configures_stack_info_renderer(self) -> None:
        configure_logging()
        config = structlog.get_config()
        processor_types = [type(p).__name__ for p in config["processors"]]
        assert "StackInfoRenderer" in processor_types

    def test_uses_bound_logger(self) -> None:
        configure_logging()
        config = structlog.get_config()
        assert config["wrapper_class"] is structlog.stdlib.BoundLogger


class TestGenerateRunId:

    def test_returns_string(self) -> None:
        run_id = generate_run_id()
        assert isinstance(run_id, str)

    def test_returns_uuid_format(self) -> None:
        run_id = generate_run_id()
        parts = run_id.split("-")
        assert len(parts) == 5

    def test_unique_ids(self) -> None:
        ids = {generate_run_id() for _ in range(100)}
        assert len(ids) == 100


class TestContextBinding:

    def setup_method(self) -> None:
        structlog.contextvars.clear_contextvars()
        run_id_var.set(None)
        iteration_id_var.set(None)

    def test_bind_run_context_sets_contextvar(self) -> None:
        bind_run_context("test-run-123")
        assert run_id_var.get() == "test-run-123"

    def test_bind_iteration_context_sets_contextvar(self) -> None:
        bind_iteration_context(5)
        assert iteration_id_var.get() == 5

    def test_run_context_propagates_to_log_output(self) -> None:
        configure_logging(output_format="json")
        output = io.StringIO()
        logger = structlog.PrintLoggerFactory(file=output)()
        bound = structlog.wrap_logger(logger)

        bind_run_context("trace-abc")

        processors = structlog.get_config()["processors"]
        event_dict: dict = {"event": "test_event"}
        for proc in processors[:-1]:
            event_dict = proc(bound, "info", event_dict)

        assert event_dict["run_id"] == "trace-abc"

    def test_iteration_context_propagates_to_log_output(self) -> None:
        configure_logging(output_format="json")
        output = io.StringIO()
        logger = structlog.PrintLoggerFactory(file=output)()
        bound = structlog.wrap_logger(logger)

        bind_iteration_context(7)

        processors = structlog.get_config()["processors"]
        event_dict: dict = {"event": "test_event"}
        for proc in processors[:-1]:
            event_dict = proc(bound, "info", event_dict)

        assert event_dict["iteration_id"] == 7

    def test_both_contexts_propagate(self) -> None:
        configure_logging(output_format="json")
        output = io.StringIO()
        logger = structlog.PrintLoggerFactory(file=output)()
        bound = structlog.wrap_logger(logger)

        bind_run_context("run-xyz")
        bind_iteration_context(3)

        processors = structlog.get_config()["processors"]
        event_dict: dict = {"event": "test_event"}
        for proc in processors[:-1]:
            event_dict = proc(bound, "info", event_dict)

        assert event_dict["run_id"] == "run-xyz"
        assert event_dict["iteration_id"] == 3


class TestJsonOutput:

    def setup_method(self) -> None:
        structlog.contextvars.clear_contextvars()

    def test_json_output_format(self) -> None:
        configure_logging(output_format="json")
        output = io.StringIO()
        log = structlog.PrintLogger(file=output)
        bound = structlog.wrap_logger(log)

        processors = structlog.get_config()["processors"]
        event_dict: dict = {"event": "test_event"}
        for proc in processors:
            result = proc(bound, "info", event_dict)
            if isinstance(result, str):
                parsed = json.loads(result)
                assert parsed["event"] == "test_event"
                assert "timestamp" in parsed
                assert "level" in parsed
                return
            event_dict = result

    def test_json_output_contains_iso_timestamp(self) -> None:
        configure_logging(output_format="json")
        output = io.StringIO()
        log = structlog.PrintLogger(file=output)
        bound = structlog.wrap_logger(log)

        processors = structlog.get_config()["processors"]
        event_dict: dict = {"event": "ts_test"}
        for proc in processors:
            result = proc(bound, "info", event_dict)
            if isinstance(result, str):
                parsed = json.loads(result)
                assert "T" in parsed["timestamp"]
                return
            event_dict = result

    def test_json_output_with_context(self) -> None:
        configure_logging(output_format="json")
        bind_run_context("run-for-json-test")

        output = io.StringIO()
        log = structlog.PrintLogger(file=output)
        bound = structlog.wrap_logger(log)

        processors = structlog.get_config()["processors"]
        event_dict: dict = {"event": "ctx_test"}
        for proc in processors:
            result = proc(bound, "info", event_dict)
            if isinstance(result, str):
                parsed = json.loads(result)
                assert parsed["run_id"] == "run-for-json-test"
                return
            event_dict = result
