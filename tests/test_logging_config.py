from __future__ import annotations

import json
import logging

from src.logging_config import JSONFormatter, configure_logging


def _make_record(
    *, level: int = logging.INFO, msg: str = "hello", extra: dict | None = None
) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test.logger",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )
    for key, value in (extra or {}).items():
        setattr(record, key, value)
    return record


def test_format_produces_valid_json_with_core_fields() -> None:
    record = _make_record(msg="Something happened")

    payload = json.loads(JSONFormatter().format(record))

    assert payload["severity"] == "INFO"
    assert payload["message"] == "Something happened"
    assert payload["logger"] == "test.logger"
    assert "timestamp" in payload


def test_format_includes_extra_fields() -> None:
    record = _make_record(extra={"event": "poll_run", "fetched": 3})

    payload = json.loads(JSONFormatter().format(record))

    assert payload["event"] == "poll_run"
    assert payload["fetched"] == 3


def test_format_excludes_standard_log_record_attributes() -> None:
    record = _make_record()

    payload = json.loads(JSONFormatter().format(record))

    assert "msg" not in payload
    assert "args" not in payload
    assert "pathname" not in payload
    assert "levelno" not in payload


def test_format_includes_exception_text_when_present() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = _make_record(level=logging.ERROR)
        record.exc_info = sys.exc_info()

    payload = json.loads(JSONFormatter().format(record))

    assert "ValueError: boom" in payload["exception"]


def test_configure_logging_does_not_remove_other_handlers() -> None:
    root = logging.getLogger()
    marker_handler = logging.NullHandler()
    root.addHandler(marker_handler)

    try:
        configure_logging()
        configure_logging()  # idempotent: calling twice shouldn't duplicate/disturb others

        assert marker_handler in root.handlers
        own_handlers = [
            h
            for h in root.handlers
            if getattr(h, "_issue_triaging_agent_json_handler", False)
        ]
        assert len(own_handlers) == 1
    finally:
        root.removeHandler(marker_handler)
        for h in list(root.handlers):
            if getattr(h, "_issue_triaging_agent_json_handler", False):
                root.removeHandler(h)
