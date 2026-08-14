from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

_RESERVED_ATTRS = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()
) | {"message", "asctime"}


class JSONFormatter(logging.Formatter):
    """Cloud Logging-compatible JSON formatter.

    Cloud Run parses valid single-line JSON written to stdout/stderr,
    promoting "severity" and "message" to structured LogEntry fields
    rather than leaving them as opaque text - see
    https://cloud.google.com/logging/docs/structured-logging. Any other
    key passed via `extra={...}` on a logging call lands under
    jsonPayload, queryable/filterable in Cloud Logging (e.g. by
    `event="judgment_latency"` or `outcome="published"`).
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "logger": record.name,
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_HANDLER_MARKER = "_issue_triaging_agent_json_handler"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger for structured (JSON) stdout output.

    Call once per entrypoint (app.py, scripts/run_daily_job.py) - not
    per-module - so every logger in the process shares one consistent
    handler/formatter rather than each module configuring its own.

    Idempotent without being destructive: only removes a handler this
    function previously added (tagged via _HANDLER_MARKER), never any
    other handler already attached to the root logger - e.g. pytest's
    caplog fixture attaches its own handler, and blindly clearing every
    handler here would silently break log-capturing tests that call
    entrypoint code (main()) which calls this function.
    """

    root = logging.getLogger()
    for existing in list(root.handlers):
        if getattr(existing, _HANDLER_MARKER, False):
            root.removeHandler(existing)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    setattr(handler, _HANDLER_MARKER, True)

    root.addHandler(handler)
    root.setLevel(level)
