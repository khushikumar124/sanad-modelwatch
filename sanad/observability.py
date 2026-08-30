"""Structured logging and per-request correlation.

Two independent, additive pieces:

1. JsonFormatter: renders each log line as one JSON object (timestamp,
   level, logger name, message, request_id, plus whatever `extra={...}`
   a call site passed -- and there are many already, e.g. `extra={
   "doc_id": ...}` throughout api/app.py). A human reading server output
   directly still gets the existing plain-text format by default
   (SANAD_LOG_FORMAT=text); JSON is opt-in (SANAD_LOG_FORMAT=json) for
   when logs are actually going to be parsed by something -- grep,
   `jq`, or a real log aggregator, none of which this app ships or
   requires.

2. request_id_var + RequestIDMiddleware (wired in api/app.py): every
   request gets a request ID -- reused from an incoming X-Request-ID
   header if the caller already has one (so a request can be traced
   across multiple hops), otherwise a fresh uuid4. Every log line
   emitted while handling that request automatically carries it (via the
   contextvar, not by threading a parameter through every function
   signature), and the response echoes it back in X-Request-ID so a
   client-reported bug can be matched to exact server-side log lines.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="")

#: Attributes every stdlib LogRecord has -- anything else on a record is
#: something a call site passed via `extra={...}` and belongs in the
#: JSON output, not this fixed set repeated for every single line.
_STANDARD_RECORD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id:
            payload["request_id"] = request_id
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str, log_format: str) -> None:
    handler = logging.StreamHandler()
    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))

    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)


async def request_id_middleware(request, call_next):
    incoming = request.headers.get("X-Request-ID")
    request_id = incoming if incoming else uuid.uuid4().hex
    token = request_id_var.set(request_id)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = request_id
    return response
