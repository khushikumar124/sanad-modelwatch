"""Tests for sanad/observability.py's JSON logging and request-ID
correlation, including a real end-to-end check against the live FastAPI
app (not just the formatter/middleware in isolation)."""
import json
import logging

from fastapi.testclient import TestClient

from sanad.api.app import app
from sanad.observability import JsonFormatter, request_id_var

client = TestClient(app)


def _make_record(msg="hello", **extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="sanad.test", level=logging.INFO, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=None,
    )
    for k, v in extra.items():
        setattr(record, k, v)
    return record


def test_json_formatter_produces_valid_json_with_the_core_fields():
    formatted = JsonFormatter().format(_make_record("something happened"))
    payload = json.loads(formatted)
    assert payload["message"] == "something happened"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "sanad.test"
    assert "timestamp" in payload


def test_json_formatter_includes_extra_fields():
    formatted = JsonFormatter().format(_make_record("uploaded", doc_id="abc123", chunk_count=5))
    payload = json.loads(formatted)
    assert payload["doc_id"] == "abc123"
    assert payload["chunk_count"] == 5


def test_json_formatter_includes_request_id_when_set():
    token = request_id_var.set("req-42")
    try:
        payload = json.loads(JsonFormatter().format(_make_record("in a request")))
        assert payload["request_id"] == "req-42"
    finally:
        request_id_var.reset(token)


def test_json_formatter_omits_request_id_when_not_set():
    payload = json.loads(JsonFormatter().format(_make_record("no request context")))
    assert "request_id" not in payload


def test_json_formatter_output_never_carries_a_stray_none_extra_by_default():
    # a record with no extras at all shouldn't pick up any stdlib
    # LogRecord internals (e.g. "msg", "args", "exc_info") as if they
    # were caller-supplied extras
    payload = json.loads(JsonFormatter().format(_make_record("plain")))
    assert set(payload) <= {"timestamp", "level", "logger", "message", "request_id", "exc_info"}


def test_response_carries_a_generated_request_id_when_none_was_sent():
    res = client.get("/api/auth/session")
    assert res.status_code == 200
    assert "X-Request-ID" in res.headers
    assert len(res.headers["X-Request-ID"]) > 0


def test_response_echoes_back_a_caller_supplied_request_id():
    res = client.get("/api/auth/session", headers={"X-Request-ID": "caller-supplied-id"})
    assert res.headers["X-Request-ID"] == "caller-supplied-id"


def test_each_request_gets_a_different_generated_id():
    first = client.get("/api/auth/session").headers["X-Request-ID"]
    second = client.get("/api/auth/session").headers["X-Request-ID"]
    assert first != second
