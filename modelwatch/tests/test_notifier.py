"""Tests for modelwatch/alerts/notifier.py against a real local HTTP
server (pytest-httpserver), not a mocked `requests`.

Config is a frozen dataclass (see modelwatch/config.py), so tests swap
the module-level `config` name that notifier.py imported, via
dataclasses.replace() -- same pattern used in
modelwatch/tests/test_engine_integration.py's hysteresis tests.
"""
from dataclasses import replace

import pytest

import modelwatch.alerts.notifier as notifier_module
from modelwatch.alerts.notifier import notify_alert
from modelwatch.config import config as base_config


def _with_config(monkeypatch, **overrides):
    monkeypatch.setattr(notifier_module, "config", replace(base_config, **overrides))


@pytest.fixture
def configured_webhook(httpserver, monkeypatch):
    url = httpserver.url_for("/hook")
    _with_config(monkeypatch, alert_webhook_url=url)
    return url


def test_no_webhook_configured_is_a_silent_noop(monkeypatch):
    _with_config(monkeypatch, alert_webhook_url="")
    assert notify_alert("model-1", "drift detected", "degraded") is False


def test_slack_format_sends_text_and_attachment(httpserver, configured_webhook, monkeypatch):
    _with_config(monkeypatch, alert_webhook_url=configured_webhook, alert_webhook_format="slack")
    httpserver.expect_request("/hook", method="POST").respond_with_json({"ok": True})

    assert notify_alert("model-1", "drift detected", "degraded") is True

    request = httpserver.log[-1][0]
    body = request.get_json()
    assert "model-1" in body["text"]
    assert body["attachments"][0]["text"] == "drift detected"


def test_generic_format_sends_structured_fields(httpserver, configured_webhook, monkeypatch):
    _with_config(monkeypatch, alert_webhook_url=configured_webhook, alert_webhook_format="generic")
    httpserver.expect_request("/hook", method="POST").respond_with_json({"ok": True})

    assert notify_alert("model-1", "drift detected", "degraded") is True

    request = httpserver.log[-1][0]
    body = request.get_json()
    assert body == {
        "source": "modelwatch", "model_id": "model-1",
        "health_state": "degraded", "message": "drift detected",
    }


def test_webhook_failure_is_swallowed_not_raised(httpserver, configured_webhook, monkeypatch):
    _with_config(monkeypatch, alert_webhook_url=configured_webhook, alert_webhook_format="generic")
    httpserver.expect_request("/hook", method="POST").respond_with_json({"error": "boom"}, status=500)

    assert notify_alert("model-1", "drift detected", "degraded") is False  # no exception raised


def test_unreachable_webhook_is_swallowed_not_raised(monkeypatch):
    _with_config(monkeypatch, alert_webhook_url="http://127.0.0.1:9")  # discard port, no listener
    assert notify_alert("model-1", "drift detected", "degraded") is False
