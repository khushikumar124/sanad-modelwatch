"""Tests for modelwatch/api/app.py's HTTP layer.

The API is a thin passthrough over MonitoringEngine (see its own module
docstring), so most behavior is already covered by testing the engine
directly. This file covers the one thing that lives only at the HTTP
layer: GET /config, which exposes real configured thresholds for the
dashboard's Statistical Analysis page to read, rather than the frontend
hardcoding a copy that could drift from modelwatch/config.py.
"""
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from modelwatch.api import app as app_module
from modelwatch.config import config as base_config


@pytest.fixture
def client():
    return TestClient(app_module.app)


def test_config_endpoint_returns_real_configured_values(client, monkeypatch):
    monkeypatch.setattr(
        app_module, "config",
        replace(base_config, rag_alpha=0.01, rag_min_events=12, mlflow_enabled=True),
    )
    res = client.get("/config")
    assert res.status_code == 200
    body = res.json()
    assert body["rag_alpha"] == 0.01
    assert body["rag_min_events"] == 12
    assert body["mlflow_enabled"] is True


def test_config_endpoint_never_exposes_paths_or_secrets(client):
    res = client.get("/config")
    body = res.json()
    forbidden_keys = {"db_path", "api_host", "api_port", "alert_webhook_url", "mlflow_tracking_uri"}
    assert forbidden_keys.isdisjoint(body.keys())


def test_config_endpoint_reflects_health_hysteresis_settings(client, monkeypatch):
    monkeypatch.setattr(
        app_module, "config",
        replace(base_config, health_degraded_after_consecutive=3),
    )
    res = client.get("/config")
    assert res.json()["health_degraded_after_consecutive"] == 3
