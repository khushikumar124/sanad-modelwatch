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


# -- Drift Lab job endpoints --------------------------------------------
# These test validation only, not real scenario execution -- a real run
# needs Ollama serving Sanad's configured model (see
# modelwatch/experiments/drift_lab.py's own docstring: "there is no
# fake-LLM path here"), which this unit test suite doesn't assume is
# available. The actual scenario execution is verified live instead.


def test_drift_lab_rejects_an_unknown_scenario(client):
    res = client.post("/drift-lab/run?scenario=not_a_real_scenario")
    assert res.status_code == 400


def test_drift_lab_rejects_n_cases_out_of_range(client):
    assert client.post("/drift-lab/run?scenario=retrieval_narrowing&n_cases=0").status_code == 400
    assert client.post("/drift-lab/run?scenario=retrieval_narrowing&n_cases=23").status_code == 400


def test_drift_lab_unknown_job_returns_404(client):
    res = client.get("/drift-lab/jobs/does-not-exist")
    assert res.status_code == 404


def test_drift_lab_accepting_a_valid_request_returns_a_pollable_job_id(client, monkeypatch):
    """Submits a real job through the real JobManager, but the scenario
    function itself is monkeypatched so this test doesn't need Ollama --
    it verifies the job plumbing (submit -> pending/running -> poll),
    not what a real scenario run measures."""
    import time

    from modelwatch.experiments import drift_lab

    def fake_retrieval_narrowing(cases, llm_client):
        return drift_lab.ScenarioResult(
            scenario="retrieval_narrowing", n_cases=0,
            baseline_events=[], current_events=[],
            drift_result=type("R", (), {"to_dict": lambda self: {"is_drifted": False}})(),
            diagnosis=type("D", (), {"to_dict": lambda self: {"likely_subsystem": None}})(),
        )

    monkeypatch.setattr(drift_lab, "retrieval_narrowing", fake_retrieval_narrowing)
    monkeypatch.setattr("sanad.rag.llm_client.OllamaClient", lambda: object())
    monkeypatch.setattr("sanad.evaluation.dataset.load_dataset", lambda path: [])

    res = client.post("/drift-lab/run?scenario=retrieval_narrowing&n_cases=1")
    assert res.status_code == 202
    job_id = res.json()["job_id"]

    for _ in range(50):
        status = client.get(f"/drift-lab/jobs/{job_id}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert status["status"] == "done", status
    assert status["result"]["scenario"] == "retrieval_narrowing"
