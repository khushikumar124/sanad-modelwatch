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


# -- Counterfactual experiment endpoints ---------------------------------
# Validation only, same reasoning as the Drift Lab tests above -- a real
# run needs Ollama.


def test_counterfactual_rejects_malformed_top_k(client):
    res = client.post("/counterfactual/run?top_k=not,numbers")
    assert res.status_code == 400


def test_counterfactual_rejects_too_few_top_k_values(client):
    res = client.post("/counterfactual/run?top_k=4")
    assert res.status_code == 400


def test_counterfactual_rejects_too_many_top_k_values(client):
    res = client.post("/counterfactual/run?top_k=1,2,3,4,5,6")
    assert res.status_code == 400


def test_counterfactual_rejects_out_of_range_top_k(client):
    res = client.post("/counterfactual/run?top_k=4,21")
    assert res.status_code == 400


def test_counterfactual_rejects_n_cases_out_of_range(client):
    assert client.post("/counterfactual/run?top_k=4,6&n_cases=0").status_code == 400
    assert client.post("/counterfactual/run?top_k=4,6&n_cases=23").status_code == 400


def test_counterfactual_unknown_job_returns_404(client):
    res = client.get("/counterfactual/jobs/does-not-exist")
    assert res.status_code == 404


def test_counterfactual_accepting_a_valid_request_returns_a_pollable_job_id(client, monkeypatch):
    import time

    from modelwatch.experiments import counterfactual

    def fake_compare_top_k(cases, llm_client, top_k_values, chroma_path=None):
        return counterfactual.CounterfactualResult(
            n_cases=0, variants=[{"top_k": v, "summary": {}} for v in top_k_values]
        )

    monkeypatch.setattr(counterfactual, "compare_top_k", fake_compare_top_k)
    monkeypatch.setattr("sanad.rag.llm_client.OllamaClient", lambda: object())
    monkeypatch.setattr("sanad.evaluation.dataset.load_dataset", lambda path: [])

    res = client.post("/counterfactual/run?top_k=4,6&n_cases=1")
    assert res.status_code == 202
    job_id = res.json()["job_id"]

    for _ in range(50):
        status = client.get(f"/counterfactual/jobs/{job_id}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert status["status"] == "done", status
    assert [v["top_k"] for v in status["result"]["variants"]] == [4, 6]


# -- Model comparison endpoint --------------------------------------------


def test_model_comparison_rejects_too_few_models(client):
    res = client.post("/counterfactual/run-models?models=phi3:3.8b")
    assert res.status_code == 400


def test_model_comparison_rejects_too_many_models(client):
    res = client.post("/counterfactual/run-models?models=a,b,c,d,e")
    assert res.status_code == 400


def test_model_comparison_rejects_n_cases_out_of_range(client):
    assert client.post("/counterfactual/run-models?models=a,b&n_cases=0").status_code == 400
    assert client.post("/counterfactual/run-models?models=a,b&n_cases=23").status_code == 400


def test_model_comparison_accepting_a_valid_request_returns_a_pollable_job_id(client, monkeypatch):
    import time

    from modelwatch.experiments import counterfactual

    def fake_compare_models(cases, model_names, chroma_path=None):
        return counterfactual.CounterfactualResult(
            n_cases=0, variants=[{"model": m, "summary": {}} for m in model_names]
        )

    monkeypatch.setattr(counterfactual, "compare_models", fake_compare_models)
    monkeypatch.setattr("sanad.evaluation.dataset.load_dataset", lambda path: [])

    res = client.post("/counterfactual/run-models?models=phi3:3.8b,qwen2.5:0.5b&n_cases=1")
    assert res.status_code == 202
    job_id = res.json()["job_id"]

    for _ in range(50):
        status = client.get(f"/counterfactual/jobs/{job_id}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert status["status"] == "done", status
    assert [v["model"] for v in status["result"]["variants"]] == ["phi3:3.8b", "qwen2.5:0.5b"]
