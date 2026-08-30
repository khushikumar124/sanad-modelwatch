"""Tests for modelwatch_client.ModelWatchClient against a real local HTTP
server (pytest-httpserver), not a mocked `requests` -- exercises the
actual request construction and JSON handling, same pattern as
sanad/tests/test_llm_client.py in the main repo."""
import pytest

from modelwatch_client import ModelWatchClient, ModelWatchError


@pytest.fixture
def client(httpserver):
    return ModelWatchClient(httpserver.url_for(""))


def test_register_model_sends_expected_payload(httpserver, client):
    httpserver.expect_request(
        "/models", method="POST",
        json={"model_id": "m1", "name": "Test", "adapter_name": "rag", "baseline_data": [1, 2], "config": {}},
    ).respond_with_json({"model_id": "m1", "name": "Test", "adapter_name": "rag", "config": {}})

    result = client.register_model("m1", "Test", "rag", [1, 2])
    assert result["model_id"] == "m1"


def test_check_returns_drift_result(httpserver, client):
    httpserver.expect_request("/models/m1/check", method="POST").respond_with_json(
        {"run_id": 1, "alert_id": None, "health_state": "healthy", "drift_score": 0.0, "quality_score": 1.0, "is_drifted": False, "signals": []}
    )
    result = client.check("m1", [{"grounded": True}])
    assert result["is_drifted"] is False
    assert result["health_state"] == "healthy"


def test_is_registered_true_on_200(httpserver, client):
    httpserver.expect_request("/models/m1", method="GET").respond_with_json({"model_id": "m1"})
    assert client.is_registered("m1") is True


def test_is_registered_false_on_404(httpserver, client):
    httpserver.expect_request("/models/m1", method="GET").respond_with_json({"detail": "not found"}, status=404)
    assert client.is_registered("m1") is False


def test_record_and_get_trace(httpserver, client):
    httpserver.expect_request("/traces", method="POST").respond_with_json(
        {"id": 1, "trace_id": "t1", "model_id": "m1", "created_at": "now", "data": {"question": "hi"}}
    )
    result = client.record_trace("t1", "m1", {"question": "hi"})
    assert result["trace_id"] == "t1"

    httpserver.expect_request("/traces/t1", method="GET").respond_with_json(
        {"id": 1, "trace_id": "t1", "model_id": "m1", "created_at": "now", "data": {"question": "hi"}}
    )
    fetched = client.get_trace("t1")
    assert fetched["data"]["question"] == "hi"


def test_diagnose_trace(httpserver, client):
    httpserver.expect_request("/traces/t1/diagnosis", method="GET").respond_with_json(
        {"category": "none", "reasoning": [], "evidence": {}, "operational_note": None}
    )
    result = client.diagnose_trace("t1")
    assert result["category"] == "none"


def test_unreachable_server_raises_modelwatch_error():
    client = ModelWatchClient("http://127.0.0.1:9")  # discard port, no listener
    with pytest.raises(ModelWatchError):
        client.check("m1", [])


def test_server_error_raises_modelwatch_error_with_status_code(httpserver, client):
    httpserver.expect_request("/models/m1/check", method="POST").respond_with_json({"detail": "boom"}, status=500)
    with pytest.raises(ModelWatchError) as exc_info:
        client.check("m1", [])
    assert exc_info.value.status_code == 500


def test_get_alerts_passes_query_params(httpserver, client):
    httpserver.expect_request(
        "/alerts", method="GET", query_string="active_only=true&model_id=m1"
    ).respond_with_json([{"id": 1, "message": "drift"}])
    alerts = client.get_alerts(model_id="m1", active_only=True)
    assert len(alerts) == 1


def test_record_experiment(httpserver, client):
    httpserver.expect_request("/experiments", method="POST").respond_with_json(
        {"id": 1, "name": "exp1", "kind": "benchmark", "created_at": "now", "config": {}, "results": {}, "status": "completed"}
    )
    result = client.record_experiment("exp1", "benchmark", {}, {})
    assert result["name"] == "exp1"
