"""Tests for modelwatch/integrations/mlflow_export.py against a real
local MLflow file-store tracking backend (mlflow.set_tracking_uri to a
tmp_path directory) -- not a mocked mlflow client. Reads results back
via mlflow's own MlflowClient, the same way a real consumer of this
integration would."""
import pytest

mlflow = pytest.importorskip("mlflow")

from modelwatch.integrations.mlflow_export import log_run_to_mlflow


def _sample_result(drift_score=0.5, quality_score=0.8, is_drifted=True):
    return {
        "drift_score": drift_score,
        "quality_score": quality_score,
        "is_drifted": is_drifted,
        "signals": [
            {"name": "retrieval", "value": 0.3, "is_drifted": True, "detail": {}},
            {"name": "refusal", "value": 0.1, "is_drifted": False, "detail": {}},
        ],
        "statistics": {"n_events": 30},
    }


def _latest_run(tracking_uri: str, experiment_name: str):
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    assert experiment is not None, f"experiment {experiment_name!r} was never created"
    runs = client.search_runs([experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
    assert runs, "no runs were logged"
    return runs[0]


def test_logs_a_run_with_drift_and_quality_metrics(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    log_run_to_mlflow(
        experiment_name="test-experiment",
        tracking_uri=tracking_uri,
        model_id="sanad-chatbot",
        version=1,
        result=_sample_result(drift_score=0.42, quality_score=0.77),
    )

    run = _latest_run(tracking_uri, "test-experiment")
    assert run.data.metrics["drift_score"] == pytest.approx(0.42)
    assert run.data.metrics["quality_score"] == pytest.approx(0.77)
    assert run.data.tags["modelwatch.model_id"] == "sanad-chatbot"
    assert run.data.tags["modelwatch.version"] == "1"


def test_logs_per_signal_metrics(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    log_run_to_mlflow(
        experiment_name="test-experiment",
        tracking_uri=tracking_uri,
        model_id="sanad-chatbot",
        version=1,
        result=_sample_result(),
    )

    run = _latest_run(tracking_uri, "test-experiment")
    assert run.data.metrics["signal.retrieval.value"] == pytest.approx(0.3)
    assert run.data.metrics["signal.retrieval.is_drifted"] == 1.0
    assert run.data.metrics["signal.refusal.is_drifted"] == 0.0


def test_omits_quality_metric_when_quality_score_is_none(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    log_run_to_mlflow(
        experiment_name="test-experiment",
        tracking_uri=tracking_uri,
        model_id="sanad-chatbot",
        version=1,
        result=_sample_result(quality_score=None),
    )

    run = _latest_run(tracking_uri, "test-experiment")
    assert "quality_score" not in run.data.metrics


def test_includes_health_state_tag_when_provided(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    log_run_to_mlflow(
        experiment_name="test-experiment",
        tracking_uri=tracking_uri,
        model_id="sanad-chatbot",
        version=1,
        result=_sample_result(),
        health_state="degraded",
    )

    run = _latest_run(tracking_uri, "test-experiment")
    assert run.data.tags["modelwatch.health_state"] == "degraded"


def test_each_call_creates_a_separate_run(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    for i in range(3):
        log_run_to_mlflow(
            experiment_name="test-experiment",
            tracking_uri=tracking_uri,
            model_id="sanad-chatbot",
            version=i,
            result=_sample_result(),
        )

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("test-experiment")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 3


def test_a_broken_tracking_uri_is_swallowed_not_raised(caplog):
    # An unwritable/invalid path -- this must not propagate and break
    # the drift check that already succeeded and was already persisted.
    with caplog.at_level("ERROR"):
        log_run_to_mlflow(
            experiment_name="test-experiment",
            tracking_uri="sqlite:////nonexistent-root-that-cannot-be-created/mlflow.db",
            model_id="sanad-chatbot",
            version=1,
            result=_sample_result(),
        )
    assert any("failed to export" in r.message for r in caplog.records)
