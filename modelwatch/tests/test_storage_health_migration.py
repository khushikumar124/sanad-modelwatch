"""Storage-level test for get_health()'s upgrade-in-place behavior: a
model that has an unresolved alert but no model_health row (i.e. it was
registered and checked before that table existed) must not report
'healthy' -- see storage.get_health()'s docstring."""
from modelwatch.core.storage import Storage


def test_model_with_open_alert_but_no_health_row_reports_degraded(tmp_path):
    storage = Storage(str(tmp_path / "test.db"))
    storage.create_model("m1", "test model", "classifier")
    run_id = storage.save_run("m1", version=1, drift_score=0.9, quality_score=0.1, is_drifted=True, signals=[])
    storage.create_alert("m1", run_id, "drift detected")

    # no set_health() call was ever made -- simulates a pre-Phase-E alert
    health = storage.get_health("m1")
    assert health["state"] == "degraded"
    assert health["consecutive_drifted"] >= 1
    storage.close()


def test_model_with_no_alerts_and_no_health_row_reports_healthy(tmp_path):
    storage = Storage(str(tmp_path / "test.db"))
    storage.create_model("m2", "test model", "classifier")
    health = storage.get_health("m2")
    assert health["state"] == "healthy"
    storage.close()


def test_model_with_resolved_alert_only_reports_healthy(tmp_path):
    storage = Storage(str(tmp_path / "test.db"))
    storage.create_model("m3", "test model", "classifier")
    run_id = storage.save_run("m3", version=1, drift_score=0.9, quality_score=0.1, is_drifted=True, signals=[])
    alert_id = storage.create_alert("m3", run_id, "drift detected")
    storage.resolve_alerts_for_model("m3")

    health = storage.get_health("m3")
    assert health["state"] == "healthy"
    storage.close()
