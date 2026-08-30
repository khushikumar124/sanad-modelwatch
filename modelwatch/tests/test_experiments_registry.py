"""Tests for the experiment registry (modelwatch/core/storage.py +
MonitoringEngine passthrough)."""
import pytest

from modelwatch.adapters.classifier_adapter import ClassifierAdapter
from modelwatch.core.engine import MonitoringEngine
from modelwatch.core.storage import Storage


@pytest.fixture
def engine(tmp_path):
    storage = Storage(str(tmp_path / "test.db"))
    yield MonitoringEngine(storage)
    storage.close()


def test_record_and_get_experiment_round_trips(engine):
    config = {"scenario": "retrieval_narrowing", "top_k_baseline": 6, "top_k_degraded": 1}
    results = {"is_drifted": True, "drift_score": 0.5}

    recorded = engine.record_experiment("driftlab-1", "drift_lab", config, results)
    assert recorded["name"] == "driftlab-1"
    assert recorded["kind"] == "drift_lab"
    assert recorded["config"] == config
    assert recorded["results"] == results
    assert recorded["status"] == "completed"

    fetched = engine.get_experiment(recorded["id"])
    assert fetched == recorded


def test_list_experiments_filters_by_kind_and_orders_newest_first(engine):
    engine.record_experiment("a", "benchmark", {}, {})
    engine.record_experiment("b", "drift_lab", {}, {})
    engine.record_experiment("c", "benchmark", {}, {})

    benchmarks = engine.list_experiments(kind="benchmark")
    assert [e["name"] for e in benchmarks] == ["c", "a"]

    all_experiments = engine.list_experiments()
    assert len(all_experiments) == 3


def test_get_experiment_returns_none_for_unknown_id(engine):
    assert engine.get_experiment(9999) is None
