"""Tests for the RAG X-Ray trace store (modelwatch/core/storage.py +
MonitoringEngine passthrough)."""
import pytest

from modelwatch.core.engine import MonitoringEngine
from modelwatch.core.storage import Storage


@pytest.fixture
def engine(tmp_path):
    storage = Storage(str(tmp_path / "test.db"))
    yield MonitoringEngine(storage)
    storage.close()


def test_record_and_get_trace_round_trips(engine):
    data = {"question": "What is the notice period?", "grounded": True, "retrieval": []}
    recorded = engine.record_trace("trace-1", "sanad-live", data)
    assert recorded["trace_id"] == "trace-1"
    assert recorded["model_id"] == "sanad-live"
    assert recorded["data"] == data

    fetched = engine.get_trace("trace-1")
    assert fetched == recorded


def test_get_trace_returns_none_for_unknown_id(engine):
    assert engine.get_trace("does-not-exist") is None


def test_list_traces_orders_newest_first(engine):
    engine.record_trace("t1", "m1", {"grounded": True})
    engine.record_trace("t2", "m1", {"grounded": True})
    engine.record_trace("t3", "m1", {"grounded": True})

    traces = engine.list_traces(model_id="m1")
    assert [t["trace_id"] for t in traces] == ["t3", "t2", "t1"]


def test_list_traces_filters_by_model_id(engine):
    engine.record_trace("a1", "model-a", {"grounded": True})
    engine.record_trace("b1", "model-b", {"grounded": True})

    a_traces = engine.list_traces(model_id="model-a")
    assert [t["trace_id"] for t in a_traces] == ["a1"]


def test_list_traces_filters_by_grounded(engine):
    engine.record_trace("refused-1", "m1", {"grounded": False})
    engine.record_trace("grounded-1", "m1", {"grounded": True})

    refusals = engine.list_traces(model_id="m1", grounded=False)
    assert [t["trace_id"] for t in refusals] == ["refused-1"]

    grounded = engine.list_traces(model_id="m1", grounded=True)
    assert [t["trace_id"] for t in grounded] == ["grounded-1"]


def test_list_traces_respects_limit(engine):
    for i in range(5):
        engine.record_trace(f"t{i}", "m1", {"grounded": True})
    assert len(engine.list_traces(model_id="m1", limit=2)) == 2
