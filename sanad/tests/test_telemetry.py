"""Tests for the telemetry event schema in sanad/api/telemetry.py.

Covers both the original five fields (which LiveTelemetryAdapter reads by
name and must not break) and the richer additive fields introduced for
structured, per-request monitoring.
"""
from sanad.api import telemetry


def setup_function(_fn):
    telemetry.snapshot(drain=True)  # clear any events left by other tests


def test_record_chat_with_only_original_fields_still_works():
    """Backward compatibility: a caller that only passes the original five
    keyword args (as older code did) must not break."""
    telemetry.record_chat(
        grounded=True, citations=1, latency_ms=120.0, parse_error=False, retrieved=3
    )
    events = telemetry.snapshot(drain=True)
    assert len(events) == 1
    e = events[0]
    assert e["grounded"] is True
    assert e["citations"] == 1
    assert e["retrieved"] == 3
    # trace_id is always generated; other additive fields default to
    # empty/zero rather than being absent when the caller omits them
    assert len(e["trace_id"]) > 0
    assert e["retrieval_scores"] == []


def test_record_chat_captures_structured_fields():
    telemetry.record_chat(
        grounded=True,
        citations=1,
        latency_ms=250.0,
        parse_error=False,
        retrieved=3,
        doc_id="doc-abc",
        model_name="phi3:3.8b",
        top_k=6,
        retrieval_scores=[0.12, 0.31, 0.44],
        retrieval_latency_ms=40.0,
        generation_latency_ms=210.0,
        citations_requested=2,
    )
    events = telemetry.snapshot(drain=True)
    assert len(events) == 1
    e = events[0]
    assert e["doc_id"] == "doc-abc"
    assert e["model_name"] == "phi3:3.8b"
    assert e["top_k"] == 6
    assert e["retrieval_scores"] == [0.12, 0.31, 0.44]
    assert e["retrieval_latency_ms"] == 40.0
    assert e["generation_latency_ms"] == 210.0
    assert e["citations_requested"] == 2
    # trace_id is generated per event and unique
    assert len(e["trace_id"]) > 0


def test_each_event_gets_a_distinct_trace_id():
    telemetry.record_chat(grounded=True, citations=1, latency_ms=1.0, parse_error=False, retrieved=1)
    telemetry.record_chat(grounded=True, citations=1, latency_ms=1.0, parse_error=False, retrieved=1)
    events = telemetry.snapshot(drain=True)
    assert events[0]["trace_id"] != events[1]["trace_id"]
