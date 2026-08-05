"""LiveTelemetryAdapter tests against controlled ground truth.

Every batch here is constructed so the correct verdict is known before
the detector runs: a batch that matches the baseline must not flag, a
batch with a deliberately worse refusal rate must.
"""
import pytest

from modelwatch.adapters.live_telemetry_adapter import LiveTelemetryAdapter


def events(n, grounded_frac=0.8, citations=2, latency_ms=1000.0):
    """n events with an exact grounded fraction, so rates are known."""
    grounded_count = round(n * grounded_frac)
    return [
        {
            "grounded": i < grounded_count,
            "citations": citations if i < grounded_count else 0,
            "latency_ms": latency_ms,
        }
        for i in range(n)
    ]


def test_baseline_records_the_rates():
    adapter = LiveTelemetryAdapter()
    base = adapter.build_baseline(events(20, grounded_frac=0.8))["metrics"]
    assert base["refusal_rate"] == pytest.approx(0.2)
    assert base["citation_rate"] == pytest.approx(0.8)
    assert base["n_events"] == 20


def test_matching_batch_is_not_flagged():
    adapter = LiveTelemetryAdapter()
    baseline = adapter.build_baseline(events(20, grounded_frac=0.8))
    result = adapter.check_drift(baseline, events(20, grounded_frac=0.8))
    assert result.is_drifted is False
    assert result.quality_score == pytest.approx(0.8)   # grounded-answer rate
    assert all(not s.is_drifted for s in result.signals)


def test_refusal_spike_is_flagged():
    """80% grounded at baseline, 20% now -- a 60 point rise in refusals,
    far past the 20 point tolerance."""
    adapter = LiveTelemetryAdapter()
    baseline = adapter.build_baseline(events(20, grounded_frac=0.8))
    result = adapter.check_drift(baseline, events(20, grounded_frac=0.2))
    assert result.is_drifted is True
    refusal = next(s for s in result.signals if s.name == "refusal rate")
    assert refusal.is_drifted is True
    assert refusal.value == pytest.approx(0.8)
    assert result.quality_score == pytest.approx(0.2)


def test_small_refusal_movement_within_tolerance_is_not_flagged():
    adapter = LiveTelemetryAdapter()
    baseline = adapter.build_baseline(events(20, grounded_frac=0.8))
    result = adapter.check_drift(baseline, events(20, grounded_frac=0.7))
    assert result.is_drifted is False


def test_latency_blowup_is_flagged():
    adapter = LiveTelemetryAdapter()
    baseline = adapter.build_baseline(events(20, latency_ms=1000))
    result = adapter.check_drift(baseline, events(20, latency_ms=9000))
    latency = next(s for s in result.signals if s.name == "latency p95")
    assert latency.is_drifted is True
    assert latency.detail["ratio"] == 9.0


def test_citation_collapse_is_flagged():
    adapter = LiveTelemetryAdapter()
    baseline = adapter.build_baseline(events(20, grounded_frac=1.0, citations=2))
    # still answering, but no longer citing anything
    result = adapter.check_drift(baseline, events(20, grounded_frac=1.0, citations=0))
    citation = next(s for s in result.signals if s.name == "citation rate")
    assert citation.is_drifted is True


def test_tiny_batch_reports_but_never_flags():
    """Two unlucky refusals out of three is not evidence of a regression.
    Below min_events the metrics are still reported, but nothing fires."""
    adapter = LiveTelemetryAdapter(min_events=5)
    baseline = adapter.build_baseline(events(20, grounded_frac=1.0))
    result = adapter.check_drift(baseline, events(3, grounded_frac=0.0))
    assert result.is_drifted is False
    assert result.statistics["sufficient_sample"] is False
    assert result.statistics["n_events"] == 3
    assert result.quality_score == pytest.approx(0.0)  # still reported honestly


def test_signal_values_stay_in_unit_range():
    """The dashboard draws every signal on a 0-1 scale, so an adapter
    returning a raw millisecond count would blow the bars out."""
    adapter = LiveTelemetryAdapter()
    baseline = adapter.build_baseline(events(20, latency_ms=100))
    result = adapter.check_drift(baseline, events(20, latency_ms=999_999))
    assert all(0.0 <= s.value <= 1.0 for s in result.signals)
