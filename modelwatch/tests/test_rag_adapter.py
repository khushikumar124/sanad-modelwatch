"""RAGAdapter tests against controlled ground truth, same spirit as
test_live_telemetry_adapter.py: every batch is constructed so the
correct verdict is known before the detector runs.
"""
import random

import pytest

from modelwatch.adapters.rag_adapter import RAGAdapter


def events(
    n,
    grounded_frac=0.9,
    retrieval_center=0.3,
    retrieval_spread=0.1,
    generation_latency_ms=800.0,
    citation_validity=1.0,
    embedding_center=None,
    embedding_dims=8,
    seed=1,
):
    """n synthetic ChatEvent-shaped dicts with controlled ground truth:
    exact grounded fraction, retrieval scores drawn from a known
    distribution, and a known citation-validity ratio.

    embedding_center=None (the default) omits question_embedding
    entirely, matching events recorded with full-trace telemetry off."""
    rnd = random.Random(seed)
    grounded_count = round(n * grounded_frac)
    out = []
    for i in range(n):
        grounded = i < grounded_count
        requested = 2 if grounded else 0
        valid = round(requested * citation_validity) if grounded else 0
        event = {
            "grounded": grounded,
            "citations": valid,
            "citations_requested": requested,
            "retrieval_scores": [
                max(0.0, rnd.gauss(retrieval_center, retrieval_spread)) for _ in range(6)
            ],
            "generation_latency_ms": generation_ms_sample(rnd, generation_latency_ms),
        }
        if embedding_center is not None:
            event["question_embedding"] = [rnd.gauss(c, 0.2) for c in embedding_center]
        out.append(event)
    return out


def _center(shift: float, dims: int = 8) -> list[float]:
    return [shift] + [0.0] * (dims - 1)


def generation_ms_sample(rnd, center):
    return max(1.0, rnd.gauss(center, center * 0.1))


def test_matching_batch_is_not_flagged():
    adapter = RAGAdapter()
    baseline = adapter.build_baseline(events(30, seed=1))
    result = adapter.check_drift(baseline, events(30, seed=2))
    assert result.is_drifted is False
    assert all(not s.is_drifted for s in result.signals)


def test_retrieval_score_shift_is_flagged():
    """Retrieval distances rising from ~0.3 to ~0.7 means retrieved chunks
    are much less similar to the query -- a real retrieval regression."""
    adapter = RAGAdapter()
    baseline = adapter.build_baseline(events(30, retrieval_center=0.3, seed=1))
    result = adapter.check_drift(baseline, events(30, retrieval_center=0.7, seed=2))
    retrieval = next(s for s in result.signals if s.name == "retrieval")
    assert retrieval.is_drifted is True
    assert result.is_drifted is True


def test_refusal_spike_is_flagged():
    adapter = RAGAdapter()
    baseline = adapter.build_baseline(events(30, grounded_frac=0.9, seed=1))
    result = adapter.check_drift(baseline, events(30, grounded_frac=0.3, seed=2))
    refusal = next(s for s in result.signals if s.name == "refusal")
    assert refusal.is_drifted is True
    assert result.quality_score == pytest.approx(0.3, abs=0.05)


def test_citation_validity_collapse_is_flagged():
    adapter = RAGAdapter()
    baseline = adapter.build_baseline(events(30, citation_validity=1.0, seed=1))
    result = adapter.check_drift(baseline, events(30, citation_validity=0.2, seed=2))
    citation = next(s for s in result.signals if s.name == "citation_validity")
    assert citation.is_drifted is True


def test_generation_latency_blowup_is_flagged():
    adapter = RAGAdapter()
    baseline = adapter.build_baseline(events(30, generation_latency_ms=800, seed=1))
    result = adapter.check_drift(baseline, events(30, generation_latency_ms=8000, seed=2))
    latency = next(s for s in result.signals if s.name == "generation_latency")
    assert latency.is_drifted is True


def test_tiny_batch_reports_but_never_flags():
    adapter = RAGAdapter(min_events=8)
    baseline = adapter.build_baseline(events(30, grounded_frac=1.0, seed=1))
    result = adapter.check_drift(baseline, events(3, grounded_frac=0.0, seed=2))
    assert result.is_drifted is False
    assert result.statistics["sufficient_sample"] is False
    assert all(not s.is_drifted for s in result.signals)


def test_every_signal_carries_a_p_value_or_documented_alternative():
    """Every signal's detail must be traceable to an actual statistical
    result -- p_value for the two hypothesis tests, or PSI/Wasserstein's
    own effect-size-based confidence for the distributional ones."""
    adapter = RAGAdapter()
    baseline = adapter.build_baseline(events(30, seed=1))
    result = adapter.check_drift(baseline, events(30, seed=2))

    refusal = next(s for s in result.signals if s.name == "refusal")
    assert refusal.detail["p_value"] is not None

    citation = next(s for s in result.signals if s.name == "citation_validity")
    assert citation.detail["p_value"] is not None

    retrieval = next(s for s in result.signals if s.name == "retrieval")
    assert retrieval.detail["ks"]["p_value"] is not None
    assert "confidence" in retrieval.detail["wasserstein"]


def test_result_is_json_serializable():
    import json

    adapter = RAGAdapter()
    baseline = adapter.build_baseline(events(30, seed=1))
    result = adapter.check_drift(baseline, events(30, seed=2))
    json.dumps(result.to_dict())


def test_embedding_signal_reports_insufficient_when_no_events_have_embeddings():
    """Matches events recorded with SANAD_TELEMETRY_FULL_TRACE off (the
    default is on, but this must degrade gracefully, not crash)."""
    adapter = RAGAdapter()
    baseline = adapter.build_baseline(events(30, seed=1))
    result = adapter.check_drift(baseline, events(30, seed=2))
    embedding = next(s for s in result.signals if s.name == "embedding")
    assert embedding.is_drifted is False
    assert embedding.detail["insufficient_sample"] is True


def test_embedding_signal_not_flagged_when_question_topics_are_stable():
    adapter = RAGAdapter()
    baseline = adapter.build_baseline(events(30, embedding_center=_center(0.0), seed=1))
    result = adapter.check_drift(baseline, events(30, embedding_center=_center(0.0), seed=2))
    embedding = next(s for s in result.signals if s.name == "embedding")
    assert embedding.is_drifted is False


def test_embedding_signal_flagged_on_a_real_topic_shift():
    """A retrieval-score/latency/refusal-invisible change: same grounded
    rate, same retrieval scores, same latency -- only the *topic* of the
    questions being asked has moved, which only the embedding signal can
    see."""
    adapter = RAGAdapter()
    baseline = adapter.build_baseline(events(30, embedding_center=_center(0.0), seed=1))
    result = adapter.check_drift(baseline, events(30, embedding_center=_center(4.0), seed=2))
    embedding = next(s for s in result.signals if s.name == "embedding")
    assert embedding.is_drifted is True
    assert result.is_drifted is True
    # citation_validity and refusal are constructed identically in both
    # batches (same grounded_frac, same citation_validity) -- if the
    # embedding shift is really independent of the other signals, these
    # two specifically must stay quiet. (retrieval/generation aren't
    # asserted here: with several independent tests run per check, an
    # occasional one crossing threshold on noise alone is the documented,
    # expected false-positive behavior -- see docs/research.md's H2.)
    assert next(s for s in result.signals if s.name == "citation_validity").is_drifted is False
    assert next(s for s in result.signals if s.name == "refusal").is_drifted is False
