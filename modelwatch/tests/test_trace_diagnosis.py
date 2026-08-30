"""Tests for modelwatch/diagnosis/trace_diagnosis.py against controlled
trace dicts, in the same shape sanad/features/trace.py's RAGTrace.to_dict()
produces."""
from modelwatch.diagnosis.trace_diagnosis import (
    CITATION_PROBLEM,
    GENERATION_PROBLEM,
    INSUFFICIENT_EVIDENCE,
    IRRELEVANT_RETRIEVAL,
    NONE_CATEGORY,
    RETRIEVAL_MISS,
    diagnose_trace,
)


def _retrieval(sims):
    return [{"rank": i + 1, "similarity": s, "cited": False} for i, s in enumerate(sims)]


def test_empty_retrieval_is_retrieval_miss():
    trace = {"retrieval": [], "grounded": False, "claims": []}
    result = diagnose_trace(trace)
    assert result.category == RETRIEVAL_MISS


def test_refusal_with_low_similarity_is_irrelevant_retrieval():
    trace = {"retrieval": _retrieval([0.1, 0.15, 0.2]), "grounded": False, "claims": []}
    result = diagnose_trace(trace)
    assert result.category == IRRELEVANT_RETRIEVAL


def test_refusal_with_moderate_similarity_is_insufficient_evidence():
    trace = {"retrieval": _retrieval([0.4, 0.35]), "grounded": False, "claims": []}
    result = diagnose_trace(trace)
    assert result.category == INSUFFICIENT_EVIDENCE


def test_refusal_despite_good_retrieval_is_generation_problem():
    trace = {"retrieval": _retrieval([0.75, 0.6]), "grounded": False, "claims": []}
    result = diagnose_trace(trace)
    assert result.category == GENERATION_PROBLEM


def test_grounded_with_low_citation_score_is_citation_problem():
    trace = {
        "retrieval": _retrieval([0.7]), "grounded": True, "claims": [],
        "citation_score": 0.5, "grounding_score": 1.0,
    }
    result = diagnose_trace(trace)
    assert result.category == CITATION_PROBLEM


def test_grounded_with_unsupported_claim_is_generation_problem():
    trace = {
        "retrieval": _retrieval([0.7]), "grounded": True,
        "claims": [{"status": "supported"}, {"status": "unsupported"}],
        "citation_score": 1.0, "grounding_score": 0.5,
    }
    result = diagnose_trace(trace)
    assert result.category == GENERATION_PROBLEM


def test_grounded_and_consistent_is_none():
    trace = {
        "retrieval": _retrieval([0.8]), "grounded": True,
        "claims": [{"status": "supported"}],
        "citation_score": 1.0, "grounding_score": 1.0,
    }
    result = diagnose_trace(trace)
    assert result.category == NONE_CATEGORY


def test_slow_retrieval_flags_operational_note_alongside_category():
    trace = {
        "retrieval": _retrieval([0.8]), "grounded": True,
        "claims": [{"status": "supported"}],
        "citation_score": 1.0, "grounding_score": 1.0,
        "retrieval_latency_ms": 20000, "generation_latency_ms": 500,
    }
    result = diagnose_trace(trace)
    assert result.category == NONE_CATEGORY  # quality is fine
    assert result.operational_note is not None
    assert "20000" in result.operational_note or "20,000" in result.operational_note


def test_evidence_contains_real_numbers_not_fabricated_confidence():
    trace = {"retrieval": _retrieval([0.5, 0.3]), "grounded": False, "claims": []}
    result = diagnose_trace(trace)
    assert result.evidence["best_similarity"] == 0.5
    assert "confidence" not in result.evidence


def test_result_is_json_serializable():
    import json

    trace = {"retrieval": _retrieval([0.5]), "grounded": False, "claims": []}
    json.dumps(diagnose_trace(trace).to_dict())
