"""Tests for modelwatch/diagnosis/engine.py against controlled signal
sets, built directly in the shape RAGAdapter actually emits."""
from modelwatch.diagnosis.engine import GENERATION, OPERATIONAL, RETRIEVAL, diagnose


def _signal(name, is_drifted, confidence, extra_detail=None):
    detail = {"confidence": confidence, **(extra_detail or {})}
    return {"name": name, "is_drifted": is_drifted, "value": 0.0, "detail": detail}


def _retrieval_signal(is_drifted, confidence):
    return {
        "name": "retrieval",
        "is_drifted": is_drifted,
        "value": 0.0,
        "detail": {"ks": {"confidence": confidence}, "wasserstein": {"confidence": confidence * 0.8}},
    }


def test_no_drifted_signals_yields_no_diagnosis():
    signals = [
        _retrieval_signal(False, 0.0),
        _signal("generation_latency", False, 0.0),
        _signal("refusal", False, 0.0),
        _signal("citation_validity", False, 0.0),
    ]
    result = diagnose(signals)
    assert result.likely_subsystem is None
    assert result.confidence == 0.0


def test_retrieval_regression_with_downstream_citation_and_refusal_points_at_retrieval():
    """The worked example from the spec: retrieval declines, citation
    declines with it, latency and refusal stay flat -> RETRIEVAL, not a
    three-way split."""
    signals = [
        _retrieval_signal(True, 0.9),
        _signal("citation_validity", True, 0.8),
        _signal("generation_latency", False, 0.0),
        _signal("refusal", False, 0.0),
    ]
    result = diagnose(signals)
    assert result.likely_subsystem == RETRIEVAL
    assert result.confidence > 0.9  # primary + supporting evidence
    assert any("citation" in r for r in result.reasoning)
    assert any("stable" in r for r in result.reasoning)


def test_citation_and_refusal_drift_without_retrieval_points_at_generation():
    """Retrieval is stable, but the model is refusing more and citing
    less accurately -- a model/prompt-side problem, not retrieval."""
    signals = [
        _retrieval_signal(False, 0.0),
        _signal("citation_validity", True, 0.85),
        _signal("refusal", True, 0.7),
        _signal("generation_latency", False, 0.0),
    ]
    result = diagnose(signals)
    assert result.likely_subsystem == GENERATION
    assert any("retrieval" in r and "stable" in r for r in result.reasoning)


def test_latency_drift_alone_points_at_operational():
    signals = [
        _retrieval_signal(False, 0.0),
        _signal("citation_validity", False, 0.0),
        _signal("refusal", False, 0.0),
        _signal("generation_latency", True, 0.95),
    ]
    result = diagnose(signals)
    assert result.likely_subsystem == OPERATIONAL


def test_ranked_list_includes_all_three_subsystems_even_when_one_dominates():
    signals = [
        _retrieval_signal(True, 0.9),
        _signal("citation_validity", False, 0.0),
        _signal("refusal", False, 0.0),
        _signal("generation_latency", False, 0.0),
    ]
    result = diagnose(signals)
    subsystems = {s for s, _ in result.ranked}
    assert subsystems == {RETRIEVAL, GENERATION, OPERATIONAL}
    assert result.ranked[0][0] == RETRIEVAL


def test_result_is_json_serializable():
    import json

    signals = [_retrieval_signal(True, 0.9), _signal("generation_latency", True, 0.5)]
    json.dumps(diagnose(signals).to_dict())
