"""Tests for modelwatch/experiments/benchmark.py.

Checks the benchmark harness itself is sound (trial generation matches
its own labels, methods return real bools, the metrics arithmetic is
correct) -- NOT that RAGAdapter beats the baselines by some specific
margin, since that's exactly what run_benchmark measures rather than
assumes. The one substantive claim tested is the qualitative one the
whole benchmark exists to demonstrate: a single-signal method is blind
to drift types that don't touch its one signal, while the full adapter
is not.
"""
from __future__ import annotations

from modelwatch.experiments.benchmark import (
    ABLATION_VARIANTS,
    MethodResult,
    generate_trial,
    run_ablation,
    run_benchmark,
)


def test_generate_trial_with_no_drift_has_matching_label():
    trial = generate_trial(seed=1, drift_type="none")
    assert trial.drift_injected is False
    assert trial.drift_type == "none"


def test_generate_trial_with_injected_type_has_matching_label():
    trial = generate_trial(seed=1, drift_type="retrieval")
    assert trial.drift_injected is True
    assert trial.drift_type == "retrieval"


def test_method_result_metrics_arithmetic():
    r = MethodResult(method="x", tp=8, fp=2, fn=2, tn=8)
    assert r.precision == 0.8
    assert r.recall == 0.8
    assert abs(r.f1 - 0.8) < 1e-9
    assert r.false_positive_rate == 0.2


def test_method_result_handles_zero_denominators_without_crashing():
    r = MethodResult(method="x")
    assert r.precision == 0.0
    assert r.recall == 0.0
    assert r.f1 == 0.0
    assert r.false_positive_rate == 0.0


def test_single_signal_methods_are_blind_to_drift_types_outside_their_signal():
    """The qualitative claim the benchmark exists to demonstrate:
    ks_retrieval_only should catch injected retrieval drift far more
    often than injected citation-only drift (which never touches
    retrieval scores at all), while rag_adapter_full catches both."""
    results = run_benchmark(n_trials=120, seed=42)

    ks_only = results["ks_retrieval_only"]
    full = results["rag_adapter_full"]

    assert ks_only.recall_by_drift_type.get("retrieval", 0) > ks_only.recall_by_drift_type.get("citation", 1)
    assert full.recall_by_drift_type.get("citation", 0) > ks_only.recall_by_drift_type.get("citation", 0)


def test_refusal_only_baseline_is_blind_to_retrieval_only_drift():
    results = run_benchmark(n_trials=120, seed=7)
    refusal_only = results["single_threshold_refusal"]
    full = results["rag_adapter_full"]

    assert refusal_only.recall_by_drift_type.get("retrieval", 0) < full.recall_by_drift_type.get("retrieval", 1)


def test_run_benchmark_returns_all_registered_methods():
    results = run_benchmark(n_trials=20, seed=1)
    assert set(results) == {"single_threshold_refusal", "ks_retrieval_only", "rag_adapter_full"}
    for r in results.values():
        assert r.tp + r.fp + r.fn + r.tn == 20


def test_ablation_returns_all_variants():
    results = run_ablation(n_trials=20, seed=1)
    assert set(results) == set(ABLATION_VARIANTS)


def test_removing_a_signal_hurts_recall_on_the_drift_type_it_exists_to_catch():
    """without_retrieval should have much worse recall on injected
    retrieval drift than the full adapter -- that's the entire point of
    an ablation."""
    results = run_ablation(n_trials=150, seed=3)
    full_recall = results["full"].recall_by_drift_type.get("retrieval", 0)
    ablated_recall = results["without_retrieval"].recall_by_drift_type.get("retrieval", 1)
    assert ablated_recall < full_recall


def test_removing_a_signal_does_not_affect_recall_on_unrelated_drift_types():
    """without_retrieval should still catch injected refusal drift just
    as well as the full adapter -- ablating one signal shouldn't silently
    break detection of drift types that never touched it."""
    results = run_ablation(n_trials=150, seed=3)
    full_recall = results["full"].recall_by_drift_type.get("refusal", 0)
    ablated_recall = results["without_retrieval"].recall_by_drift_type.get("refusal", 0)
    assert abs(ablated_recall - full_recall) < 0.15
