"""Benchmark: compares RAGAdapter's multidimensional detection against
simpler single-signal baselines, on SYNTHETIC trials with known ground
truth.

DEMO / SIMULATED DATA: every trial here is a synthetically generated
batch of ChatEvent-shaped dicts (same generator style as
test_rag_adapter.py), not live traffic -- because measuring precision/
recall against a live model requires knowing the true label ("was this
batch actually degraded?"), which is only available by construction in a
synthetic trial. That makes this a real, reproducible measurement of
each *detection method's* statistical behavior (given a known shift, how
often does it catch it, and how often does it cry wolf on a clean
batch), not a claim about Sanad's live quality. See docs/research.md's
"Limitations" section before citing these numbers as evidence about a
production model.

Three methods compared, all given the exact same trial data:

  single_threshold_refusal  -- the ORIGINAL LiveTelemetryAdapter rule:
                                flag if refusal rate rises more than a
                                fixed tolerance. One signal, no
                                significance test, no sample-size
                                awareness.
  ks_retrieval_only         -- one real statistical test (KS on
                                retrieval scores), but only one signal --
                                an ablation of RAGAdapter down to a
                                single dimension.
  rag_adapter_full          -- the full RAGAdapter: all four signals
                                (retrieval, generation_latency, refusal,
                                citation_validity), each a real
                                statistical test.

This directly tests H1 from docs/research.md ("multidimensional
monitoring detects more of the injected degradation types than
single-metric monitoring") against however many of the 3 injected drift
*types* below each method can see at all -- single_threshold_refusal and
ks_retrieval_only are each blind by construction to the drift types that
don't touch their one signal, which is exactly the effect being measured
here, not a flaw in how they're run.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from modelwatch.adapters.rag_adapter import RAGAdapter
from modelwatch.drift.detectors import ks_test

# The exact tolerance LiveTelemetryAdapter uses by default (config.py) --
# reimplemented standalone here (not imported) so this benchmark keeps
# working unchanged even if that adapter's defaults are retuned later.
_REFUSAL_TOLERANCE = 0.20


def _synthetic_batch(
    n: int,
    grounded_frac: float,
    retrieval_center: float,
    retrieval_spread: float,
    generation_latency_ms: float,
    citation_validity: float,
    seed: int,
) -> list[dict[str, Any]]:
    rnd = random.Random(seed)
    grounded_count = round(n * grounded_frac)
    out = []
    for i in range(n):
        grounded = i < grounded_count
        requested = 2 if grounded else 0
        valid = round(requested * citation_validity) if grounded else 0
        out.append(
            {
                "grounded": grounded,
                "citations": valid,
                "citations_requested": requested,
                "retrieval_scores": [
                    max(0.0, rnd.gauss(retrieval_center, retrieval_spread)) for _ in range(6)
                ],
                "generation_latency_ms": max(1.0, rnd.gauss(generation_latency_ms, generation_latency_ms * 0.1)),
            }
        )
    return out


@dataclass
class Trial:
    baseline: list[dict[str, Any]]
    current: list[dict[str, Any]]
    drift_injected: bool
    drift_type: str  # "none", "retrieval", "refusal", "citation", "latency"


#: One injected drift type per non-null trial, so the benchmark measures
#: detection across the different kinds of degradation RAGAdapter is
#: meant to catch (see docs/drift_detection.md), not just one.
_DRIFT_TYPES = ["retrieval", "refusal", "citation", "latency"]


def generate_trial(seed: int, n: int = 30, drift_type: str | None = None) -> Trial:
    """drift_type=None means a random 50/50 choice between "no drift" and
    one of _DRIFT_TYPES, each equally likely -- used by run_benchmark to
    build a balanced trial set. Pass an explicit drift_type to force it
    (used by the ablation study, which needs every drift type run through
    every method)."""
    rnd = random.Random(seed)
    if drift_type is None:
        drift_type = rnd.choice(["none"] + _DRIFT_TYPES)

    base_kwargs = dict(
        n=n, grounded_frac=0.9, retrieval_center=0.3, retrieval_spread=0.08,
        generation_latency_ms=800.0, citation_validity=0.95,
    )
    baseline = _synthetic_batch(seed=seed * 2, **base_kwargs)

    current_kwargs = dict(base_kwargs)
    if drift_type == "retrieval":
        current_kwargs["retrieval_center"] = 0.65
    elif drift_type == "refusal":
        current_kwargs["grounded_frac"] = 0.5
    elif drift_type == "citation":
        current_kwargs["citation_validity"] = 0.3
    elif drift_type == "latency":
        current_kwargs["generation_latency_ms"] = 4000.0
    current = _synthetic_batch(seed=seed * 2 + 1, **current_kwargs)

    return Trial(baseline, current, drift_injected=(drift_type != "none"), drift_type=drift_type)


def _single_threshold_refusal(trial: Trial) -> bool:
    base_refusal = 1 - sum(1 for e in trial.baseline if e["grounded"]) / len(trial.baseline)
    curr_refusal = 1 - sum(1 for e in trial.current if e["grounded"]) / len(trial.current)
    return (curr_refusal - base_refusal) > _REFUSAL_TOLERANCE


def _ks_retrieval_only(trial: Trial) -> bool:
    base_scores = [s for e in trial.baseline for s in e["retrieval_scores"]]
    curr_scores = [s for e in trial.current for s in e["retrieval_scores"]]
    return ks_test(base_scores, curr_scores).drift_detected


def _rag_adapter_full(trial: Trial) -> bool:
    adapter = RAGAdapter(min_events=8)
    baseline = adapter.build_baseline(trial.baseline)
    return adapter.check_drift(baseline, trial.current).is_drifted


METHODS: dict[str, Callable[[Trial], bool]] = {
    "single_threshold_refusal": _single_threshold_refusal,
    "ks_retrieval_only": _ks_retrieval_only,
    "rag_adapter_full": _rag_adapter_full,
}


@dataclass
class MethodResult:
    method: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    #: recall broken down by which drift type was injected -- the whole
    #: point of comparing methods is that a single-signal method is
    #: blind to drift types that don't touch its one signal.
    recall_by_drift_type: dict[str, float] = field(default_factory=dict)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def false_positive_rate(self) -> float:
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "false_positive_rate": self.false_positive_rate,
            "recall_by_drift_type": self.recall_by_drift_type,
        }


#: Ablation variants: the full RAGAdapter vs. the same adapter with one
#: signal's is_drifted verdict ignored when deciding overall drift. Unlike
#: METHODS above (independently built single-signal detectors), every
#: ablation variant runs the SAME adapter/statistics -- only which
#: signals are allowed to contribute to the final "drifted?" call
#: differs. That isolates each signal's marginal contribution rather
#: than conflating it with a difference in detector implementation.
ABLATION_VARIANTS: dict[str, set[str] | None] = {
    "full": None,  # None = all signals (RAGAdapter's own is_drifted)
    "without_retrieval": {"generation_latency", "refusal", "citation_validity"},
    "without_generation_latency": {"retrieval", "refusal", "citation_validity"},
    "without_refusal": {"retrieval", "generation_latency", "citation_validity"},
    "without_citation_validity": {"retrieval", "generation_latency", "refusal"},
}


def _ablation_detect(trial: Trial, allowed_signals: set[str] | None) -> bool:
    adapter = RAGAdapter(min_events=8)
    baseline = adapter.build_baseline(trial.baseline)
    result = adapter.check_drift(baseline, trial.current)
    if allowed_signals is None:
        return result.is_drifted
    if result.statistics.get("sufficient_sample") is False:
        return False
    return any(s.is_drifted for s in result.signals if s.name in allowed_signals)


def run_ablation(n_trials: int = 200, n_events: int = 30, seed: int = 0) -> dict[str, MethodResult]:
    """Same trial set and same accounting as run_benchmark, but comparing
    RAGAdapter against itself with one signal at a time excluded from the
    drift verdict -- answers "how much does dropping this signal cost
    recall on the drift type it exists to catch", per docs/research.md's
    ablation-study next step."""
    trials = [generate_trial(seed=seed * 10_000 + i, n=n_events) for i in range(n_trials)]

    results = {name: MethodResult(method=name) for name in ABLATION_VARIANTS}
    hits_by_type: dict[str, dict[str, list[bool]]] = {name: {} for name in ABLATION_VARIANTS}

    for trial in trials:
        for name, allowed in ABLATION_VARIANTS.items():
            predicted = _ablation_detect(trial, allowed)
            r = results[name]
            if trial.drift_injected and predicted:
                r.tp += 1
            elif trial.drift_injected and not predicted:
                r.fn += 1
            elif not trial.drift_injected and predicted:
                r.fp += 1
            else:
                r.tn += 1
            if trial.drift_injected:
                hits_by_type[name].setdefault(trial.drift_type, []).append(predicted)

    for name, r in results.items():
        r.recall_by_drift_type = {
            drift_type: sum(hits) / len(hits) for drift_type, hits in hits_by_type[name].items()
        }
    return results


def run_benchmark(n_trials: int = 200, n_events: int = 30, seed: int = 0) -> dict[str, MethodResult]:
    """Runs every method in METHODS against the same n_trials synthetic
    trials (so a difference between methods is due to the method, not
    different data), and returns real tp/fp/fn/tn counts."""
    trials = [generate_trial(seed=seed * 10_000 + i, n=n_events) for i in range(n_trials)]

    results = {name: MethodResult(method=name) for name in METHODS}
    hits_by_type: dict[str, dict[str, list[bool]]] = {name: {} for name in METHODS}

    for trial in trials:
        for name, method in METHODS.items():
            predicted = method(trial)
            r = results[name]
            if trial.drift_injected and predicted:
                r.tp += 1
            elif trial.drift_injected and not predicted:
                r.fn += 1
            elif not trial.drift_injected and predicted:
                r.fp += 1
            else:
                r.tn += 1

            if trial.drift_injected:
                hits_by_type[name].setdefault(trial.drift_type, []).append(predicted)

    for name, r in results.items():
        r.recall_by_drift_type = {
            drift_type: sum(hits) / len(hits) for drift_type, hits in hits_by_type[name].items()
        }
    return results
