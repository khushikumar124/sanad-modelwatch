"""ModelAdapter for RAG applications, monitored on real statistical tests
over the structured per-event telemetry Sanad emits (sanad/api/telemetry.py).

This supersedes what LiveTelemetryAdapter can do -- that adapter only ever
sees aggregate rates (refusal_rate, citation_rate) per batch, so its drift
rule is necessarily "did the rate move more than a fixed tolerance". This
adapter keeps the *raw* per-event values in the baseline (every retrieval
score, every latency, every grounded/citation outcome) and runs real
distributional/proportion tests from modelwatch.drift.detectors against
them, so:

* sample size is accounted for (a two-event blip can't trip a
  two-proportion z-test the way it could trip a bare delta check)
* a broad, gradual shift in retrieval quality is caught by Wasserstein
  distance even where KS's max-CDF-gap statistic might miss it
* every verdict comes with a p-value/effect-size, not just a boolean

It is registered under adapter_name "rag", a new name -- LiveTelemetryAdapter
keeps working unmodified under "live_telemetry" for anything already
registered against it. This is the adapter new RAG monitoring should
register against going forward (see docs/drift_detection.md).

Represents RAG health as five independent signals rather than collapsing
everything into one opaque number (the "quality vector" from
docs/research.md):
  - retrieval:   is the distribution of retrieval similarity scores
                 still what it was at baseline? (KS + Wasserstein)
  - generation:  is response generation latency still what it was?
                 (KS) -- a proxy for "is the model itself behaving
                 differently" (swapped, under load, etc).
  - refusal:     has the refusal rate changed? (two-proportion z-test)
  - citation:    has the fraction of requested citations that were
                 actually valid changed? (two-proportion z-test)
  - embedding:   has the *distribution* of question embedding vectors
                 shifted (a topic/phrasing shift in what's being asked),
                 as opposed to any single scalar summary of it? (MMD,
                 see drift/detectors.py's embedding_drift). Requires
                 Sanad's full-trace telemetry to be on (question
                 embeddings ride along under that same gate); with it
                 off, this signal reports insufficient_sample rather
                 than being silently omitted -- consistent with every
                 other detector's "insufficient data", not a fabricated
                 verdict.
"""
from __future__ import annotations

import logging
from typing import Any

from modelwatch.config import config
from modelwatch.core.adapter_base import DriftCheckResult, ModelAdapter, SignalResult
from modelwatch.drift.detectors import (
    DetectorResult,
    embedding_drift,
    ks_test,
    two_proportion_ztest,
    wasserstein_distance,
)

logger = logging.getLogger(__name__)


def _extract(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Flattens a batch of ChatEvent-shaped dicts into the raw arrays the
    detectors need. retrieval_scores is a list-of-lists in the wire
    format (one list per event) -- flattened here into one pool, since
    the question this adapter asks is "has the overall distribution of
    retrieval scores shifted", not "did any single request's retrieval
    look different"."""
    retrieval_scores: list[float] = []
    for e in events:
        retrieval_scores.extend(e.get("retrieval_scores") or [])
    return {
        "retrieval_scores": retrieval_scores,
        "generation_latency_ms": [float(e.get("generation_latency_ms") or 0.0) for e in events],
        # One vector per event, not flattened like retrieval_scores --
        # embedding_drift compares distributions of whole vectors, not
        # pooled scalar components. Events recorded with full-trace
        # telemetry off (or from before this field existed) simply have
        # no embedding and are dropped here, not zero-filled.
        "question_embeddings": [e["question_embedding"] for e in events if e.get("question_embedding")],
        "n_events": len(events),
        "n_grounded": sum(1 for e in events if e.get("grounded")),
        "n_citations_valid": sum(int(e.get("citations") or 0) for e in events),
        "n_citations_requested": sum(int(e.get("citations_requested") or 0) for e in events),
    }


def _embedding_drift_signal(baseline_vectors: list, current_vectors: list) -> DetectorResult:
    try:
        return embedding_drift(baseline_vectors, current_vectors)
    except ValueError as e:
        # Mismatched embedding dimensionality (e.g. the monitored app
        # changed its embedding model mid-operation) -- a real detector
        # can't compare across models, but that's not a reason to crash
        # the whole drift check over one signal.
        logger.warning("embedding_drift signal could not be computed", extra={"error": str(e)})
        return DetectorResult(
            detector="embedding_drift",
            statistic=0.0,
            p_value=None,
            effect_size=0.0,
            drift_detected=False,
            confidence=0.0,
            insufficient_sample=True,
            n_baseline=len(baseline_vectors),
            n_current=len(current_vectors),
            detail={"reason": str(e)},
        )


class RAGAdapter(ModelAdapter):
    adapter_name = "rag"

    def __init__(self, alpha: float | None = None, min_events: int | None = None):
        self.alpha = alpha if alpha is not None else config.rag_alpha
        self.min_events = min_events if min_events is not None else config.rag_min_events

    def build_baseline(self, data: list[dict[str, Any]]) -> dict[str, Any]:
        """data = a window of ChatEvent-shaped dicts (as returned by
        Sanad's /api/telemetry) collected during normal operation."""
        return _extract(list(data))

    def check_drift(self, baseline: dict[str, Any], new_data: list[dict[str, Any]]) -> DriftCheckResult:
        current = _extract(list(new_data))
        n = current["n_events"]
        enough = n >= self.min_events and baseline["n_events"] >= self.min_events

        retrieval_ks = ks_test(baseline["retrieval_scores"], current["retrieval_scores"], alpha=self.alpha)
        retrieval_wasserstein = wasserstein_distance(baseline["retrieval_scores"], current["retrieval_scores"])
        retrieval_drifted = enough and (retrieval_ks.drift_detected or retrieval_wasserstein.drift_detected)

        generation_ks = ks_test(
            baseline["generation_latency_ms"], current["generation_latency_ms"], alpha=self.alpha
        )

        refusal_test = two_proportion_ztest(
            baseline_successes=baseline["n_events"] - baseline["n_grounded"],
            baseline_n=baseline["n_events"],
            current_successes=current["n_events"] - current["n_grounded"],
            current_n=current["n_events"],
            alpha=self.alpha,
        )

        citation_test = two_proportion_ztest(
            baseline_successes=baseline["n_citations_valid"],
            baseline_n=max(baseline["n_citations_requested"], 1),
            current_successes=current["n_citations_valid"],
            current_n=max(current["n_citations_requested"], 1),
            alpha=self.alpha,
        )

        embedding_test = _embedding_drift_signal(baseline["question_embeddings"], current["question_embeddings"])

        signals = [
            SignalResult(
                name="retrieval",
                value=retrieval_ks.effect_size,
                is_drifted=retrieval_drifted,
                detail={"ks": retrieval_ks.to_dict(), "wasserstein": retrieval_wasserstein.to_dict()},
            ),
            SignalResult(
                name="generation_latency",
                value=generation_ks.effect_size,
                is_drifted=enough and generation_ks.drift_detected,
                detail={"ks": generation_ks.to_dict()},
            ),
            SignalResult(
                name="refusal",
                value=refusal_test.detail.get("current_rate", 0.0) if refusal_test.detail else 0.0,
                is_drifted=enough and refusal_test.drift_detected,
                detail=refusal_test.to_dict(),
            ),
            SignalResult(
                name="citation_validity",
                value=citation_test.detail.get("current_rate", 0.0) if citation_test.detail else 0.0,
                is_drifted=enough and citation_test.drift_detected,
                detail=citation_test.to_dict(),
            ),
            SignalResult(
                name="embedding",
                value=embedding_test.effect_size,
                is_drifted=enough and embedding_test.drift_detected and not embedding_test.insufficient_sample,
                detail=embedding_test.to_dict(),
            ),
        ]

        drifted_count = sum(1 for s in signals if s.is_drifted)
        drift_score = drifted_count / len(signals)
        is_drifted = enough and drifted_count > 0
        quality_score = current["n_grounded"] / n if n > 0 else None

        if not enough:
            logger.info(
                "rag adapter batch below min_events, reporting without flagging",
                extra={"n_events": n, "baseline_n_events": baseline["n_events"], "min_events": self.min_events},
            )

        return DriftCheckResult(
            drift_score=drift_score,
            quality_score=quality_score,
            is_drifted=is_drifted,
            signals=signals,
            statistics={
                "n_events": n,
                "baseline_n_events": baseline["n_events"],
                "min_events": self.min_events,
                "sufficient_sample": enough,
                "alpha": self.alpha,
                "n_signals_drifted": drifted_count,
            },
        )
