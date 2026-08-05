"""ModelAdapter for live operational signals from an LLM application.

The golden-set adapter (`llm_adapter.py`) answers "are the answers still
correct?", which requires knowing the correct answer, which means it can
only ever run against a fixed test set. That leaves real usage unmonitored:
a user asking a real question produces no signal at all.

This adapter watches the signals that *don't* need ground truth, and so
can be computed for every real request:

* **refusal rate** -- the fraction of questions answered with "the document
  does not address this". A rising refusal rate is the clearest early
  warning available for this application, because over-refusal is its
  known failure mode.
* **citation rate** -- how often an answer was backed by a real retrieved
  clause. Answers drifting away from citations means grounding is eroding.
* **latency** -- p95 response time, which catches a model being swapped
  for a heavier one, or the machine being under load.

Scope, stated plainly because the name is more general than the thing:
this adapter is **specific to a RAG chatbot**. Refusal rate and citation
rate are not general ML concepts -- they only exist for an application
that can decline to answer and can cite sources. Unlike ClassifierAdapter
(any tabular model) or LLMAdapter (any LLM app with a golden set), this
does not transfer to an arbitrary model. The general equivalent would
watch predictions against labels.

None of these say an answer was *correct*. They say the system is still
behaving the way it did when the baseline was taken. That is a genuinely
different question from the golden set's, which is why this is a separate
adapter rather than a bigger LLMAdapter.

Rates over a handful of events are extremely noisy, so a batch smaller
than `min_events` reports its metrics but never raises drift -- otherwise
two unlucky refusals in a batch of three would page someone.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

from modelwatch.config import config
from modelwatch.core.adapter_base import DriftCheckResult, ModelAdapter, SignalResult

logger = logging.getLogger(__name__)


def _percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile. No numpy dependency for four numbers."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round(pct / 100 * len(ordered) + 0.5)) - 1))
    return float(ordered[k])


def _metrics(events: list[dict[str, Any]]) -> dict[str, float]:
    n = len(events)
    if n == 0:
        return {"refusal_rate": 0.0, "citation_rate": 0.0, "latency_p95_ms": 0.0, "n_events": 0}
    grounded = sum(1 for e in events if e.get("grounded"))
    cited = sum(1 for e in events if (e.get("citations") or 0) > 0)
    latencies = [float(e.get("latency_ms") or 0) for e in events]
    return {
        "refusal_rate": (n - grounded) / n,
        "citation_rate": cited / n,
        "latency_p95_ms": _percentile(latencies, 95),
        "n_events": n,
    }


class LiveTelemetryAdapter(ModelAdapter):
    adapter_name = "live_telemetry"

    def __init__(
        self,
        refusal_tolerance: float | None = None,
        citation_tolerance: float | None = None,
        latency_multiplier: float | None = None,
        min_events: int | None = None,
    ):
        self.refusal_tolerance = (
            refusal_tolerance if refusal_tolerance is not None else config.telemetry_refusal_tolerance
        )
        self.citation_tolerance = (
            citation_tolerance if citation_tolerance is not None else config.telemetry_citation_tolerance
        )
        self.latency_multiplier = (
            latency_multiplier if latency_multiplier is not None else config.telemetry_latency_multiplier
        )
        self.min_events = min_events if min_events is not None else config.telemetry_min_events

    def build_baseline(self, data: list[dict[str, Any]]) -> dict[str, Any]:
        """data = [{"grounded": bool, "citations": int, "latency_ms": float}, ...]

        A window of normal operation. Everything later is judged against
        how the system behaved here.
        """
        return {"metrics": _metrics(list(data))}

    def check_drift(self, baseline: dict[str, Any], new_data: list[dict[str, Any]]) -> DriftCheckResult:
        base = baseline["metrics"]
        now = _metrics(list(new_data))
        n = int(now["n_events"])
        enough = n >= self.min_events

        refusal_delta = now["refusal_rate"] - base["refusal_rate"]
        citation_delta = base["citation_rate"] - now["citation_rate"]
        latency_ratio = (
            now["latency_p95_ms"] / base["latency_p95_ms"] if base["latency_p95_ms"] > 0 else 1.0
        )

        signals = [
            SignalResult(
                name="refusal rate",
                value=now["refusal_rate"],
                is_drifted=enough and refusal_delta > self.refusal_tolerance,
                detail={
                    "baseline": base["refusal_rate"],
                    "current": now["refusal_rate"],
                    "delta": refusal_delta,
                    "tolerance": self.refusal_tolerance,
                },
            ),
            SignalResult(
                name="citation rate",
                value=now["citation_rate"],
                is_drifted=enough and citation_delta > self.citation_tolerance,
                detail={
                    "baseline": base["citation_rate"],
                    "current": now["citation_rate"],
                    "drop": citation_delta,
                    "tolerance": self.citation_tolerance,
                },
            ),
            SignalResult(
                # Normalised so every signal shares the 0-1 scale the
                # dashboard draws on; the raw milliseconds are in detail.
                name="latency p95",
                value=min(1.0, latency_ratio / (self.latency_multiplier * 2)),
                is_drifted=enough and latency_ratio > self.latency_multiplier,
                detail={
                    "baseline_ms": base["latency_p95_ms"],
                    "current_ms": now["latency_p95_ms"],
                    "ratio": latency_ratio,
                    "max_ratio": self.latency_multiplier,
                },
            ),
        ]

        # Grounded-answer rate, so "higher is better" matches the other
        # adapters and the dashboard's quality tile needs no special case.
        quality = 1.0 - now["refusal_rate"]
        drifted_count = sum(1 for s in signals if s.is_drifted)
        drift_score = drifted_count / len(signals)
        is_drifted = drifted_count > 0

        if not enough:
            logger.info(
                "telemetry batch below min_events, reporting without flagging",
                extra={"n_events": n, "min_events": self.min_events},
            )

        return DriftCheckResult(
            drift_score=drift_score,
            quality_score=quality,
            is_drifted=is_drifted,
            signals=signals,
            statistics={
                "n_events": n,
                "min_events": self.min_events,
                "sufficient_sample": enough,
                "baseline_n_events": int(base.get("n_events", 0)),
                "refusal_rate": now["refusal_rate"],
                "citation_rate": now["citation_rate"],
                "latency_p95_ms": now["latency_p95_ms"],
                "refusal_tolerance": self.refusal_tolerance,
                "citation_tolerance": self.citation_tolerance,
                "latency_max_ratio": self.latency_multiplier,
            },
        )
