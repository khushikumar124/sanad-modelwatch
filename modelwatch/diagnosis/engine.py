"""Root-cause diagnosis for a RAGAdapter drift check: not just "drift
detected", but "which subsystem is most likely responsible, and why".

Input is the `signals` list from a RAGAdapter DriftCheckResult (or the
equivalent dicts from a stored run) -- four named signals: retrieval,
generation_latency, refusal, citation_validity (see
modelwatch/adapters/rag_adapter.py). This module never re-derives
statistics itself; it only reads the confidence/p-value/detail each
signal already carries and applies a fixed, documented rule set to rank
three candidate subsystems:

* RETRIEVAL   -- the vector store / embedding / chunking layer
* GENERATION  -- the LLM's own behavior (prompt, model weights/version)
* OPERATIONAL -- infrastructure (latency, load, resource contention)

The rule set encodes one specific piece of domain reasoning: refusal and
citation-validity drift are *expected side effects* of a retrieval
regression (worse retrieved context -> the model correctly refuses more,
or cites something that no longer supports the answer), so when
retrieval has also drifted, evidence is attributed to RETRIEVAL rather
than double-counted as three independent problems. When retrieval is
*not* drifted but refusal/citation still are, that pattern instead points
at GENERATION -- the model's own behavior changed independent of what it
was given to work with. This mirrors the worked example in
docs/drift_detection.md.

`confidence` for a subsystem is not invented: it's the sum of the
confidence values the contributing signals already report (each in
[0, 1] from their own detector, see modelwatch/drift/detectors.py),
clamped to 1.0. A subsystem hypothesis built on one weakly-confident
signal scores low; one corroborated by two or three signals scores
higher, and the ranking (not just the top pick) is always returned so a
reader can see how close the runner-up was.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

RETRIEVAL = "retrieval"
GENERATION = "generation"
OPERATIONAL = "operational"

_SUPPORTING_EVIDENCE_WEIGHT = 0.5  # a corroborating signal counts for less than its own primary evidence


@dataclass
class DiagnosisResult:
    likely_subsystem: str | None
    confidence: float
    reasoning: list[str]
    ranked: list[tuple[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "likely_subsystem": self.likely_subsystem,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "ranked": [{"subsystem": s, "score": sc} for s, sc in self.ranked],
        }


def _signal_confidence(signal: dict[str, Any]) -> float:
    """Pulls a [0,1] confidence out of a RAGAdapter signal's detail,
    whichever shape it has (a direct detector result, or the retrieval
    signal's {"ks": ..., "wasserstein": ...} pair -- takes the stronger
    of the two, since either alone is sufficient evidence of a shift)."""
    detail = signal.get("detail") or {}
    if "confidence" in detail:
        return float(detail["confidence"])
    if "ks" in detail or "wasserstein" in detail:
        return max(
            float(detail.get("ks", {}).get("confidence", 0.0)),
            float(detail.get("wasserstein", {}).get("confidence", 0.0)),
        )
    return 0.0


def diagnose(signals: list[dict[str, Any]]) -> DiagnosisResult:
    """signals: list of {"name", "is_drifted", "value", "detail"} dicts,
    as produced by RAGAdapter and stored in a run's signals_json."""
    by_name = {s["name"]: s for s in signals}
    drifted = {name: s for name, s in by_name.items() if s.get("is_drifted")}

    if not drifted:
        return DiagnosisResult(None, 0.0, ["no signals are drifted"], [])

    scores = {RETRIEVAL: 0.0, GENERATION: 0.0, OPERATIONAL: 0.0}
    reasoning: list[str] = []

    if "retrieval" in drifted:
        conf = _signal_confidence(drifted["retrieval"])
        scores[RETRIEVAL] += conf
        reasoning.append(f"retrieval similarity distribution shifted (confidence={conf:.2f})")

        for downstream in ("citation_validity", "refusal"):
            if downstream in drifted:
                weight = _signal_confidence(drifted[downstream]) * _SUPPORTING_EVIDENCE_WEIGHT
                scores[RETRIEVAL] += weight
                reasoning.append(
                    f"{downstream.replace('_', ' ')} also declined -- consistent with worse retrieved "
                    "context rather than an independent model change"
                )
            elif downstream in by_name:
                reasoning.append(f"{downstream.replace('_', ' ')} remained stable")
    else:
        for downstream, label in (("citation_validity", "citation validity"), ("refusal", "refusal rate")):
            if downstream in drifted:
                conf = _signal_confidence(drifted[downstream])
                scores[GENERATION] += conf
                reasoning.append(
                    f"{label} shifted (confidence={conf:.2f}) while retrieval stayed stable -- "
                    "consistent with a model/prompt behavior change rather than a retrieval problem"
                )
        if "retrieval" in by_name:
            reasoning.append("retrieval remained stable")

    if "generation_latency" in drifted:
        conf = _signal_confidence(drifted["generation_latency"])
        scores[OPERATIONAL] += conf
        reasoning.append(f"generation latency distribution shifted (confidence={conf:.2f})")
    elif "generation_latency" in by_name:
        reasoning.append("latency remained stable")

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top_subsystem, top_score = ranked[0]
    if top_score <= 0.0:
        return DiagnosisResult(None, 0.0, reasoning, ranked)

    return DiagnosisResult(top_subsystem, min(1.0, top_score), reasoning, ranked)
