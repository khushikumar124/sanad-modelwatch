"""Per-request diagnosis: given ONE RAG trace (sanad/features/trace.py's
output, as stored by ModelWatch's traces table), classify why an answer
looks bad -- not "drift detected" across a batch, but "what likely went
wrong on this one request".

This is a different question from modelwatch/diagnosis/engine.py's
diagnose(), which ranks a subsystem from a *batch's* statistical drift
signals. A single trace has no distribution to test, so there is no
p-value to report here and none is invented: `TraceDiagnosis.evidence`
holds the actual numbers the rule looked at (best retrieval similarity,
citation/grounding scores, latency), and the reader can judge the call
themselves rather than trusting a fabricated confidence.

Taxonomy and the rule that assigns each:

  retrieval_miss        -- nothing was retrieved at all (empty index,
                            or a doc_id with no indexed content)
  irrelevant_retrieval   -- something was retrieved, but even the best
                            match is a poor semantic fit for the
                            question (low similarity) and the model
                            correctly refused
  insufficient_evidence  -- retrieval found a plausible but not strong
                            match, and the model correctly refused
                            rather than guess
  generation_problem     -- retrieval surfaced good evidence, but the
                            model still refused, or claims in its
                            answer don't hold up against what it was
                            given -- the failure is downstream of
                            retrieval
  citation_problem       -- the model was grounded and had good
                            evidence, but named citations that don't
                            hold up (citation_score < 1.0)
  operational_problem    -- generation or retrieval took unusually
                            long, independent of quality (can co-occur
                            with any of the above -- reported as a
                            secondary note, not a replacement category)
  none                   -- no problem pattern matched; looks healthy

One documented limitation: "query problem" (the question itself was
malformed, out of domain, or embedded poorly) is NOT separately
distinguished from irrelevant_retrieval here. Telling those apart from
one trace alone would need a reference distribution of "normal" query
embeddings to compare against (that's exactly what RAGAdapter's
retrieval signal does at the batch level, see
modelwatch/adapters/rag_adapter.py) -- a single trace has nothing to
compare its own query against, so this module folds both into
irrelevant_retrieval rather than inventing a distinction it can't
actually make.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

RETRIEVAL_MISS = "retrieval_miss"
IRRELEVANT_RETRIEVAL = "irrelevant_retrieval"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
GENERATION_PROBLEM = "generation_problem"
CITATION_PROBLEM = "citation_problem"
NONE_CATEGORY = "none"

#: Similarity bands for a refusal's best-retrieved-chunk score. Uncalibrated
#: defaults (not derived from a validation study, like trace.py's claim
#: thresholds) -- a reasonable starting split, not a certified boundary.
_IRRELEVANT_MAX_SIM = 0.30
_INSUFFICIENT_MAX_SIM = 0.50

#: Absolute latency past which a stage is flagged regardless of category --
#: high because a 3-8B local model on CPU is already slow; this is meant
#: to catch a real anomaly (a stuck request, a swapped-in bigger model),
#: not to complain about ordinary local-LLM latency.
_SLOW_MS = 15_000


@dataclass
class TraceDiagnosis:
    category: str
    reasoning: list[str]
    evidence: dict[str, Any]
    operational_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "reasoning": self.reasoning,
            "evidence": self.evidence,
            "operational_note": self.operational_note,
        }


def diagnose_trace(trace: dict[str, Any]) -> TraceDiagnosis:
    retrieval = trace.get("retrieval") or []
    grounded = trace.get("grounded", False)
    claims = trace.get("claims") or []
    citation_score = trace.get("citation_score")
    grounding_score = trace.get("grounding_score")
    retrieval_latency = trace.get("retrieval_latency_ms", 0.0)
    generation_latency = trace.get("generation_latency_ms", 0.0)

    best_similarity = max((r.get("similarity", 0.0) for r in retrieval), default=0.0)
    evidence = {
        "retrieved_count": len(retrieval),
        "best_similarity": round(best_similarity, 4),
        "grounded": grounded,
        "citation_score": citation_score,
        "grounding_score": grounding_score,
        "retrieval_latency_ms": retrieval_latency,
        "generation_latency_ms": generation_latency,
    }

    operational_note = None
    if retrieval_latency > _SLOW_MS:
        operational_note = f"retrieval took {retrieval_latency:.0f}ms, unusually slow"
    elif generation_latency > _SLOW_MS:
        operational_note = f"generation took {generation_latency:.0f}ms, unusually slow"

    if not retrieval:
        return TraceDiagnosis(
            RETRIEVAL_MISS,
            ["no chunks were retrieved at all -- empty index, or the document has no indexed content"],
            evidence,
            operational_note,
        )

    if not grounded:
        if best_similarity < _IRRELEVANT_MAX_SIM:
            return TraceDiagnosis(
                IRRELEVANT_RETRIEVAL,
                [
                    f"best retrieval similarity was only {best_similarity:.2f} -- even the closest "
                    "match is a poor fit for the question, so refusing was likely correct"
                ],
                evidence,
                operational_note,
            )
        if best_similarity < _INSUFFICIENT_MAX_SIM:
            return TraceDiagnosis(
                INSUFFICIENT_EVIDENCE,
                [
                    f"best retrieval similarity was {best_similarity:.2f} -- plausibly related, but "
                    "not strong enough to answer from, so refusing was a reasonable call"
                ],
                evidence,
                operational_note,
            )
        return TraceDiagnosis(
            GENERATION_PROBLEM,
            [
                f"best retrieval similarity was {best_similarity:.2f}, a good match, but the model "
                "refused anyway -- the evidence was there and the model didn't use it"
            ],
            evidence,
            operational_note,
        )

    # grounded == True from here on
    reasoning: list[str] = []
    if citation_score is not None and citation_score < 1.0:
        reasoning.append(
            f"citation score {citation_score:.2f} -- some citations the model named didn't point at "
            "a real excerpt"
        )
        return TraceDiagnosis(CITATION_PROBLEM, reasoning, evidence, operational_note)

    unsupported = [c for c in claims if c.get("status") == "unsupported"]
    if unsupported:
        reasoning.append(
            f"{len(unsupported)} of {len(claims)} claim(s) in the answer don't hold up against the "
            "retrieved evidence, despite the answer being marked grounded"
        )
        return TraceDiagnosis(GENERATION_PROBLEM, reasoning, evidence, operational_note)

    if grounding_score is not None and grounding_score < 0.5:
        reasoning.append(f"grounding score {grounding_score:.2f} is low despite being marked grounded")
        return TraceDiagnosis(GENERATION_PROBLEM, reasoning, evidence, operational_note)

    reasoning.append("retrieval, citations and claim verification all look consistent")
    return TraceDiagnosis(NONE_CATEGORY, reasoning, evidence, operational_note)
