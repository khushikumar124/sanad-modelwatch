"""Aggregates a batch of CaseResult into summary statistics.

This is the "statistical" layer referenced in metrics.py's docstring:
means/rates over a batch, plus a mean-only breakdown per category so a
regression that's concentrated in one clause type (e.g. citation
correctness collapsing only for "dispute_resolution" questions) doesn't
get diluted into a moving-but-not-alarming overall number.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from sanad.evaluation.metrics import CaseResult


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round(pct / 100 * len(ordered) + 0.5)) - 1))
    return ordered[k]


@dataclass
class EvalSummary:
    n_cases: int
    retrieval_hit_rate: float
    citation_correctness: float
    refusal_rate: float
    parse_error_rate: float
    mean_semantic_similarity: float
    latency_p95_ms: float
    retrieval_latency_p95_ms: float
    generation_latency_p95_ms: float
    by_category: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_cases": self.n_cases,
            "retrieval_hit_rate": self.retrieval_hit_rate,
            "citation_correctness": self.citation_correctness,
            "refusal_rate": self.refusal_rate,
            "parse_error_rate": self.parse_error_rate,
            "mean_semantic_similarity": self.mean_semantic_similarity,
            "latency_p95_ms": self.latency_p95_ms,
            "retrieval_latency_p95_ms": self.retrieval_latency_p95_ms,
            "generation_latency_p95_ms": self.generation_latency_p95_ms,
            "by_category": self.by_category,
        }


def summarize(results: list[CaseResult]) -> EvalSummary:
    n = len(results)
    if n == 0:
        return EvalSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {})

    grounded_results = [r for r in results if not r.refused]

    by_category: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        by_category[r.category].append(r)

    return EvalSummary(
        n_cases=n,
        retrieval_hit_rate=sum(r.retrieval_hit for r in results) / n,
        # citation correctness is scoped to answers that were actually
        # grounded -- a refusal has no citation to be correct or wrong.
        citation_correctness=(
            sum(r.citation_correct for r in grounded_results) / len(grounded_results)
            if grounded_results
            else 0.0
        ),
        refusal_rate=sum(r.refused for r in results) / n,
        parse_error_rate=sum(r.parse_error for r in results) / n,
        mean_semantic_similarity=_mean([r.semantic_similarity for r in results]),
        latency_p95_ms=_percentile([r.latency_ms for r in results], 95),
        retrieval_latency_p95_ms=_percentile([r.retrieval_latency_ms for r in results], 95),
        generation_latency_p95_ms=_percentile([r.generation_latency_ms for r in results], 95),
        by_category={
            category: {
                "n_cases": len(rs),
                "retrieval_hit_rate": sum(r.retrieval_hit for r in rs) / len(rs),
                "mean_semantic_similarity": _mean([r.semantic_similarity for r in rs]),
            }
            for category, rs in by_category.items()
        },
    )
