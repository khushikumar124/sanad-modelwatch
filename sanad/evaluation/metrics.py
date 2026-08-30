"""RAG evaluation metrics for one (EvalCase, ChatAnswer) pair.

Every metric here is labelled by how it's computed, because they carry
different amounts of trust:

* deterministic  -- exact set/boolean comparisons. No model in the loop.
* embedding      -- cosine similarity of sentence-transformer embeddings.
                     Not exact, but reproducible: same inputs, same output.
* statistical    -- aggregates (mean, stdev) computed only in aggregate.py,
                     over a batch of the metrics below.

There is deliberately no LLM-judge metric yet -- see docs/evaluation.md's
limitations section. Add one only behind an explicit flag, never as a
silent default, since it introduces a second model's own quality/cost/
non-determinism into what is otherwise a reproducible evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sanad.evaluation.dataset import EvalCase
from sanad.features.chatbot import ChatAnswer
from sanad.rag.embeddings import Embedder


def _chunk_indices(chunks: list[dict]) -> list[int]:
    return [c["metadata"]["chunk_index"] for c in chunks]


@dataclass
class CaseResult:
    case_id: str
    category: str

    # -- deterministic --------------------------------------------------
    retrieval_hit: bool  # was any relevant_chunk among the retrieved chunks
    retrieval_rank: int | None  # 1-indexed rank of the first relevant chunk retrieved, None if missed
    citation_correct: bool  # did the answer cite at least one relevant chunk (only meaningful if grounded)
    refused: bool  # answer declined to answer (not grounded)
    parse_error: bool
    latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float

    # -- embedding-based --------------------------------------------------
    semantic_similarity: float  # cosine sim of expected vs actual answer embeddings

    detail: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "deterministic": {
                "retrieval_hit": self.retrieval_hit,
                "retrieval_rank": self.retrieval_rank,
                "citation_correct": self.citation_correct,
                "refused": self.refused,
                "parse_error": self.parse_error,
                "latency_ms": self.latency_ms,
                "retrieval_latency_ms": self.retrieval_latency_ms,
                "generation_latency_ms": self.generation_latency_ms,
            },
            "embedding": {
                "semantic_similarity": self.semantic_similarity,
            },
            "detail": self.detail,
        }


def score_case(
    case: EvalCase,
    answer: ChatAnswer,
    latency_ms: float,
    embedder: Embedder | None = None,
) -> CaseResult:
    """Score one evaluated case. `latency_ms` is the caller's own
    end-to-end timer (mirrors how the API endpoint measures it); the
    retrieval/generation split comes from the answer itself."""
    embedder = embedder or Embedder()

    retrieved_indices = _chunk_indices(answer.retrieved_chunks)
    relevant = set(case.relevant_chunks)

    retrieval_hit = bool(relevant & set(retrieved_indices))
    retrieval_rank = None
    for rank, idx in enumerate(retrieved_indices, start=1):
        if idx in relevant:
            retrieval_rank = rank
            break

    cited_indices = set(_chunk_indices(answer.cited_chunks))
    citation_correct = bool(cited_indices & relevant) if answer.grounded else False

    if answer.answer.strip() and case.expected_answer.strip():
        vectors = embedder.embed([case.expected_answer, answer.answer])
        semantic_similarity = _cosine(vectors[0], vectors[1])
    else:
        semantic_similarity = 0.0

    return CaseResult(
        case_id=case.id,
        category=case.category,
        retrieval_hit=retrieval_hit,
        retrieval_rank=retrieval_rank,
        citation_correct=citation_correct,
        refused=not answer.grounded,
        parse_error=answer.parse_error,
        latency_ms=latency_ms,
        retrieval_latency_ms=answer.retrieval_latency_ms,
        generation_latency_ms=answer.generation_latency_ms,
        semantic_similarity=semantic_similarity,
        detail={
            "relevant_chunks": sorted(relevant),
            "retrieved_chunks": retrieved_indices,
            "cited_chunks": sorted(cited_indices),
        },
    )


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
