"""Builds an observable RAG trace for one answered question: query ->
retrieval (ranked, scored) -> evidence -> generation -> claim-level
verification -> grounding/citation scores.

This exists to make the pipeline inspectable, not to explain the model's
reasoning: everything here is either a direct fact about the pipeline
(which chunks were retrieved, at what distance, which were cited) or a
computed, reproducible check (does this sentence of the answer overlap
enough with retrieved evidence to call it supported). There is no
chain-of-thought here, hidden or otherwise -- an LLM asked to explain
itself is not a reliable source about its own reasoning, and Sanad
doesn't ask.

Claim verification is a heuristic, and is documented as one. It splits
the answer into sentences and scores each against the retrieved chunks
via the same sentence-transformer embeddings the RAG pipeline itself
uses (cosine similarity) -- reusing the "embedding-based" metric
category from sanad/evaluation/metrics.py rather than inventing a new
scoring method. Thresholds (0.55 supported, 0.35 partial) are
uncalibrated defaults, not derived from a validation study; treat the
per-claim labels as a triage aid for a human reader, not a certified
fact-check.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from sanad.features.chatbot import CHATBOT_PROMPT_VERSION, ChatAnswer
from sanad.rag.embeddings import Embedder

SUPPORTED_THRESHOLD = 0.55
PARTIAL_THRESHOLD = 0.35

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_MIN_CLAIM_CHARS = 12


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _embedding_dims(embedder: Embedder) -> int:
    get_dims = getattr(embedder.model, "get_embedding_dimension", None) or embedder.model.get_sentence_embedding_dimension
    return get_dims()


def _split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]
    return [s for s in sentences if len(s) >= _MIN_CLAIM_CHARS]


@dataclass
class RetrievedChunkTrace:
    rank: int
    chunk_index: int
    heading: str | None
    text_preview: str
    distance: float
    similarity: float
    cited: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "chunk_index": self.chunk_index,
            "heading": self.heading,
            "text_preview": self.text_preview,
            "distance": round(self.distance, 4),
            "similarity": round(self.similarity, 4),
            "cited": self.cited,
        }


@dataclass
class ClaimVerification:
    claim: str
    status: str  # "supported" | "partial" | "unsupported"
    best_similarity: float
    best_evidence_chunk_index: int | None
    best_evidence_preview: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "status": self.status,
            "best_similarity": round(self.best_similarity, 4),
            "best_evidence_chunk_index": self.best_evidence_chunk_index,
            "best_evidence_preview": self.best_evidence_preview,
        }


@dataclass
class RAGTrace:
    question: str
    embedding_model: str
    embedding_dims: int
    top_k: int
    model_name: str
    prompt_version: str
    retrieval: list[RetrievedChunkTrace]
    answer: str
    grounded: bool
    claims: list[ClaimVerification]
    grounding_score: float | None
    citation_score: float | None
    retrieval_latency_ms: float
    generation_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "embedding_model": self.embedding_model,
            "embedding_dims": self.embedding_dims,
            "top_k": self.top_k,
            "model_name": self.model_name,
            "prompt_version": self.prompt_version,
            "retrieval": [r.to_dict() for r in self.retrieval],
            "answer": self.answer,
            "grounded": self.grounded,
            "claims": [c.to_dict() for c in self.claims],
            "grounding_score": self.grounding_score,
            "citation_score": self.citation_score,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "generation_latency_ms": self.generation_latency_ms,
        }


def _evidence_sentence_pool(chunks: list[dict]) -> list[tuple[str, int]]:
    """Flattens evidence chunks into (sentence, chunk_index) pairs.
    Comparing a claim sentence against individual evidence sentences
    (rather than whole chunks) avoids the dilution a short claim suffers
    against a long multi-paragraph chunk's averaged embedding -- a real
    match on one sentence inside a chunk should score as a real match,
    not get watered down by three unrelated sentences next to it. Falls
    back to the whole chunk text if it has no sentence-length content
    (e.g. a heading-only fragment)."""
    pool: list[tuple[str, int]] = []
    for chunk in chunks:
        sentences = _split_sentences(chunk["text"])
        if not sentences:
            sentences = [chunk["text"]]
        pool.extend((s, chunk["metadata"]["chunk_index"]) for s in sentences)
    return pool


def _verify_claim(claim: str, chunks: list[dict], embedder: Embedder) -> ClaimVerification:
    pool = _evidence_sentence_pool(chunks)
    if not pool:
        return ClaimVerification(claim, "unsupported", 0.0, None, None)

    claim_vec = embedder.embed_one(claim)
    sentence_vecs = embedder.embed([s for s, _ in pool])

    best_sim, best_sentence, best_chunk_index = -1.0, None, None
    for (sentence, chunk_index), vec in zip(pool, sentence_vecs):
        sim = _cosine(claim_vec, vec)
        if sim > best_sim:
            best_sim, best_sentence, best_chunk_index = sim, sentence, chunk_index

    status = (
        "supported" if best_sim >= SUPPORTED_THRESHOLD
        else "partial" if best_sim >= PARTIAL_THRESHOLD
        else "unsupported"
    )
    return ClaimVerification(claim, status, max(best_sim, 0.0), best_chunk_index, best_sentence)


def build_trace(
    question: str,
    answer: ChatAnswer,
    model_name: str,
    top_k: int,
    embedder: Embedder | None = None,
) -> RAGTrace:
    embedder = embedder or Embedder()
    cited_indices = {c["metadata"]["chunk_index"] for c in answer.cited_chunks}

    retrieval = [
        RetrievedChunkTrace(
            rank=i + 1,
            chunk_index=hit["metadata"]["chunk_index"],
            heading=hit["metadata"].get("heading") or None,
            text_preview=hit["text"][:200] + ("…" if len(hit["text"]) > 200 else ""),
            distance=hit["distance"],
            similarity=max(0.0, 1.0 - hit["distance"]),
            cited=hit["metadata"]["chunk_index"] in cited_indices,
        )
        for i, hit in enumerate(answer.retrieved_chunks)
    ]

    claims: list[ClaimVerification] = []
    grounding_score: float | None = None
    if answer.grounded and answer.answer.strip():
        evidence_pool = answer.cited_chunks or answer.retrieved_chunks
        claims = [_verify_claim(c, evidence_pool, embedder) for c in _split_sentences(answer.answer)]
        if claims:
            weight = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}
            grounding_score = sum(weight[c.status] for c in claims) / len(claims)

    citation_score: float | None = None
    if answer.citations_requested > 0:
        citation_score = len(answer.cited_chunks) / answer.citations_requested
    elif not answer.grounded:
        citation_score = None  # refusal: no citation was expected

    return RAGTrace(
        question=question,
        embedding_model=embedder.model_name,
        embedding_dims=_embedding_dims(embedder),
        top_k=top_k,
        model_name=model_name,
        prompt_version=CHATBOT_PROMPT_VERSION,
        retrieval=retrieval,
        answer=answer.answer,
        grounded=answer.grounded,
        claims=claims,
        grounding_score=grounding_score,
        citation_score=citation_score,
        retrieval_latency_ms=answer.retrieval_latency_ms,
        generation_latency_ms=answer.generation_latency_ms,
    )
