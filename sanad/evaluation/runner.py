"""Runs an evaluation dataset through Sanad's real retrieval + chatbot
pipeline (in-process: no HTTP, no running server needed) and scores it.

Indexes each unique source document into a scratch vector store once,
then calls the same `sanad.features.chatbot.ask()` the API endpoint
calls, so what's measured here is the actual production code path, not
a reimplementation of it.
"""
from __future__ import annotations

import time
from pathlib import Path

from sanad.evaluation.aggregate import EvalSummary, summarize
from sanad.evaluation.dataset import EvalCase
from sanad.evaluation.metrics import CaseResult, score_case
from sanad.features.chatbot import ask
from sanad.ingestion.chunking import chunk_document
from sanad.ingestion.extraction import extract_document
from sanad.rag.embeddings import Embedder
from sanad.rag.llm_client import LLMClient
from sanad.rag.vector_store import VectorStore


def index_documents(cases: list[EvalCase], vector_store: VectorStore) -> dict[str, str]:
    """Indexes every unique relevant_document referenced by `cases`.
    Returns {source_file: doc_id}."""
    doc_ids: dict[str, str] = {}
    for case in cases:
        source_file = case.relevant_document
        if source_file in doc_ids:
            continue
        doc = extract_document(source_file)
        chunks = chunk_document(doc.text)
        doc_id = Path(source_file).stem
        vector_store.add_document(doc_id, chunks)
        doc_ids[source_file] = doc_id
    return doc_ids


def run_evaluation(
    cases: list[EvalCase],
    vector_store: VectorStore,
    llm_client: LLMClient,
    embedder: Embedder | None = None,
    doc_ids: dict[str, str] | None = None,
) -> tuple[list[CaseResult], EvalSummary]:
    embedder = embedder or Embedder()
    doc_ids = doc_ids if doc_ids is not None else index_documents(cases, vector_store)

    results: list[CaseResult] = []
    for case in cases:
        doc_id = doc_ids[case.relevant_document]
        started = time.perf_counter()
        answer = ask(doc_id, case.question, vector_store, llm_client)
        latency_ms = (time.perf_counter() - started) * 1000
        results.append(score_case(case, answer, latency_ms, embedder=embedder))

    return results, summarize(results)
