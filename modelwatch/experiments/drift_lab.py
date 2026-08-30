"""Drift Lab: controlled interventions on Sanad's real RAG pipeline, used
to check whether RAGAdapter (modelwatch/adapters/rag_adapter.py) and the
diagnosis engine (modelwatch/diagnosis/engine.py) actually detect and
correctly attribute the kind of degradation each intervention causes.

Architectural note: everywhere else in modelwatch/core and
modelwatch/adapters is deliberately model-agnostic (see
modelwatch/core/adapter_base.py's docstring) -- this module is the one
documented exception, in the same place modelwatch/examples/*.py already
is: it exists specifically to exercise ModelWatch against its first real
integration, Sanad, so it imports Sanad concretely rather than through
an adapter.

Every scenario runs Sanad's real code path (sanad.features.chatbot.ask
against a real VectorStore) with one thing deliberately changed, and
reports MEASURED effects -- there is no fabricated "expected detection"
here (see Phase 34's data-integrity rule): a scenario can legitimately
fail to trip RAGAdapter's thresholds, and run() reports that honestly
rather than asserting it must have worked.

Scenarios:
  - retrieval_narrowing: shrink top_k so far that the correct chunk is
    often not retrieved at all -- simulates a retrieval regression
    (a broken/degraded reranker, an index that lost recall).
  - chunk_fragmentation: re-chunk source documents into much smaller
    pieces, fragmenting clauses across chunk boundaries -- simulates a
    chunking-pipeline regression (bad config, or a document type the
    clause-boundary heuristics don't suit).

Both require Ollama serving Sanad's configured model -- there is no
fake-LLM path here, for the same reason sanad/evaluation/run_eval.py has
none: the whole point is to measure what the real model actually does
under the intervention.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from modelwatch.adapters.rag_adapter import RAGAdapter
from modelwatch.diagnosis.engine import DiagnosisResult, diagnose
from sanad.evaluation.dataset import EvalCase, load_dataset
from sanad.features.chatbot import ChatAnswer, ask
from sanad.ingestion.chunking import chunk_document
from sanad.ingestion.extraction import extract_document
from sanad.rag.llm_client import LLMClient
from sanad.rag.vector_store import VectorStore

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "datasets" / "sanad_eval" / "sanad_eval_v1.jsonl"


def _to_event(answer: ChatAnswer) -> dict[str, Any]:
    """Shapes a real ChatAnswer into the ChatEvent-like dict RAGAdapter
    expects -- the same shape Sanad's own telemetry emits (see
    sanad/api/telemetry.py), so a scenario's synthetic traffic is
    interchangeable with real production events."""
    return {
        "grounded": answer.grounded,
        "citations": len(answer.cited_chunks),
        "citations_requested": answer.citations_requested,
        "retrieval_scores": [c["distance"] for c in answer.retrieved_chunks],
        "generation_latency_ms": answer.generation_latency_ms,
    }


@dataclass
class ScenarioResult:
    scenario: str
    n_cases: int
    baseline_events: list[dict[str, Any]]
    current_events: list[dict[str, Any]]
    drift_result: Any  # DriftCheckResult
    diagnosis: DiagnosisResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "n_cases": self.n_cases,
            "drift_result": self.drift_result.to_dict(),
            "diagnosis": self.diagnosis.to_dict(),
        }

    def print_report(self) -> None:
        print(f"\n=== Drift Lab: {self.scenario} ({self.n_cases} cases) ===")
        print(f"is_drifted:      {self.drift_result.is_drifted}")
        print(f"drift_score:     {self.drift_result.drift_score:.3f}")
        for s in self.drift_result.signals:
            flag = " <- DRIFTED" if s.is_drifted else ""
            print(f"  {s.name:<20} value={s.value:.3f}{flag}")
        print(f"diagnosis:       {self.diagnosis.likely_subsystem} (confidence={self.diagnosis.confidence:.2f})")
        for line in self.diagnosis.reasoning:
            print(f"  - {line}")


def _run_cases(
    cases: list[EvalCase],
    vector_store: VectorStore,
    llm_client: LLMClient,
    doc_ids: dict[str, str],
    ask_fn: Callable[..., ChatAnswer],
) -> list[dict[str, Any]]:
    events = []
    for case in cases:
        doc_id = doc_ids[case.relevant_document]
        answer = ask_fn(doc_id, case.question, vector_store, llm_client)
        events.append(_to_event(answer))
    return events


def _index(cases: list[EvalCase], vector_store: VectorStore, max_chars: int | None = None) -> dict[str, str]:
    doc_ids: dict[str, str] = {}
    for case in cases:
        source_file = case.relevant_document
        if source_file in doc_ids:
            continue
        doc = extract_document(source_file)
        chunks = chunk_document(doc.text, max_chars=max_chars) if max_chars else chunk_document(doc.text)
        doc_id = f"{Path(source_file).stem}-{max_chars or 'default'}"
        vector_store.add_document(doc_id, chunks)
        doc_ids[source_file] = doc_id
    return doc_ids


def _build_result(scenario: str, baseline_events: list[dict], current_events: list[dict]) -> ScenarioResult:
    adapter = RAGAdapter(min_events=max(1, len(baseline_events) // 2))
    baseline = adapter.build_baseline(baseline_events)
    drift_result = adapter.check_drift(baseline, current_events)
    diagnosis = diagnose([s.to_dict() for s in drift_result.signals])
    return ScenarioResult(scenario, len(current_events), baseline_events, current_events, drift_result, diagnosis)


def retrieval_narrowing(
    cases: list[EvalCase],
    llm_client: LLMClient,
    baseline_top_k: int = 6,
    degraded_top_k: int = 1,
    chroma_path: str | None = None,
) -> ScenarioResult:
    """Shrinks top_k from `baseline_top_k` to `degraded_top_k` so the real
    supporting chunk is retrieved far less often, simulating a retrieval
    regression -- without touching the index or the model at all."""
    vector_store = VectorStore(persist_path=chroma_path or f"/tmp/driftlab_retrieval_{int(time.time())}")
    doc_ids = _index(cases, vector_store)

    def ask_baseline(doc_id, question, store, client):
        return ask(doc_id, question, store, client, top_k=baseline_top_k)

    def ask_degraded(doc_id, question, store, client):
        return ask(doc_id, question, store, client, top_k=degraded_top_k)

    baseline_events = _run_cases(cases, vector_store, llm_client, doc_ids, ask_baseline)
    current_events = _run_cases(cases, vector_store, llm_client, doc_ids, ask_degraded)
    return _build_result("retrieval_narrowing", baseline_events, current_events)


def chunk_fragmentation(
    cases: list[EvalCase],
    llm_client: LLMClient,
    baseline_max_chars: int = 1500,
    degraded_max_chars: int = 200,
    chroma_path: str | None = None,
) -> ScenarioResult:
    """Re-chunks source documents into much smaller pieces, fragmenting
    clauses across boundaries -- simulates a chunking-pipeline
    regression. Baseline and degraded chunkings are indexed as separate
    documents in the same store so both can be queried from one run."""
    vector_store = VectorStore(persist_path=chroma_path or f"/tmp/driftlab_chunking_{int(time.time())}")
    baseline_doc_ids = _index(cases, vector_store, max_chars=baseline_max_chars)
    degraded_doc_ids = _index(cases, vector_store, max_chars=degraded_max_chars)

    baseline_events = _run_cases(cases, vector_store, llm_client, baseline_doc_ids, ask)
    current_events = _run_cases(cases, vector_store, llm_client, degraded_doc_ids, ask)
    return _build_result("chunk_fragmentation", baseline_events, current_events)


SCENARIOS: dict[str, Callable[..., ScenarioResult]] = {
    "retrieval_narrowing": retrieval_narrowing,
    "chunk_fragmentation": chunk_fragmentation,
}
