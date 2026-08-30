"""Counterfactual experiments: run the SAME real evaluation dataset
through Sanad's real pipeline under different configurations, and
compare real measured quality/latency -- never simulated numbers.

Architectural note: same documented exception as
modelwatch/experiments/drift_lab.py -- this imports Sanad concretely
rather than through an adapter, because its entire purpose is
exercising Sanad's real pipeline, not staying model-agnostic.

Two comparisons are supported, both sharing one indexed corpus across
variants since neither requires re-indexing:
  - compare_top_k: varies how many chunks are retrieved per question.
  - compare_models: varies which installed Ollama model generates the
    answer (retrieval and indexing stay identical across variants).

NOT supported: comparing chunk size or embedding model, since either
would change what's actually indexed, requiring a distinct doc_id per
variant (see drift_lab.py's chunk_fragmentation for that pattern) --
a real next step, not implemented here, so this module doesn't claim
to support it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sanad.evaluation.dataset import EvalCase
from sanad.evaluation.runner import index_documents, run_evaluation
from sanad.rag.llm_client import LLMClient, OllamaClient
from sanad.rag.vector_store import VectorStore

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "datasets" / "sanad_eval" / "sanad_eval_v1.jsonl"


@dataclass
class CounterfactualResult:
    n_cases: int
    variants: list[dict[str, Any]]  # [{top_k, summary: EvalSummary.to_dict()}, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"n_cases": self.n_cases, "variants": self.variants}


def compare_top_k(
    cases: list[EvalCase],
    llm_client: LLMClient,
    top_k_values: list[int],
    chroma_path: str | None = None,
) -> CounterfactualResult:
    """Indexes `cases`' documents once, then runs the full evaluation
    once per value in `top_k_values` against that same index -- so any
    difference in the results is attributable to top_k, not to a
    different indexing run."""
    vector_store = VectorStore(persist_path=chroma_path or f"/tmp/counterfactual_topk_{int(time.time())}")
    doc_ids = index_documents(cases, vector_store)

    variants = []
    for top_k in top_k_values:
        _results, summary = run_evaluation(cases, vector_store, llm_client, doc_ids=doc_ids, top_k=top_k)
        variants.append({"top_k": top_k, "summary": summary.to_dict()})

    return CounterfactualResult(n_cases=len(cases), variants=variants)


def compare_models(
    cases: list[EvalCase],
    model_names: list[str],
    chroma_path: str | None = None,
) -> CounterfactualResult:
    """Same one-index-many-runs pattern as compare_top_k, but varying
    the Ollama model instead of top_k -- safe to share the index across
    models for the same reason: indexing (extraction, chunking,
    embedding) doesn't depend on which LLM generates the answer, only
    retrieval does, and retrieval isn't what's being varied here.

    Each model_name is passed to a fresh OllamaClient(model=name), so
    this compares whatever Ollama models are actually installed locally
    (see `ollama list`) -- it does not download or validate a model
    exists before running; a missing model will surface as a real
    connection/generation error on that variant, not a fabricated
    result for it.
    """
    vector_store = VectorStore(persist_path=chroma_path or f"/tmp/counterfactual_models_{int(time.time())}")
    doc_ids = index_documents(cases, vector_store)

    variants = []
    for model_name in model_names:
        client = OllamaClient(model=model_name)
        _results, summary = run_evaluation(cases, vector_store, client, doc_ids=doc_ids)
        variants.append({"model": model_name, "summary": summary.to_dict()})

    return CounterfactualResult(n_cases=len(cases), variants=variants)
