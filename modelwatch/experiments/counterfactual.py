"""Counterfactual experiments: run the SAME real evaluation dataset
through Sanad's real pipeline under different top_k values, and compare
real measured quality/latency -- never simulated numbers.

Architectural note: same documented exception as
modelwatch/experiments/drift_lab.py -- this imports Sanad concretely
rather than through an adapter, because its entire purpose is
exercising Sanad's real pipeline, not staying model-agnostic.

Scoped to top_k only, not chunk size or embedding model: varying top_k
doesn't require re-indexing (the same indexed documents are reused for
every value, only how many chunks are retrieved per question changes),
so there's no risk of stale ChromaDB state across variants. Comparing
chunk sizes or embedding models would each require re-indexing under a
distinct doc_id per configuration (see drift_lab.py's chunk_fragmentation
for the pattern) -- a real next step, not implemented here, so this
module doesn't claim to support it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sanad.evaluation.dataset import EvalCase
from sanad.evaluation.runner import index_documents, run_evaluation
from sanad.rag.llm_client import LLMClient
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
