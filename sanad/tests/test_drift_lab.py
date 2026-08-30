"""Tests for modelwatch/experiments/drift_lab.py's scenario mechanics.

Lives under sanad/tests (not modelwatch/tests) because it exercises real
Sanad retrieval, matching where test_evaluation.py and test_chatbot.py
live -- and because modelwatch's own test suite intentionally has no
Sanad-specific dependencies (see drift_lab.py's own architecture note).

Uses a FakeLLMClient with a fixed response, so what's actually verified
here is the scenario *mechanics* (top_k is really applied, chunking
config is really applied, the resulting events reach RAGAdapter and
diagnose() without error) rather than the statistics themselves, which
are already covered by test_rag_adapter.py and test_diagnosis.py against
controlled synthetic distributions.
"""
from __future__ import annotations

import json

from modelwatch.experiments.drift_lab import chunk_fragmentation, retrieval_narrowing
from sanad.evaluation.dataset import EvalCase
from sanad.rag.llm_client import LLMClient

SAMPLE_DOC = "sanad/sample_docs/freelance/freelance_agreement_sample2.pdf"

CASES = [
    EvalCase(
        id="ip-1",
        question="Who owns the intellectual property in the deliverables?",
        expected_answer="The Company owns it.",
        relevant_document=SAMPLE_DOC,
        category="intellectual_property",
        relevant_chunks=[0],
        expected_citations=[0],
    ),
    EvalCase(
        id="term-1",
        question="Is the consultant an employee?",
        expected_answer="No, independent contractor.",
        relevant_document=SAMPLE_DOC,
        category="employment_status",
        relevant_chunks=[0],
        expected_citations=[0],
    ),
]


class FixedLLMClient(LLMClient):
    def generate(self, system_prompt: str, user_prompt: str, response_schema: dict | None = None) -> str:
        return json.dumps({"grounded": True, "answer": "A fixed answer.", "cited_excerpts": [1]})


def test_retrieval_narrowing_actually_changes_retrieved_chunk_count():
    result = retrieval_narrowing(
        CASES, FixedLLMClient(), baseline_top_k=6, degraded_top_k=1, chroma_path="/tmp/test_driftlab_retrieval"
    )
    assert result.n_cases == len(CASES)
    # baseline retrieved up to 6 chunks per case, degraded only 1
    assert len(result.baseline_events[0]["retrieval_scores"]) <= 6
    assert len(result.current_events[0]["retrieval_scores"]) == 1
    assert result.drift_result is not None
    assert result.diagnosis is not None


def test_chunk_fragmentation_produces_more_chunks_at_smaller_max_chars():
    from sanad.ingestion.chunking import chunk_document
    from sanad.ingestion.extraction import extract_document

    doc = extract_document(SAMPLE_DOC)
    coarse = chunk_document(doc.text, max_chars=1500)
    fine = chunk_document(doc.text, max_chars=200)
    assert len(fine) > len(coarse)


def test_chunk_fragmentation_scenario_runs_end_to_end():
    result = chunk_fragmentation(
        CASES,
        FixedLLMClient(),
        baseline_max_chars=1500,
        degraded_max_chars=200,
        chroma_path="/tmp/test_driftlab_chunking",
    )
    assert result.n_cases == len(CASES)
    assert len(result.baseline_events) == len(CASES)
    assert len(result.current_events) == len(CASES)


def test_scenario_result_serializes_to_json():
    result = retrieval_narrowing(
        CASES, FixedLLMClient(), baseline_top_k=6, degraded_top_k=1, chroma_path="/tmp/test_driftlab_retrieval2"
    )
    json.dumps(result.to_dict())
