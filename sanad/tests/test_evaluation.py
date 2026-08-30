"""Tests for sanad/evaluation/*.

Uses a real VectorStore (indexed from real sample contracts) and a real
Embedder, but a FakeLLMClient with scripted responses -- same pattern as
test_chatbot.py -- so these run fast and deterministically with no Ollama.
"""
from __future__ import annotations

import json

import pytest

from sanad.evaluation.aggregate import summarize
from sanad.evaluation.dataset import EvalCase, load_dataset, save_dataset
from sanad.evaluation.metrics import score_case
from sanad.evaluation.runner import index_documents, run_evaluation
from sanad.ingestion.chunking import chunk_document
from sanad.ingestion.extraction import extract_document
from sanad.rag.llm_client import LLMClient
from sanad.rag.vector_store import VectorStore

SAMPLE_DOC = "sanad/sample_docs/freelance/freelance_agreement_sample2.pdf"


class ScriptedLLMClient(LLMClient):
    """Returns responses in order, one per call, cycling if exhausted."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str, response_schema: dict | None = None, timeout: float = 180) -> str:
        response = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        return response


@pytest.fixture(scope="module")
def indexed_freelance_doc(tmp_path_factory):
    doc = extract_document(SAMPLE_DOC)
    chunks = chunk_document(doc.text)
    store = VectorStore(persist_path=str(tmp_path_factory.mktemp("chroma")))
    store.add_document("freelance-sample-2", chunks)
    return store, chunks


def test_dataset_round_trips_through_jsonl(tmp_path):
    cases = [
        EvalCase(
            id="c1",
            question="Who owns the IP?",
            expected_answer="The company does.",
            relevant_document=SAMPLE_DOC,
            category="intellectual_property",
            relevant_chunks=[3],
            expected_citations=[3],
        )
    ]
    path = tmp_path / "eval.jsonl"
    save_dataset(cases, path)

    loaded = load_dataset(path)
    assert len(loaded) == 1
    assert loaded[0] == cases[0]


def test_shipped_dataset_loads_and_has_expected_shape():
    from datasets.sanad_eval.build_dataset import OUT_PATH

    cases = load_dataset(OUT_PATH)
    assert len(cases) > 0
    for case in cases:
        assert case.question
        assert case.expected_answer
        assert case.relevant_chunks, f"{case.id} has no ground-truth chunks"


def test_score_case_flags_retrieval_hit_and_citation_correctness(indexed_freelance_doc):
    store, chunks = indexed_freelance_doc
    ip_chunk = next(c for c in chunks if "intellectual property" in c.text.lower() or "work for hire" in c.text.lower())

    case = EvalCase(
        id="ip-1",
        question="Who owns the intellectual property in the deliverables?",
        expected_answer="The Deliverables are work for hire and IP vests with the Company.",
        relevant_document=SAMPLE_DOC,
        category="intellectual_property",
        relevant_chunks=[ip_chunk.index],
        expected_citations=[ip_chunk.index],
    )

    from sanad.features.chatbot import ask

    # Cite whichever excerpt number the real retrieval assigned to the IP chunk.
    hits = store.query("freelance-sample-2", case.question, top_k=6)
    cite_num = next(
        i + 1 for i, h in enumerate(hits) if h["metadata"]["chunk_index"] == ip_chunk.index
    )
    response = json.dumps(
        {"grounded": True, "answer": "The Company owns it.", "cited_excerpts": [cite_num]}
    )
    client = ScriptedLLMClient([response])
    answer = ask("freelance-sample-2", case.question, store, client)

    result = score_case(case, answer, latency_ms=100.0)

    assert result.retrieval_hit is True
    assert result.citation_correct is True
    assert result.refused is False
    assert 0.0 <= result.semantic_similarity <= 1.0


def test_score_case_flags_retrieval_miss_when_relevant_chunk_never_retrieved(indexed_freelance_doc):
    store, chunks = indexed_freelance_doc
    case = EvalCase(
        id="miss-1",
        question="Who owns the intellectual property in the deliverables?",
        expected_answer="The Company owns it.",
        relevant_document=SAMPLE_DOC,
        category="intellectual_property",
        relevant_chunks=[99999],  # chunk index that can never be retrieved
        expected_citations=[99999],
    )
    from sanad.features.chatbot import ask

    response = json.dumps({"grounded": False, "answer": "Not found.", "cited_excerpts": []})
    answer = ask("freelance-sample-2", case.question, store, ScriptedLLMClient([response]))

    result = score_case(case, answer, latency_ms=50.0)

    assert result.retrieval_hit is False
    assert result.retrieval_rank is None
    assert result.citation_correct is False


def test_refused_answers_are_excluded_from_citation_correctness_denominator():
    from sanad.evaluation.metrics import CaseResult

    results = [
        CaseResult("a", "cat", True, 1, True, False, False, 100, 10, 90, 0.9, {}),
        CaseResult("b", "cat", True, 1, False, True, False, 100, 10, 90, 0.0, {}),  # refused
    ]
    summary = summarize(results)
    # only the non-refused case counts toward citation_correctness
    assert summary.citation_correctness == 1.0
    assert summary.refusal_rate == 0.5


def test_run_evaluation_end_to_end_with_fake_llm(indexed_freelance_doc):
    store, chunks = indexed_freelance_doc
    case = EvalCase(
        id="e2e-1",
        question="Is the consultant an employee?",
        expected_answer="No, the consultant is an independent contractor.",
        relevant_document=SAMPLE_DOC,
        category="employment_status",
        relevant_chunks=[chunks[0].index],
        expected_citations=[chunks[0].index],
    )
    response = json.dumps(
        {"grounded": True, "answer": "No, independent contractor.", "cited_excerpts": [1]}
    )
    client = ScriptedLLMClient([response])

    doc_ids = {SAMPLE_DOC: "freelance-sample-2"}
    results, summary = run_evaluation([case], store, client, doc_ids=doc_ids)

    assert summary.n_cases == 1
    assert len(results) == 1
    assert client.calls == 1


def test_run_evaluation_top_k_override_changes_retrieved_chunk_count(indexed_freelance_doc, monkeypatch):
    """top_k=None (the default) must behave exactly as before; passing
    an explicit value must actually reach vector_store.query() as the
    real top_k, not get silently dropped."""
    store, chunks = indexed_freelance_doc
    case = EvalCase(
        id="topk-1", question="Is the consultant an employee?",
        expected_answer="No.", relevant_document=SAMPLE_DOC, category="employment_status",
        relevant_chunks=[chunks[0].index], expected_citations=[chunks[0].index],
    )
    response = json.dumps({"grounded": True, "answer": "No.", "cited_excerpts": [1]})
    client = ScriptedLLMClient([response, response])
    doc_ids = {SAMPLE_DOC: "freelance-sample-2"}

    seen_top_k = []
    original_query = store.query
    def spying_query(doc_id, query_text, top_k=None):
        seen_top_k.append(top_k)
        return original_query(doc_id, query_text, top_k=top_k)
    monkeypatch.setattr(store, "query", spying_query)

    run_evaluation([case], store, client, doc_ids=doc_ids, top_k=2)
    run_evaluation([case], store, client, doc_ids=doc_ids, top_k=None)

    from sanad.config import config
    assert seen_top_k == [2, config.retrieval_top_k]


def test_index_documents_reuses_cached_doc_id_for_repeated_source(tmp_path):
    store = VectorStore(persist_path=str(tmp_path / "chroma"))
    cases = [
        EvalCase("c1", "q1", "a1", SAMPLE_DOC, "cat", [0], [0]),
        EvalCase("c2", "q2", "a2", SAMPLE_DOC, "cat", [1], [1]),
    ]
    doc_ids = index_documents(cases, store)
    assert len(doc_ids) == 1
    assert doc_ids[SAMPLE_DOC]
