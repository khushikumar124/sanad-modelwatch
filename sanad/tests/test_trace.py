"""Tests for sanad/features/trace.py -- real VectorStore (indexed from a
real sample contract) and a real Embedder, scripted LLM responses, same
pattern as test_chatbot.py."""
import json

import pytest

from sanad.features.chatbot import ask
from sanad.features.trace import build_trace
from sanad.ingestion.chunking import chunk_document
from sanad.ingestion.extraction import extract_document
from sanad.rag.llm_client import LLMClient
from sanad.rag.vector_store import VectorStore

SAMPLE_DOC = "sanad/sample_docs/freelance/freelance_agreement_sample2.pdf"


class ScriptedLLMClient(LLMClient):
    def __init__(self, response: str):
        self.response = response

    def generate(self, system_prompt: str, user_prompt: str, response_schema: dict | None = None) -> str:
        return self.response


@pytest.fixture(scope="module")
def indexed_doc_id(tmp_path_factory):
    doc = extract_document(SAMPLE_DOC)
    chunks = chunk_document(doc.text)
    store = VectorStore(persist_path=str(tmp_path_factory.mktemp("chroma")))
    store.add_document("trace-test-doc", chunks)
    return store


def test_trace_records_ranked_retrieval_with_scores(indexed_doc_id):
    store = indexed_doc_id
    response = json.dumps({"grounded": True, "answer": "Payment is made per milestone.", "cited_excerpts": [1]})
    answer = ask("trace-test-doc", "What are the payment terms?", store, ScriptedLLMClient(response))

    trace = build_trace("What are the payment terms?", answer, model_name="test-model", top_k=6, embedder=store.embedder)

    assert len(trace.retrieval) == len(answer.retrieved_chunks)
    ranks = [r.rank for r in trace.retrieval]
    assert ranks == sorted(ranks)  # ranked in retrieval order, 1-indexed
    assert trace.retrieval[0].rank == 1
    # the cited chunk (excerpt 1 == first retrieved hit) is flagged cited
    assert trace.retrieval[0].cited is True
    assert all(0.0 <= r.similarity <= 1.0 for r in trace.retrieval)


def test_grounded_answer_gets_claim_verification(indexed_doc_id):
    store = indexed_doc_id
    response = json.dumps(
        {"grounded": True, "answer": "The Deliverables are work for hire. IP vests with the Company.", "cited_excerpts": [1, 2]}
    )
    answer = ask("trace-test-doc", "Who owns the IP?", store, ScriptedLLMClient(response))

    trace = build_trace("Who owns the IP?", answer, model_name="test-model", top_k=6, embedder=store.embedder)

    assert len(trace.claims) == 2  # two sentences
    assert trace.grounding_score is not None
    assert 0.0 <= trace.grounding_score <= 1.0
    for claim in trace.claims:
        assert claim.status in ("supported", "partial", "unsupported")


def test_refusal_gets_no_claims_and_no_grounding_score(indexed_doc_id):
    store = indexed_doc_id
    response = json.dumps({"grounded": False, "answer": "The document does not address this.", "cited_excerpts": []})
    answer = ask("trace-test-doc", "What is the weather in Mumbai?", store, ScriptedLLMClient(response))

    trace = build_trace("What is the weather in Mumbai?", answer, model_name="test-model", top_k=6, embedder=store.embedder)

    assert trace.claims == []
    assert trace.grounding_score is None
    assert trace.citation_score is None


def test_citation_score_reflects_valid_vs_requested_ratio(indexed_doc_id):
    store = indexed_doc_id
    # model requests 2 citations but only excerpt 1 is a real, valid index
    response = json.dumps(
        {"grounded": True, "answer": "Payment is made per milestone.", "cited_excerpts": [1, 99]}
    )
    answer = ask("trace-test-doc", "What are the payment terms?", store, ScriptedLLMClient(response))

    trace = build_trace("What are the payment terms?", answer, model_name="test-model", top_k=6, embedder=store.embedder)

    assert trace.citation_score == pytest.approx(0.5)  # 1 valid / 2 requested


def test_a_clearly_supported_claim_is_labeled_supported(indexed_doc_id):
    """The claim text is a near-exact echo of the real IP clause, so it
    must score above the 'supported' threshold against it."""
    store = indexed_doc_id
    response = json.dumps(
        {
            "grounded": True,
            "answer": "The Deliverables are deemed work for hire and Intellectual Property Rights vest solely with the Company.",
            "cited_excerpts": [1],
        }
    )
    answer = ask("trace-test-doc", "Who owns the IP?", store, ScriptedLLMClient(response))
    trace = build_trace("Who owns the IP?", answer, model_name="test-model", top_k=6, embedder=store.embedder)

    assert len(trace.claims) == 1
    assert trace.claims[0].status == "supported"
    assert trace.claims[0].best_evidence_chunk_index is not None


def test_trace_is_json_serializable(indexed_doc_id):
    store = indexed_doc_id
    response = json.dumps({"grounded": True, "answer": "Payment is made per milestone.", "cited_excerpts": [1]})
    answer = ask("trace-test-doc", "What are the payment terms?", store, ScriptedLLMClient(response))
    trace = build_trace("What are the payment terms?", answer, model_name="test-model", top_k=6, embedder=store.embedder)
    json.dumps(trace.to_dict())
