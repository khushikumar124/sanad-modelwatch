"""Chatbot tests using a real VectorStore (indexed from a real sample
contract) but a FakeLLMClient, so retrieval is real while the model's
answer is controlled for deterministic assertions -- no Ollama required.
"""
import json
import uuid

import pytest

from sanad.features.chatbot import ask
from sanad.ingestion.chunking import chunk_document
from sanad.ingestion.extraction import extract_document
from sanad.rag.llm_client import LLMClient
from sanad.rag.vector_store import VectorStore

SAMPLE_DOC = "sanad/sample_docs/freelance/freelance_agreement_sample2.pdf"


class FakeLLMClient(LLMClient):
    def __init__(self, response: str):
        self.response = response
        self.called = False

    def generate(self, system_prompt: str, user_prompt: str, response_schema: dict | None = None, timeout: float = 180) -> str:
        self.called = True
        return self.response


class NeverCalledLLMClient(LLMClient):
    def generate(self, system_prompt: str, user_prompt: str, response_schema: dict | None = None, timeout: float = 180) -> str:
        raise AssertionError("LLM should not be called when there is no retrieved context")


@pytest.fixture(scope="module")
def indexed_doc_id(tmp_path_factory):
    doc = extract_document(SAMPLE_DOC)
    chunks = chunk_document(doc.text)
    store = VectorStore(persist_path=str(tmp_path_factory.mktemp("chroma")))
    doc_id = f"test-{uuid.uuid4()}"
    store.add_document(doc_id, chunks)
    return store, doc_id


def test_grounded_answer_returns_cited_chunk(indexed_doc_id):
    store, doc_id = indexed_doc_id
    response = json.dumps({"grounded": True, "answer": "Payment is made per milestone.", "cited_excerpts": [1]})
    client = FakeLLMClient(response)

    result = ask(doc_id, "What are the payment terms?", store, client)

    assert result.grounded is True
    assert result.answer == "Payment is made per milestone."
    assert len(result.cited_chunks) == 1
    assert result.cited_chunks[0] in result.retrieved_chunks


def test_explicit_refusal_when_model_says_ungrounded(indexed_doc_id):
    store, doc_id = indexed_doc_id
    response = json.dumps(
        {"grounded": False, "answer": "The document does not mention this.", "cited_excerpts": []}
    )
    client = FakeLLMClient(response)

    result = ask(doc_id, "What is the weather like in Mumbai?", store, client)

    assert result.grounded is False
    assert result.cited_chunks == []
    assert "does not mention" in result.answer


def test_no_indexed_content_short_circuits_without_calling_llm(tmp_path):
    store = VectorStore(persist_path=str(tmp_path / "chroma"))
    client = NeverCalledLLMClient()

    result = ask("nonexistent-doc-id", "What is the notice period?", store, client)

    assert result.grounded is False
    assert result.retrieved_chunks == []
    assert "no indexed content" in result.answer.lower()


def test_grounded_claim_with_no_citation_is_downgraded_to_refusal(indexed_doc_id):
    store, doc_id = indexed_doc_id
    # model claims grounded but doesn't actually cite anything -- shouldn't be trusted
    response = json.dumps({"grounded": True, "answer": "Some confident-sounding answer.", "cited_excerpts": []})
    client = FakeLLMClient(response)

    result = ask(doc_id, "What are the payment terms?", store, client)

    assert result.grounded is False
    assert result.cited_chunks == []
    assert result.answer != "Some confident-sounding answer."


def test_valid_citation_outweighs_a_false_grounded_flag(indexed_doc_id):
    """Regression test from live testing: llama3.2:3b returned
    "grounded": false alongside a correct answer AND a valid citation,
    which rendered a good answer as a refusal. A citation is checkable
    against the retrieved chunks; the model's claim about itself is not,
    so the citation wins.
    """
    store, doc_id = indexed_doc_id
    response = json.dumps(
        {
            "grounded": False,
            "answer": "According to excerpt [1], payment is made per milestone.",
            "cited_excerpts": [1],
        }
    )
    client = FakeLLMClient(response)

    result = ask(doc_id, "What are the payment terms?", store, client)

    assert result.grounded is True
    assert len(result.cited_chunks) == 1
    assert "milestone" in result.answer


def test_malformed_output_fails_safe(indexed_doc_id):
    store, doc_id = indexed_doc_id
    client = FakeLLMClient("not json at all, just rambling")

    result = ask(doc_id, "What are the payment terms?", store, client)

    assert result.grounded is False
    assert result.parse_error is True
    assert result.cited_chunks == []


def test_retrieval_and_generation_latency_are_measured_separately(indexed_doc_id):
    store, doc_id = indexed_doc_id
    response = json.dumps({"grounded": True, "answer": "Payment is made per milestone.", "cited_excerpts": [1]})
    client = FakeLLMClient(response)

    result = ask(doc_id, "What are the payment terms?", store, client)

    assert result.retrieval_latency_ms > 0
    assert result.generation_latency_ms >= 0


def test_no_indexed_content_still_reports_retrieval_latency(tmp_path):
    store = VectorStore(persist_path=str(tmp_path / "chroma"))
    client = NeverCalledLLMClient()

    result = ask("nonexistent-doc-id", "What is the notice period?", store, client)

    assert result.retrieval_latency_ms > 0
    assert result.generation_latency_ms == 0.0


def test_citations_requested_counts_before_filtering_to_valid(indexed_doc_id):
    """cited_excerpts named an out-of-range index (99) alongside a valid
    one -- citations_requested should count both, cited_chunks only the
    valid one, so a caller can compute a citation-validity ratio."""
    store, doc_id = indexed_doc_id
    response = json.dumps(
        {"grounded": True, "answer": "Payment is made per milestone.", "cited_excerpts": [1, 99]}
    )
    client = FakeLLMClient(response)

    result = ask(doc_id, "What are the payment terms?", store, client)

    assert result.citations_requested == 2
    assert len(result.cited_chunks) == 1
