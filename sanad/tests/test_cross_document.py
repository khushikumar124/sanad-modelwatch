"""Cross-document Q&A tests using a real VectorStore, indexed from two
different real sample contracts, but a FakeLLMClient -- retrieval is
real (each document's own top matches, merged and correctly labeled),
the model's answer is controlled for deterministic assertions."""
import json
import uuid

import pytest

from sanad.features.cross_document import NO_CONTEXT_ANSWER, ask_across_documents
from sanad.ingestion.chunking import chunk_document
from sanad.ingestion.extraction import extract_document
from sanad.rag.llm_client import LLMClient
from sanad.rag.vector_store import VectorStore

RENTAL_DOC = "sanad/sample_docs/rental/rental_agreement_sample_1.pdf"
EMPLOYMENT_DOC = "sanad/sample_docs/employment/employment_letter_samples.pdf"


class FakeLLMClient(LLMClient):
    def __init__(self, response: str):
        self.response = response
        self.last_user_prompt = None

    def generate(self, system_prompt: str, user_prompt: str, response_schema: dict | None = None, timeout: float = 180) -> str:
        self.last_user_prompt = user_prompt
        return self.response


@pytest.fixture(scope="module")
def two_indexed_docs(tmp_path_factory):
    store = VectorStore(persist_path=str(tmp_path_factory.mktemp("chroma")))
    rental_id = f"test-rental-{uuid.uuid4()}"
    employment_id = f"test-employment-{uuid.uuid4()}"
    store.add_document(rental_id, chunk_document(extract_document(RENTAL_DOC).text))
    store.add_document(employment_id, chunk_document(extract_document(EMPLOYMENT_DOC).text))
    return store, rental_id, employment_id


def test_grounded_answer_cites_chunks_from_both_documents(two_indexed_docs):
    store, rental_id, employment_id = two_indexed_docs
    documents = [(rental_id, "rental.pdf"), (employment_id, "employment.pdf")]
    # top_k_per_doc=4 -> excerpts 1-4 are rental's, 5-8 are employment's
    response = json.dumps(
        {"grounded": True, "answer": "The two documents have different notice periods.", "cited_excerpts": [1, 5]}
    )
    client = FakeLLMClient(response)

    result = ask_across_documents(documents, "Compare the notice periods.", store, client)

    assert result.grounded is True
    cited_labels = {c["doc_label"] for c in result.cited_chunks}
    assert cited_labels == {"rental.pdf", "employment.pdf"}


def test_retrieved_chunks_are_tagged_with_their_source_document(two_indexed_docs):
    store, rental_id, employment_id = two_indexed_docs
    documents = [(rental_id, "rental.pdf"), (employment_id, "employment.pdf")]
    client = FakeLLMClient(json.dumps({"grounded": False, "answer": "n/a", "cited_excerpts": []}))

    result = ask_across_documents(documents, "What are the terms?", store, client)

    doc_ids_seen = {c["doc_id"] for c in result.retrieved_chunks}
    assert doc_ids_seen == {rental_id, employment_id}


def test_prompt_labels_each_excerpt_with_its_document(two_indexed_docs):
    store, rental_id, employment_id = two_indexed_docs
    documents = [(rental_id, "rental.pdf"), (employment_id, "employment.pdf")]
    client = FakeLLMClient(json.dumps({"grounded": False, "answer": "n/a", "cited_excerpts": []}))

    ask_across_documents(documents, "What are the terms?", store, client)

    assert '(from "rental.pdf")' in client.last_user_prompt
    assert '(from "employment.pdf")' in client.last_user_prompt


def test_explicit_refusal_when_model_says_ungrounded(two_indexed_docs):
    store, rental_id, employment_id = two_indexed_docs
    documents = [(rental_id, "rental.pdf"), (employment_id, "employment.pdf")]
    response = json.dumps({"grounded": False, "answer": "Neither document addresses this.", "cited_excerpts": []})
    client = FakeLLMClient(response)

    result = ask_across_documents(documents, "What is the penalty for late rent on Mars?", store, client)

    assert result.grounded is False
    assert result.cited_chunks == []


def test_model_claiming_grounded_with_no_citation_is_downgraded_to_refusal(two_indexed_docs):
    store, rental_id, employment_id = two_indexed_docs
    documents = [(rental_id, "rental.pdf"), (employment_id, "employment.pdf")]
    response = json.dumps({"grounded": True, "answer": "Something confident-sounding.", "cited_excerpts": []})
    client = FakeLLMClient(response)

    result = ask_across_documents(documents, "Anything?", store, client)

    assert result.grounded is False


def test_no_documents_produces_no_context_answer_without_calling_the_llm():
    class NeverCalledLLMClient(LLMClient):
        def generate(self, system_prompt, user_prompt, response_schema=None, timeout=180):
            raise AssertionError("LLM should not be called with no documents")

    result = ask_across_documents([], "Anything?", VectorStore(persist_path="/tmp/unused"), NeverCalledLLMClient())
    assert result.answer == NO_CONTEXT_ANSWER
    assert result.grounded is False


def test_malformed_json_response_is_reported_as_a_parse_error(two_indexed_docs):
    store, rental_id, employment_id = two_indexed_docs
    documents = [(rental_id, "rental.pdf"), (employment_id, "employment.pdf")]
    client = FakeLLMClient("not json at all")

    result = ask_across_documents(documents, "What are the terms?", store, client)

    assert result.parse_error is True
    assert result.grounded is False
