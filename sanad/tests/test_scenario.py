"""Scenario Simulator tests using a real VectorStore (indexed from a
real sample contract) but a FakeLLMClient, so retrieval is real while
the model's answer is controlled for deterministic assertions."""
import json
import uuid

import pytest

from sanad.features.scenario import NO_CONTEXT_ANSWER, UNGROUNDED_CITATION_ANSWER, simulate_scenario
from sanad.ingestion.chunking import chunk_document
from sanad.ingestion.extraction import extract_document
from sanad.rag.llm_client import LLMClient
from sanad.rag.vector_store import VectorStore

RENTAL_DOC = "sanad/sample_docs/rental/rental_agreement_sample_1.pdf"


class FakeLLMClient(LLMClient):
    def __init__(self, response: str):
        self.response = response
        self.last_user_prompt = None

    def generate(self, system_prompt: str, user_prompt: str, response_schema: dict | None = None, timeout: float = 180) -> str:
        self.last_user_prompt = user_prompt
        return self.response


@pytest.fixture(scope="module")
def indexed_rental_doc(tmp_path_factory):
    doc = extract_document(RENTAL_DOC)
    chunks = chunk_document(doc.text)
    store = VectorStore(persist_path=str(tmp_path_factory.mktemp("chroma")))
    doc_id = f"test-{uuid.uuid4()}"
    store.add_document(doc_id, chunks)
    return store, doc_id


def test_grounded_scenario_returns_cited_chunk(indexed_rental_doc):
    store, doc_id = indexed_rental_doc
    response = json.dumps({
        "grounded": True,
        "answer": "You may terminate with one month's written notice.",
        "cited_excerpts": [1],
    })
    client = FakeLLMClient(response)

    result = simulate_scenario(doc_id, "What happens if I want to leave before the lease ends?", store, client)

    assert result.grounded is True
    assert len(result.cited_chunks) == 1
    assert result.cited_chunks[0] in result.retrieved_chunks


def test_prompt_frames_the_input_as_a_scenario(indexed_rental_doc):
    store, doc_id = indexed_rental_doc
    client = FakeLLMClient(json.dumps({"grounded": False, "answer": "n/a", "cited_excerpts": []}))
    simulate_scenario(doc_id, "What if I want to sublet?", store, client)
    assert "Scenario:" in client.last_user_prompt
    assert "What if I want to sublet?" in client.last_user_prompt


def test_explicit_refusal_when_model_says_ungrounded(indexed_rental_doc):
    store, doc_id = indexed_rental_doc
    response = json.dumps({
        "grounded": False,
        "answer": "The document does not address this scenario.",
        "cited_excerpts": [],
    })
    client = FakeLLMClient(response)

    result = simulate_scenario(doc_id, "What if the building burns down?", store, client)

    assert result.grounded is False
    assert result.cited_chunks == []


def test_model_claiming_grounded_with_no_citation_is_downgraded_to_refusal(indexed_rental_doc):
    store, doc_id = indexed_rental_doc
    response = json.dumps({"grounded": True, "answer": "Something confident-sounding.", "cited_excerpts": []})
    client = FakeLLMClient(response)

    result = simulate_scenario(doc_id, "What if I stop paying rent?", store, client)

    assert result.grounded is False
    assert result.answer == UNGROUNDED_CITATION_ANSWER


def test_citation_numbers_as_strings_are_still_accepted(indexed_rental_doc):
    """A real observed model quirk without schema enforcement: citation
    numbers coming back as numeric strings ("1") instead of integers.
    SCENARIO_SCHEMA constrains a schema-aware backend to real integers,
    but parsing accepts strings too as a defensive fallback -- same
    spirit as rag/llm_client.py's other defensive-parsing precedent."""
    store, doc_id = indexed_rental_doc
    response = json.dumps({"grounded": True, "answer": "Real answer.", "cited_excerpts": ["1", "2"]})
    client = FakeLLMClient(response)

    result = simulate_scenario(doc_id, "What if I damage the property?", store, client)

    assert result.grounded is True
    assert len(result.cited_chunks) == 2


def test_no_retrieved_chunks_produces_no_context_answer_without_calling_the_llm():
    class NeverCalledLLMClient(LLMClient):
        def generate(self, system_prompt, user_prompt, response_schema=None, timeout=180):
            raise AssertionError("LLM should not be called with no retrieved context")

    empty_store = VectorStore(persist_path="/tmp/unused_scenario_store")
    result = simulate_scenario("never-uploaded", "What if?", empty_store, NeverCalledLLMClient())
    assert result.answer == NO_CONTEXT_ANSWER
    assert result.grounded is False


def test_malformed_json_response_is_reported_as_a_parse_error(indexed_rental_doc):
    store, doc_id = indexed_rental_doc
    client = FakeLLMClient("not json at all")

    result = simulate_scenario(doc_id, "What if I want to renew?", store, client)

    assert result.parse_error is True
    assert result.grounded is False
