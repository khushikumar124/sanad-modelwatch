"""Tests for modelwatch_client.integrations.langchain.

Calls the callback handler's methods directly with real langchain-core
objects (Document, LLMResult), in the sequence LangChain itself invokes
them for a retrieve-then-generate chain -- this is the standard way to
unit-test a callback handler without spinning up a real chain/LLM.
"""
from uuid import uuid4

import pytest

pytest.importorskip("langchain_core")

from langchain_core.documents import Document
from langchain_core.outputs import Generation, LLMResult

from modelwatch_client.client import ModelWatchError
from modelwatch_client.integrations.langchain import ModelWatchCallbackHandler


class FakeClient:
    def __init__(self):
        self.checks: list[tuple[str, list[dict]]] = []

    def check(self, model_id, new_data):
        self.checks.append((model_id, new_data))
        return {"is_drifted": False}


class FailingClient:
    def check(self, model_id, new_data):
        raise ModelWatchError("server unreachable")


def _run_retrieve_then_generate(handler, chain_run_id, documents, answer_text="an answer"):
    handler.on_retriever_start({}, "what is the notice period?", run_id=uuid4(), parent_run_id=chain_run_id)
    handler.on_retriever_end(documents, run_id=uuid4(), parent_run_id=chain_run_id)
    handler.on_llm_start({}, ["prompt"], run_id=uuid4(), parent_run_id=chain_run_id)
    handler.on_llm_end(LLMResult(generations=[[Generation(text=answer_text)]]), run_id=uuid4(), parent_run_id=chain_run_id)


def test_reports_one_event_per_completed_chain_run():
    client = FakeClient()
    handler = ModelWatchCallbackHandler(client, "my-rag-app")

    _run_retrieve_then_generate(handler, uuid4(), [Document(page_content="clause 1")])

    assert len(client.checks) == 1
    model_id, events = client.checks[0]
    assert model_id == "my-rag-app"
    assert len(events) == 1


def test_reported_event_includes_retrieval_and_generation_latency():
    client = FakeClient()
    handler = ModelWatchCallbackHandler(client, "my-rag-app")

    _run_retrieve_then_generate(handler, uuid4(), [Document(page_content="clause 1")])

    event = client.checks[0][1][0]
    assert event["retrieval_latency_ms"] is not None
    assert event["generation_latency_ms"] is not None
    assert event["retrieved"] == 1


def test_retrieval_scores_are_pulled_from_document_metadata_when_present():
    client = FakeClient()
    handler = ModelWatchCallbackHandler(client, "my-rag-app")
    docs = [Document(page_content="a", metadata={"score": 0.8}), Document(page_content="b", metadata={"score": 0.6})]

    _run_retrieve_then_generate(handler, uuid4(), docs)

    event = client.checks[0][1][0]
    assert event["retrieval_scores"] == [0.8, 0.6]


def test_no_retrieval_scores_key_when_documents_have_no_score_metadata():
    client = FakeClient()
    handler = ModelWatchCallbackHandler(client, "my-rag-app")

    _run_retrieve_then_generate(handler, uuid4(), [Document(page_content="a")])

    event = client.checks[0][1][0]
    assert "retrieval_scores" not in event


def test_content_excluded_by_default():
    client = FakeClient()
    handler = ModelWatchCallbackHandler(client, "my-rag-app")

    _run_retrieve_then_generate(handler, uuid4(), [Document(page_content="a")], answer_text="the real answer")

    event = client.checks[0][1][0]
    assert "answer" not in event
    assert "query" not in event


def test_content_included_when_opted_in():
    client = FakeClient()
    handler = ModelWatchCallbackHandler(client, "my-rag-app", include_content=True)

    _run_retrieve_then_generate(handler, uuid4(), [Document(page_content="a")], answer_text="the real answer")

    event = client.checks[0][1][0]
    assert event["answer"] == "the real answer"
    assert event["query"] == "what is the notice period?"


def test_concurrent_chain_runs_do_not_cross_contaminate_events():
    client = FakeClient()
    handler = ModelWatchCallbackHandler(client, "my-rag-app", include_content=True)
    chain_a, chain_b = uuid4(), uuid4()

    handler.on_retriever_start({}, "query A", run_id=uuid4(), parent_run_id=chain_a)
    handler.on_retriever_start({}, "query B", run_id=uuid4(), parent_run_id=chain_b)
    handler.on_retriever_end([Document(page_content="a")], run_id=uuid4(), parent_run_id=chain_a)
    handler.on_retriever_end([Document(page_content="b")], run_id=uuid4(), parent_run_id=chain_b)
    handler.on_llm_start({}, ["p"], run_id=uuid4(), parent_run_id=chain_a)
    handler.on_llm_start({}, ["p"], run_id=uuid4(), parent_run_id=chain_b)
    handler.on_llm_end(LLMResult(generations=[[Generation(text="answer A")]]), run_id=uuid4(), parent_run_id=chain_a)
    handler.on_llm_end(LLMResult(generations=[[Generation(text="answer B")]]), run_id=uuid4(), parent_run_id=chain_b)

    answers = {c[1][0]["query"]: c[1][0]["answer"] for c in client.checks}
    assert answers == {"query A": "answer A", "query B": "answer B"}


def test_llm_error_clears_pending_run_without_reporting():
    client = FakeClient()
    handler = ModelWatchCallbackHandler(client, "my-rag-app")
    chain_id = uuid4()

    handler.on_retriever_start({}, "q", run_id=uuid4(), parent_run_id=chain_id)
    handler.on_retriever_end([Document(page_content="a")], run_id=uuid4(), parent_run_id=chain_id)
    handler.on_llm_start({}, ["p"], run_id=uuid4(), parent_run_id=chain_id)
    handler.on_llm_error(RuntimeError("model crashed"), run_id=uuid4(), parent_run_id=chain_id)

    assert client.checks == []
    assert str(chain_id) not in handler._runs


def test_reporting_failure_is_logged_not_raised_by_default():
    handler = ModelWatchCallbackHandler(FailingClient(), "my-rag-app")
    _run_retrieve_then_generate(handler, uuid4(), [Document(page_content="a")])  # must not raise


def test_reporting_failure_raises_when_on_error_is_raise():
    handler = ModelWatchCallbackHandler(FailingClient(), "my-rag-app", on_error="raise")
    with pytest.raises(ModelWatchError):
        _run_retrieve_then_generate(handler, uuid4(), [Document(page_content="a")])


def test_bare_run_without_parent_id_still_correlates_on_run_id():
    """A chain-less retriever+LLM pair (parent_run_id=None both times,
    same run_id reused as its own key) should still correlate."""
    client = FakeClient()
    handler = ModelWatchCallbackHandler(client, "my-rag-app")
    shared_id = uuid4()

    handler.on_retriever_start({}, "q", run_id=shared_id, parent_run_id=None)
    handler.on_retriever_end([Document(page_content="a")], run_id=shared_id, parent_run_id=None)
    handler.on_llm_start({}, ["p"], run_id=shared_id, parent_run_id=None)
    handler.on_llm_end(LLMResult(generations=[[Generation(text="ans")]]), run_id=shared_id, parent_run_id=None)

    assert len(client.checks) == 1
    assert client.checks[0][1][0]["retrieved"] == 1
