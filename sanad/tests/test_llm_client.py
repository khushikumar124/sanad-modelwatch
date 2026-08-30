"""OllamaClient error-handling tests.

These deliberately point at a port with no listener rather than relying on
the real Ollama being absent -- an earlier version used Ollama's own port
and silently passed only on machines where Ollama wasn't installed, then
broke the moment it was. Test outcomes shouldn't depend on whether an
optional service happens to be running.
"""
import time

import pytest
from werkzeug.wrappers import Response

from sanad.rag.llm_client import LLMConnectionError, OllamaClient

# Port 9 (discard) is reserved and has no listener on a normal machine.
UNREACHABLE_URL = "http://127.0.0.1:9"


def test_raises_clear_error_when_server_unreachable():
    client = OllamaClient(base_url=UNREACHABLE_URL)

    with pytest.raises(LLMConnectionError, match="ollama serve"):
        client.generate("system", "user")


def test_raises_clear_error_when_model_not_pulled(httpserver):
    """Ollama running but the model was never pulled -> 404 with an error
    body. Must surface an actionable `ollama pull` hint, not a raw
    requests.HTTPError (which would become an opaque HTTP 500)."""
    httpserver.expect_request("/api/chat").respond_with_json(
        {"error": 'model "llama3.2:3b" not found'}, status=404
    )
    client = OllamaClient(base_url=httpserver.url_for(""), model="llama3.2:3b")

    with pytest.raises(LLMConnectionError, match="ollama pull"):
        client.generate("system", "user")


def test_raises_clear_error_on_server_error(httpserver):
    httpserver.expect_request("/api/chat").respond_with_data("boom", status=500)
    client = OllamaClient(base_url=httpserver.url_for(""))

    with pytest.raises(LLMConnectionError, match="500"):
        client.generate("system", "user")


def test_returns_content_on_success(httpserver):
    httpserver.expect_request("/api/chat").respond_with_json(
        {"message": {"role": "assistant", "content": "hello from the model"}}
    )
    client = OllamaClient(base_url=httpserver.url_for(""))

    assert client.generate("system", "user") == "hello from the model"


def test_raises_clear_error_on_timeout_instead_of_an_unhandled_exception(httpserver):
    """Regression test: a slow response (e.g. array-of-objects schema
    extraction on a full document, see obligations.py) previously
    propagated as a raw requests.exceptions.ReadTimeout all the way to
    the API layer, surfacing as an opaque 500 instead of the clean 503
    every other LLM-unavailable path produces."""
    def slow_handler(_request):
        time.sleep(0.3)
        return Response(
            '{"message": {"role": "assistant", "content": "too slow"}}',
            content_type="application/json",
        )

    httpserver.expect_request("/api/chat").respond_with_handler(slow_handler)
    client = OllamaClient(base_url=httpserver.url_for(""))

    with pytest.raises(LLMConnectionError, match="did not respond within"):
        client.generate("system", "user", timeout=0.05)
