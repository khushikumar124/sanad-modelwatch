"""OllamaClient error-handling test. Ollama isn't running in this test
environment, so this exercises the real connection-refused path rather
than mocking it -- confirming a clear, actionable error surfaces instead
of a raw requests traceback."""
import pytest

from sanad.rag.llm_client import LLMConnectionError, OllamaClient


def test_ollama_client_raises_clear_error_when_unreachable():
    client = OllamaClient(base_url="http://localhost:11434")

    with pytest.raises(LLMConnectionError, match="ollama pull"):
        client.generate("system", "user")
