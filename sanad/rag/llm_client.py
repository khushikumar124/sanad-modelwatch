"""LLM backend abstraction.

Both the summarizer and the chatbot call an LLMClient rather than talking
to Ollama directly, so the RAG/feature logic never depends on which
backend is running -- swapping Ollama for something else later means
writing one new LLMClient subclass, not touching summarizer.py or
chatbot.py.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import requests

from sanad.config import config

logger = logging.getLogger(__name__)


class LLMConnectionError(Exception):
    """Raised when the configured LLM backend isn't usable -- either it
    can't be reached at all, or it's reachable but can't serve the
    requested model (e.g. the model was never pulled). Both are
    operator-fixable setup problems, and both map to a 503 at the API
    layer, so they share one exception type."""


class LLMClient(ABC):
    @abstractmethod
    def generate(
        self, system_prompt: str, user_prompt: str, response_schema: dict | None = None
    ) -> str:
        """Run one system+user prompt through the model and return its
        text response.

        response_schema, when given, is a JSON Schema the backend should
        constrain output to. Backends that can't enforce it may ignore it
        -- callers must still parse defensively.
        """
        raise NotImplementedError


class OllamaClient(LLMClient):
    """Talks to a local Ollama server's /api/chat endpoint.

    llama3.2:3b is the configured default (see sanad/config.py) -- picked
    for a 16GB-RAM laptop with no discrete GPU: it leaves headroom for the
    embedding model, ChromaDB, and everything else running alongside it.
    A larger model (e.g. Mistral 7B) gives noticeably better extraction/QA
    quality at the cost of a tighter memory margin and slower responses.
    """

    def __init__(self, model: str | None = None, base_url: str | None = None, num_ctx: int = 4096):
        self.model = model or config.ollama_model
        self.base_url = base_url or config.ollama_base_url
        self.num_ctx = num_ctx

    def generate(
        self, system_prompt: str, user_prompt: str, response_schema: dict | None = None
    ) -> str:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            # low temperature: extraction/grounded QA should be
            # deterministic-leaning, not creative
            "options": {"num_ctx": self.num_ctx, "temperature": 0.1},
        }
        if response_schema is not None:
            # Constrained decoding. Small models routinely emit not-quite-JSON
            # when merely *asked* for it (observed: an unquoted string value
            # for "answer"), which threw away otherwise-correct responses.
            # Ollama enforces the schema during sampling instead.
            payload["format"] = response_schema

        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=180)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise LLMConnectionError(
                f"could not reach Ollama at {self.base_url} -- is `ollama serve` running?"
            ) from e
        except requests.exceptions.HTTPError as e:
            # Ollama is up but rejected the request. The common case by far is
            # the model never having been pulled, which returns 404 with an
            # {"error": ...} body -- surface that instead of a raw traceback.
            detail = ""
            try:
                detail = e.response.json().get("error", "")
            except ValueError:
                detail = (e.response.text or "").strip()[:200]
            hint = f" -- try `ollama pull {self.model}`" if e.response.status_code == 404 else ""
            raise LLMConnectionError(
                f"Ollama at {self.base_url} returned {e.response.status_code}: {detail}{hint}"
            ) from e

        return response.json()["message"]["content"]
