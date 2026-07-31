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
    """Raised when the configured LLM backend can't be reached."""


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Run one system+user prompt through the model and return its
        text response."""
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

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    # low temperature: extraction/grounded QA should be
                    # deterministic-leaning, not creative
                    "options": {"num_ctx": self.num_ctx, "temperature": 0.1},
                },
                timeout=180,
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise LLMConnectionError(
                f"could not reach Ollama at {self.base_url} -- is `ollama serve` running "
                f"and has `ollama pull {self.model}` been run?"
            ) from e

        return response.json()["message"]["content"]
