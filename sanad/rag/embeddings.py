"""Local embedding model wrapper.

Wraps sentence-transformers so the rest of the RAG pipeline depends on this
thin interface rather than the library directly. all-MiniLM-L6-v2 is the
configured default (see sanad/config.py) -- ~80MB, CPU-friendly, and fast
enough for interactive use on a laptop with no GPU, at some cost to
retrieval quality versus a larger model.
"""
from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

from sanad.config import config

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.embedding_model
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("loading embedding model", extra={"model": self.model_name})
            # Pinned to CPU deliberately. A newer sentence-transformers
            # release started auto-selecting "mps" (Apple Silicon's GPU
            # backend) whenever it's available, which this module's own
            # docstring never intended -- and it's not just a slower
            # choice: MPS initialized inside a forked worker process
            # (which sentence-transformers/chromadb spin up internally
            # for batch encoding) crashes the whole server with no
            # Python traceback at all, a real bug this caused during a
            # live demo run. all-MiniLM-L6-v2 is small enough that CPU
            # was always the intended, and only verified, target.
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
