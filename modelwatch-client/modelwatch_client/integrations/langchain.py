"""A LangChain callback handler that reports retrieval/generation
telemetry to ModelWatch automatically -- no changes to an existing
LangChain retrieval chain beyond passing this handler in.

This is the highest-leverage integration this SDK can ship: most RAG
apps built in the wild are built on LangChain or LlamaIndex, and a
five-minute `callbacks=[ModelWatchCallbackHandler(...)]` beats asking
someone to restructure their app around this project's own telemetry
shape.

Requires `langchain-core` (the lightweight core package, not the full
`langchain` metapackage) -- an optional dependency, imported lazily so
installing plain `modelwatch-client` never pulls it in. Install with
`pip install "modelwatch-client[langchain]"`.

Correlation model: a retrieval chain typically fires `on_retriever_*`
and `on_llm_*` as siblings under one parent chain run. This handler
keys pending events by `parent_run_id` (falling back to `run_id` for a
bare, chain-less retriever+LLM pair) and reports one event to
ModelWatch when the LLM call for that run finishes -- it does not
attempt to reconstruct LangChain's full run tree, only the common
retrieve-then-generate shape.

Reports operational telemetry only (latencies, retrieved-document count,
any similarity scores LangChain's retriever already attached to
document metadata) -- never document or generation content, unless the
caller explicitly opts in via `include_content=True`. That default
mirrors the same reasoning Sanad's own telemetry uses (see
sanad/api/telemetry.py in the main repo): a monitoring integration
should not silently start exporting an app's private content.
"""
from __future__ import annotations

import logging
import time
from typing import Any
from uuid import UUID

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError as e:  # pragma: no cover -- exercised by the "not installed" test
    raise ImportError(
        "ModelWatchCallbackHandler requires langchain-core. Install it with "
        "`pip install \"modelwatch-client[langchain]\"` or `pip install langchain-core`."
    ) from e

from modelwatch_client.client import ModelWatchClient, ModelWatchError

logger = logging.getLogger(__name__)


class ModelWatchCallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
        client: ModelWatchClient,
        model_id: str,
        include_content: bool = False,
        on_error: str = "log",  # "log" | "raise" -- a reporting failure should rarely raise, but a caller can ask
    ):
        self.client = client
        self.model_id = model_id
        self.include_content = include_content
        self.on_error = on_error
        self._runs: dict[str, dict[str, Any]] = {}

    def _key(self, run_id: UUID, parent_run_id: UUID | None) -> str:
        return str(parent_run_id or run_id)

    def on_retriever_start(self, serialized: dict, query: str, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        run = self._runs.setdefault(self._key(run_id, parent_run_id), {})
        run["retrieval_started"] = time.time()
        if self.include_content:
            run["query"] = query

    def on_retriever_end(self, documents: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        run = self._runs.setdefault(self._key(run_id, parent_run_id), {})
        started = run.pop("retrieval_started", None)
        run["retrieval_latency_ms"] = (time.time() - started) * 1000 if started else None
        run["retrieved"] = len(documents)
        scores = [d.metadata.get("score") for d in documents if getattr(d, "metadata", None) and d.metadata.get("score") is not None]
        if scores:
            run["retrieval_scores"] = scores

    def on_llm_start(self, serialized: dict, prompts: list[str], *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        run = self._runs.setdefault(self._key(run_id, parent_run_id), {})
        run["generation_started"] = time.time()

    def on_llm_end(self, response: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        key = self._key(run_id, parent_run_id)
        run = self._runs.pop(key, {})
        started = run.pop("generation_started", None)
        run["generation_latency_ms"] = (time.time() - started) * 1000 if started else None
        if self.include_content:
            try:
                run["answer"] = response.generations[0][0].text
            except (AttributeError, IndexError):
                pass
        self._report(run)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        self._runs.pop(self._key(run_id, parent_run_id), None)

    def _report(self, event: dict[str, Any]) -> None:
        try:
            self.client.check(self.model_id, [event])
        except ModelWatchError as e:
            if self.on_error == "raise":
                raise
            logger.warning("failed to report event to ModelWatch", extra={"model_id": self.model_id, "error": str(e)})
