"""Records what happened on each answered question, for monitoring.

Deliberately holds no reference to ModelWatch. Sanad exposes a buffer of
recent events and knows nothing about who reads it; a separate reporter
(`modelwatch/examples/telemetry_reporter.py`) polls that endpoint and
forwards to the monitoring API. That keeps the dependency pointing one
way -- Sanad runs perfectly well with nothing watching it, and swapping
the monitoring system out touches nothing in this app.

Operational facts (whether the answer was grounded, how many clauses it
cited, how long it took, retrieval/generation latency split, retrieval
similarity scores) are always recorded and never include contract text.
doc_id is kept because it is already an opaque identifier Sanad assigns
on upload, not contract content.

`full_trace`, gated by `config.telemetry_full_trace` (default on -- see
that config field's own comment), is the one deliberate exception: when
enabled, the complete RAG trace (sanad/features/trace.py) -- question,
answer, retrieved/cited clause text, claim verification -- rides along
too, because ModelWatch's RAG X-Ray needs real content to reconstruct a
request's pipeline, not just rates. This is a real privacy tradeoff, not
a free one: turn `SANAD_TELEMETRY_FULL_TRACE=false` off for a deployment
where the monitor must never see contract content.

The five original fields (grounded, citations, latency_ms, parse_error,
retrieved) are kept exactly as they were: `LiveTelemetryAdapter` reads
them by name, and this stays a superset rather than a breaking change.
"""
from __future__ import annotations

import logging
import threading
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Bounded so a long-running server cannot grow this without limit. Old
# events fall off the back; the reporter is expected to poll far more
# often than it takes to fill.
_MAX_EVENTS = 500


@dataclass(frozen=True)
class ChatEvent:
    at: str
    grounded: bool
    citations: int
    latency_ms: float
    parse_error: bool
    retrieved: int

    # -- richer, additive fields (defaulted so old call sites still work) --
    trace_id: str = ""
    doc_id: str = ""
    model_name: str = ""
    top_k: int = 0
    #: cosine distances of retrieved chunks (lower = more similar), never
    #: chunk text -- lets a reader see retrieval quality degrade without
    #: exposing what was retrieved.
    retrieval_scores: list[float] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    #: how many excerpt numbers the model named before filtering to valid
    #: ones; citations / citations_requested is a citation-validity ratio.
    citations_requested: int = 0
    #: full RAGTrace.to_dict() when config.telemetry_full_trace is on,
    #: else None. See this module's docstring -- the one field here that
    #: can carry real contract text.
    full_trace: dict[str, Any] | None = None


_events: deque[ChatEvent] = deque(maxlen=_MAX_EVENTS)
_lock = threading.Lock()


def record_chat(
    *,
    grounded: bool,
    citations: int,
    latency_ms: float,
    parse_error: bool,
    retrieved: int,
    doc_id: str = "",
    model_name: str = "",
    top_k: int = 0,
    retrieval_scores: list[float] | None = None,
    retrieval_latency_ms: float = 0.0,
    generation_latency_ms: float = 0.0,
    citations_requested: int = 0,
    full_trace: dict[str, Any] | None = None,
) -> None:
    trace_id = uuid.uuid4().hex
    if full_trace is not None:
        full_trace = {**full_trace, "trace_id": trace_id}
    with _lock:
        _events.append(
            ChatEvent(
                at=datetime.now(timezone.utc).isoformat(),
                grounded=grounded,
                citations=citations,
                latency_ms=round(latency_ms, 1),
                parse_error=parse_error,
                retrieved=retrieved,
                trace_id=trace_id,
                doc_id=doc_id,
                model_name=model_name,
                top_k=top_k,
                retrieval_scores=[round(float(s), 4) for s in (retrieval_scores or [])],
                retrieval_latency_ms=round(retrieval_latency_ms, 1),
                generation_latency_ms=round(generation_latency_ms, 1),
                citations_requested=citations_requested,
                full_trace=full_trace,
            )
        )


def snapshot(drain: bool = False) -> list[dict[str, Any]]:
    """Return recent events. With drain=True they are consumed, so a
    polling reporter sees each question exactly once instead of
    re-reporting the same window every cycle."""
    with _lock:
        events = [asdict(e) for e in _events]
        if drain:
            _events.clear()
    return events


def count() -> int:
    with _lock:
        return len(_events)
