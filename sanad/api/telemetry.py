"""Records what happened on each answered question, for monitoring.

Deliberately holds no reference to ModelWatch. Sanad exposes a buffer of
recent events and knows nothing about who reads it; a separate reporter
(`modelwatch/examples/telemetry_reporter.py`) polls that endpoint and
forwards to the monitoring API. That keeps the dependency pointing one
way -- Sanad runs perfectly well with nothing watching it, and swapping
the monitoring system out touches nothing in this app.

Only operational facts are stored: whether the answer was grounded, how
many clauses it cited, how long it took. No question text, no answer
text, no document identifiers. Contracts are confidential, and a
monitoring buffer is not a place to put their contents.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import asdict, dataclass
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


_events: deque[ChatEvent] = deque(maxlen=_MAX_EVENTS)
_lock = threading.Lock()


def record_chat(
    *, grounded: bool, citations: int, latency_ms: float, parse_error: bool, retrieved: int
) -> None:
    with _lock:
        _events.append(
            ChatEvent(
                at=datetime.now(timezone.utc).isoformat(),
                grounded=grounded,
                citations=citations,
                latency_ms=round(latency_ms, 1),
                parse_error=parse_error,
                retrieved=retrieved,
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
