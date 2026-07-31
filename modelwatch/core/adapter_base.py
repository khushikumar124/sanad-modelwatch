"""The ModelAdapter contract.

This is the one seam the monitoring engine is allowed to depend on. The
engine (core/engine.py) never branches on model type, never imports scipy,
sklearn, or anything else adapter-specific -- it only calls build_baseline()
and check_drift() through this interface. New model types (a new tabular
model, a vision model, a different LLM app) are added by writing a new
ModelAdapter subclass, not by editing the engine.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SignalResult:
    """One measured signal within a drift check (e.g. one feature's KS test)."""

    name: str
    value: float
    is_drifted: bool
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "is_drifted": self.is_drifted,
            "detail": self.detail,
        }


@dataclass
class DriftCheckResult:
    """The outcome of running check_drift() against one batch of new data.

    drift_score is normalized to [0, 1] where higher means more drift,
    regardless of which adapter produced it -- this is what lets the engine
    store and chart results from different model types on one timeline.
    quality_score is normalized to [0, 1] where higher is better, or None
    when quality can't be computed (e.g. no labels supplied).
    """

    drift_score: float
    quality_score: float | None
    is_drifted: bool
    signals: list[SignalResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "drift_score": self.drift_score,
            "quality_score": self.quality_score,
            "is_drifted": self.is_drifted,
            "signals": [s.to_dict() for s in self.signals],
        }


class ModelAdapter(ABC):
    """Abstract interface every monitored model type must implement.

    Baselines are plain JSON-serializable dicts so the engine can persist
    them (SQLite, as a JSON blob) without knowing their internal shape.
    """

    #: Short identifier used for display/logging and stored in the DB.
    #: Concrete subclasses must override this.
    adapter_name: str = "base"

    @abstractmethod
    def build_baseline(self, data: Any) -> dict[str, Any]:
        """Build a JSON-serializable baseline snapshot from reference data."""
        raise NotImplementedError

    @abstractmethod
    def check_drift(self, baseline: dict[str, Any], new_data: Any) -> DriftCheckResult:
        """Compare new_data against a previously built baseline."""
        raise NotImplementedError
