"""MonitoringEngine: orchestrates adapters + storage.

This is the model-agnostic core. It never imports scipy, sklearn, or any
adapter-specific library -- it only calls ModelAdapter.build_baseline() /
check_drift() and persists whatever JSON-serializable data comes back.
Adding a new model type never requires touching this file.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from modelwatch.core.adapter_base import ModelAdapter
from modelwatch.core.storage import ModelNotFoundError, Storage

logger = logging.getLogger(__name__)


class ModelNotRegisteredError(Exception):
    """Raised when an operation needs a live adapter instance that hasn't
    been registered in this engine process (e.g. after a restart without
    re-registering)."""


class MonitoringEngine:
    def __init__(self, storage: Storage):
        self._storage = storage
        self._adapters: dict[str, ModelAdapter] = {}

    # -- registration -----------------------------------------------------

    def register_model(
        self,
        model_id: str,
        name: str,
        adapter: ModelAdapter,
        baseline_data: Any,
    ) -> dict[str, Any]:
        """Register a new monitored model and build its initial baseline."""
        model = self._storage.create_model(model_id, name, adapter.adapter_name)
        self._adapters[model_id] = adapter

        baseline = adapter.build_baseline(baseline_data)
        self._storage.save_baseline(model_id, version=1, data=baseline)
        self._storage.add_version(model_id, version=1, reason="initial registration")
        logger.info("baseline built", extra={"model_id": model_id, "version": 1})
        return model

    def attach_adapter(self, model_id: str, adapter: ModelAdapter) -> None:
        """Re-attach a live adapter instance for a model already in storage
        (e.g. after a process restart) without rebuilding its baseline."""
        if self._storage.get_model(model_id) is None:
            raise ModelNotFoundError(f"model '{model_id}' is not registered")
        self._adapters[model_id] = adapter

    def _get_adapter(self, model_id: str) -> ModelAdapter:
        adapter = self._adapters.get(model_id)
        if adapter is None:
            if self._storage.get_model(model_id) is None:
                raise ModelNotFoundError(f"model '{model_id}' is not registered")
            raise ModelNotRegisteredError(
                f"model '{model_id}' exists in storage but has no live adapter attached "
                "in this process -- call attach_adapter() first"
            )
        return adapter

    # -- checks -------------------------------------------------------------

    def run_check(self, model_id: str, new_data: Any) -> dict[str, Any]:
        """Run a drift/quality check for a model against its current baseline.

        Persists the run unconditionally, and creates exactly one alert if
        the adapter reports drift.
        """
        adapter = self._get_adapter(model_id)
        baseline = self._storage.get_latest_baseline(model_id)
        if baseline is None:
            raise ModelNotFoundError(f"model '{model_id}' has no baseline")

        result = adapter.check_drift(baseline["data"], new_data)
        model = self._storage.get_model(model_id)

        run_id = self._storage.save_run(
            model_id=model_id,
            version=model["current_version"],
            drift_score=result.drift_score,
            quality_score=result.quality_score,
            is_drifted=result.is_drifted,
            signals=[s.to_dict() for s in result.signals],
            statistics=result.statistics,
        )

        alert_id = None
        if result.is_drifted:
            message = (
                f"Drift detected for model '{model_id}' (v{model['current_version']}): "
                f"drift_score={result.drift_score:.3f}"
                + (
                    f", quality_score={result.quality_score:.3f}"
                    if result.quality_score is not None
                    else ""
                )
            )
            alert_id = self._storage.create_alert(model_id, run_id, message)
            logger.warning("drift alert raised", extra={"model_id": model_id, "run_id": run_id})

        return {"run_id": run_id, "alert_id": alert_id, **result.to_dict()}

    # -- reads ----------------------------------------------------------------

    def list_models(self) -> list[dict[str, Any]]:
        return self._storage.list_models()

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        return self._storage.get_model(model_id)

    def get_history(self, model_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        return self._storage.get_history(model_id, limit=limit)

    def get_alerts(self, model_id: str | None = None, active_only: bool = False) -> list[dict[str, Any]]:
        return self._storage.get_alerts(model_id=model_id, active_only=active_only)

    def get_versions(self, model_id: str) -> list[dict[str, Any]]:
        return self._storage.get_versions(model_id)

    # -- self-healing ---------------------------------------------------------

    def trigger_retrain(
        self,
        model_id: str,
        retrain_fn: Callable[[Any], Any],
        new_training_data: Any,
    ) -> dict[str, Any]:
        """Retrain, reset the baseline to fresh data, and bump the model's
        version. This is what makes ModelWatch self-healing rather than a
        pure alerting system: after this call, subsequent checks compare
        against the new baseline and previously open alerts are resolved.

        retrain_fn is caller-supplied and does the actual retraining (e.g.
        refitting a classifier, or nothing for a prompt-only LLM app) -- the
        engine only knows it's a callable that consumes new_training_data.
        """
        adapter = self._get_adapter(model_id)
        model = self._storage.get_model(model_id)
        if model is None:
            raise ModelNotFoundError(f"model '{model_id}' is not registered")

        retrain_fn(new_training_data)

        new_baseline = adapter.build_baseline(new_training_data)
        new_version = model["current_version"] + 1
        self._storage.save_baseline(model_id, version=new_version, data=new_baseline)
        self._storage.set_current_version(model_id, new_version)
        self._storage.add_version(model_id, version=new_version, reason="retrain")
        resolved_count = self._storage.resolve_alerts_for_model(model_id)

        logger.info(
            "model retrained",
            extra={"model_id": model_id, "new_version": new_version, "alerts_resolved": resolved_count},
        )
        return self._storage.get_model(model_id)
