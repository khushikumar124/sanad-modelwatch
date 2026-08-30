"""MonitoringEngine: orchestrates adapters + storage.

This is the model-agnostic core. It never imports scipy, sklearn, or any
adapter-specific library -- it only calls ModelAdapter.build_baseline() /
check_drift() and persists whatever JSON-serializable data comes back.
Adding a new model type never requires touching this file.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from modelwatch.config import config
from modelwatch.core.adapter_base import ModelAdapter
from modelwatch.core.health import next_health_state
from modelwatch.core.storage import ModelNotFoundError, Storage
from modelwatch.diagnosis.engine import DiagnosisResult, diagnose


class RunNotFoundError(Exception):
    pass

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
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register a new monitored model and build its initial baseline.

        config is an optional registry record (embedding model, chunk
        size, top-k, prompt version, dataset version, ...) describing
        what this model *is* -- purely informational to the engine, which
        never reads it back, but persisted so a later reader can answer
        "what changed between this registration and the last one" without
        needing to remember it out-of-band.
        """
        model = self._storage.create_model(model_id, name, adapter.adapter_name, config=config)
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

        health = self._storage.get_health(model_id)
        transition = next_health_state(
            current_state=health["state"],
            is_drifted=result.is_drifted,
            consecutive_drifted=health["consecutive_drifted"],
            consecutive_clean=health["consecutive_clean"],
            warning_after_consecutive=config.health_warning_after_consecutive,
            degraded_after_consecutive=config.health_degraded_after_consecutive,
            recovery_after_consecutive=config.health_recovery_after_consecutive,
        )
        self._storage.set_health(
            model_id, transition.state, transition.consecutive_drifted, transition.consecutive_clean
        )

        alert_id = None
        if transition.should_create_alert:
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
        elif transition.should_resolve_alerts:
            resolved = self._storage.resolve_alerts_for_model(model_id)
            if resolved:
                logger.info(
                    "model health recovered, alerts resolved",
                    extra={"model_id": model_id, "alerts_resolved": resolved},
                )

        return {
            "run_id": run_id,
            "alert_id": alert_id,
            "health_state": transition.state,
            **result.to_dict(),
        }

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

    def get_health(self, model_id: str) -> dict[str, Any]:
        return self._storage.get_health(model_id)

    # -- experiments ------------------------------------------------------

    def record_experiment(
        self, name: str, kind: str, config: dict[str, Any], results: dict[str, Any], status: str = "completed"
    ) -> dict[str, Any]:
        experiment_id = self._storage.create_experiment(name, kind, config, results, status)
        return self._storage.get_experiment(experiment_id)

    def list_experiments(self, kind: str | None = None) -> list[dict[str, Any]]:
        return self._storage.list_experiments(kind=kind)

    def get_experiment(self, experiment_id: int) -> dict[str, Any] | None:
        return self._storage.get_experiment(experiment_id)

    # -- traces (RAG X-Ray) -------------------------------------------------

    def record_trace(self, trace_id: str, model_id: str, data: dict[str, Any]) -> dict[str, Any]:
        self._storage.create_trace(trace_id, model_id, data)
        return self._storage.get_trace(trace_id)

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        return self._storage.get_trace(trace_id)

    def list_traces(
        self, model_id: str | None = None, limit: int = 50, grounded: bool | None = None
    ) -> list[dict[str, Any]]:
        return self._storage.list_traces(model_id=model_id, limit=limit, grounded=grounded)

    def diagnose_run(self, run_id: int) -> DiagnosisResult:
        """Root-cause diagnosis for one stored run. Only meaningful for
        adapters whose signals carry the confidence-bearing detail shape
        modelwatch/diagnosis/engine.py expects (currently RAGAdapter) --
        for others it degrades gracefully to "no signals are drifted"-
        style output rather than raising, since a signal with no
        recognisable confidence field just contributes zero evidence."""
        run = self._storage.get_run(run_id)
        if run is None:
            raise RunNotFoundError(f"run '{run_id}' not found")
        return diagnose(run["signals"])

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
        # A fresh baseline means "current" is now the reference again --
        # any drifted/clean streak measured against the old baseline no
        # longer means anything, so the health state resets rather than
        # carrying over a stale streak count.
        self._storage.set_health(model_id, "healthy", consecutive_drifted=0, consecutive_clean=0)

        logger.info(
            "model retrained",
            extra={"model_id": model_id, "new_version": new_version, "alerts_resolved": resolved_count},
        )
        return self._storage.get_model(model_id)
