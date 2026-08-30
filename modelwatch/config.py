"""Central configuration for ModelWatch, sourced from environment variables.

Nothing here is a hardcoded magic value used elsewhere in the codebase --
thresholds and paths all flow through this module so they can be tuned
without touching engine/adapter logic.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _float_env(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    # Storage
    db_path: str = os.environ.get("MODELWATCH_DB_PATH", "modelwatch.db")

    # API / dashboard
    api_host: str = os.environ.get("MODELWATCH_API_HOST", "0.0.0.0")
    api_port: int = _int_env("MODELWATCH_API_PORT", 8000)

    # ClassifierAdapter: Kolmogorov-Smirnov two-sample test significance level.
    # A feature's distribution is flagged as drifted when the KS test p-value
    # falls below this threshold (i.e. we reject the "same distribution" null).
    ks_pvalue_threshold: float = _float_env("MODELWATCH_KS_PVALUE_THRESHOLD", 0.05)

    # Fraction of features that must be individually flagged as drifted for
    # the ClassifierAdapter's overall check_drift() result to be is_drifted=True.
    classifier_drift_feature_fraction: float = _float_env(
        "MODELWATCH_CLASSIFIER_DRIFT_FEATURE_FRACTION", 0.3
    )

    # Divide ks_pvalue_threshold by the number of features tested, so testing
    # many features separately doesn't inflate the false-alarm rate. Set to
    # false to test each feature at the raw threshold (more sensitive to small
    # single-feature drift, but noisier -- see classifier_adapter.py).
    classifier_bonferroni_correction: bool = _bool_env(
        "MODELWATCH_CLASSIFIER_BONFERRONI_CORRECTION", True
    )

    # LLMAdapter: minimum average TF-IDF cosine similarity between actual and
    # expected answers before a batch is considered drifted.
    llm_similarity_threshold: float = _float_env("MODELWATCH_LLM_SIMILARITY_THRESHOLD", 0.35)

    # LiveTelemetryAdapter: how far an operational metric may move from its
    # baseline before it counts as drift. Absolute for rates, a ratio for
    # latency. min_events guards against rates computed from a handful of
    # requests, where one unlucky refusal would look like a 33% regression.
    telemetry_refusal_tolerance: float = _float_env("MODELWATCH_TELEMETRY_REFUSAL_TOLERANCE", 0.20)
    telemetry_citation_tolerance: float = _float_env("MODELWATCH_TELEMETRY_CITATION_TOLERANCE", 0.25)
    telemetry_latency_multiplier: float = _float_env("MODELWATCH_TELEMETRY_LATENCY_MULTIPLIER", 2.5)
    telemetry_min_events: int = _int_env("MODELWATCH_TELEMETRY_MIN_EVENTS", 5)

    # RAGAdapter: significance level for its statistical detectors (KS,
    # two-proportion z-test), and the minimum events required in both the
    # baseline and current batch before any of them run at all -- below
    # this, a distributional test's p-value looks precise but isn't.
    rag_alpha: float = _float_env("MODELWATCH_RAG_ALPHA", 0.05)
    rag_min_events: int = _int_env("MODELWATCH_RAG_MIN_EVENTS", 8)

    # Alert hysteresis (modelwatch/core/health.py). Defaults reproduce the
    # original "alert on the very first drifted check, clear on the very
    # first clean one" behavior -- raise degraded_after_consecutive (and
    # optionally recovery_after_consecutive) to require sustained drift
    # before paging, which is the recommended production setting once a
    # model has enough traffic for a couple of consecutive batches to mean
    # something.
    health_warning_after_consecutive: int = _int_env("MODELWATCH_WARNING_AFTER_CONSECUTIVE", 1)
    health_degraded_after_consecutive: int = _int_env("MODELWATCH_DEGRADED_AFTER_CONSECUTIVE", 1)
    health_recovery_after_consecutive: int = _int_env("MODELWATCH_RECOVERY_AFTER_CONSECUTIVE", 1)

    # Alert delivery (modelwatch/alerts/notifier.py). Unset by default --
    # a fresh install stays silent rather than failing to reach a webhook
    # nobody configured. Set MODELWATCH_ALERT_WEBHOOK_URL to a Slack
    # Incoming Webhook URL (with format left as "slack") or any generic
    # webhook receiver's URL (format "generic").
    alert_webhook_url: str = os.environ.get("MODELWATCH_ALERT_WEBHOOK_URL", "")
    alert_webhook_format: str = os.environ.get("MODELWATCH_ALERT_WEBHOOK_FORMAT", "slack")

    # Default logging level for library/service code.
    log_level: str = os.environ.get("MODELWATCH_LOG_LEVEL", "INFO")

    # Optional MLflow export (modelwatch/integrations/mlflow_export.py).
    # Off by default -- mlflow is not a hard dependency of this project
    # (see that module's docstring), so a fresh install must not require
    # it. tracking_uri defaults to a local SQLite-backed store (no server
    # needed) -- mlflow's older plain file store ("file:./mlruns") is in
    # maintenance mode as of mlflow 3.x and rejects new runs by default.
    # Point this at a real MLflow tracking server URL to use one instead.
    mlflow_enabled: bool = _bool_env("MODELWATCH_MLFLOW_ENABLED", False)
    mlflow_tracking_uri: str = os.environ.get("MODELWATCH_MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow_experiment_name: str = os.environ.get("MODELWATCH_MLFLOW_EXPERIMENT_NAME", "modelwatch")


config = Config()
