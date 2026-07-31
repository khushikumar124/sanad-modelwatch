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

    # LLMAdapter: minimum average TF-IDF cosine similarity between actual and
    # expected answers before a batch is considered drifted.
    llm_similarity_threshold: float = _float_env("MODELWATCH_LLM_SIMILARITY_THRESHOLD", 0.35)

    # Default logging level for library/service code.
    log_level: str = os.environ.get("MODELWATCH_LOG_LEVEL", "INFO")


config = Config()
