"""ModelAdapter for tabular classifier models.

Drift: per-feature Kolmogorov-Smirnov two-sample test comparing incoming
feature distributions against a stored baseline sample. KS is chosen over a
simple mean/variance comparison because it's distribution-shape-agnostic --
it catches shifts, spread changes, and multimodal changes a moments-based
test would miss, and needs no assumption of normality.

Quality: accuracy against ground-truth labels, when both predictions and
labels are supplied in a batch.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

from scipy import stats

from modelwatch.config import config
from modelwatch.core.adapter_base import DriftCheckResult, ModelAdapter, SignalResult

logger = logging.getLogger(__name__)


class ClassifierAdapter(ModelAdapter):
    adapter_name = "classifier"

    def __init__(
        self,
        ks_pvalue_threshold: float | None = None,
        drift_feature_fraction: float | None = None,
    ):
        self.ks_pvalue_threshold = ks_pvalue_threshold or config.ks_pvalue_threshold
        self.drift_feature_fraction = (
            drift_feature_fraction
            if drift_feature_fraction is not None
            else config.classifier_drift_feature_fraction
        )

    def build_baseline(self, data: dict[str, Any]) -> dict[str, Any]:
        """data = {"features": {feature_name: [values, ...]}}"""
        features: dict[str, Sequence[float]] = data["features"]
        return {
            "features": {name: list(values) for name, values in features.items()},
            "n_samples": len(next(iter(features.values()))) if features else 0,
        }

    def check_drift(self, baseline: dict[str, Any], new_data: dict[str, Any]) -> DriftCheckResult:
        """new_data = {"features": {...}, "predictions": [...]?, "labels": [...]?}"""
        baseline_features: dict[str, list[float]] = baseline["features"]
        new_features: dict[str, Sequence[float]] = new_data["features"]

        signals: list[SignalResult] = []
        for name, baseline_values in baseline_features.items():
            if name not in new_features:
                logger.warning("feature missing from new batch", extra={"feature": name})
                continue
            statistic, pvalue = stats.ks_2samp(baseline_values, new_features[name])
            is_drifted = bool(pvalue < self.ks_pvalue_threshold)
            signals.append(
                SignalResult(
                    name=name,
                    value=float(statistic),
                    is_drifted=is_drifted,
                    detail={
                        "pvalue": float(pvalue),
                        "n_baseline": len(baseline_values),
                        "n_new": len(new_features[name]),
                    },
                )
            )

        drifted_count = sum(1 for s in signals if s.is_drifted)
        drift_score = drifted_count / len(signals) if signals else 0.0
        is_drifted = drift_score >= self.drift_feature_fraction

        quality_score = None
        predictions = new_data.get("predictions")
        labels = new_data.get("labels")
        if predictions is not None and labels is not None and len(labels) > 0:
            correct = sum(1 for p, l in zip(predictions, labels) if p == l)
            quality_score = correct / len(labels)

        return DriftCheckResult(
            drift_score=drift_score,
            quality_score=quality_score,
            is_drifted=is_drifted,
            signals=signals,
        )
