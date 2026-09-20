"""Tests the threshold-sweep mechanics against known synthetic
probabilities -- not the real embedding/logistic-regression pipeline,
which is exercised for real in ml/threshold_tuning.py's __main__ and
documented with real numbers in docs/ml_experiment.md."""
from __future__ import annotations

from ml.threshold_tuning import sweep


def test_sweep_at_a_perfect_separation_point_gives_perfect_precision_and_recall():
    # Every "flagged" example scores >= 0.9; every "none" example scores <= 0.1.
    labels = ["flagged"] * 5 + ["none"] * 20
    probs = [0.95] * 5 + [0.05] * 20
    rows = sweep(labels, probs)

    row_50 = next(r for r in rows if r["threshold"] == 0.50)
    assert row_50["precision"] == 1.0
    assert row_50["recall"] == 1.0
    assert row_50["f1"] == 1.0


def test_sweep_recall_is_monotonically_non_increasing_as_threshold_rises():
    labels = ["flagged"] * 10 + ["none"] * 40
    probs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05] + [0.15] * 40
    rows = sweep(labels, probs)

    recalls = [r["recall"] for r in rows]
    assert all(recalls[i] >= recalls[i + 1] for i in range(len(recalls) - 1))


def test_sweep_reports_zero_precision_not_a_crash_when_nothing_is_predicted_flagged():
    labels = ["flagged", "none", "none"]
    probs = [0.05, 0.02, 0.01]  # nothing clears even the lowest threshold tried
    rows = sweep(labels, probs)

    row_10 = next(r for r in rows if r["threshold"] == 0.10)
    assert row_10["n_predicted_flagged"] == 0
    assert row_10["precision"] == 0.0
    assert row_10["recall"] == 0.0
