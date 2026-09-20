"""Tests the LOO-CV experiment's mechanics against small synthetic
embeddings -- not the real sentence-transformer (slow, and not the thing
being tested here: the logic under test is the LOO-CV wiring and metric
computation, not whether MiniLM produces good embeddings)."""
from __future__ import annotations

import json

import numpy as np

from ml.train_embeddings_loocv import _candidates, _loocv_predict, evaluate


def _toy_embeddings():
    """Two well-separated synthetic clusters in 2D -- "flagged" points
    near (1, 1), "none" points near (0, 0) -- so a real classifier should
    clearly beat a majority-class baseline, and a broken LOO-CV wiring
    (e.g. leaking the held-out point into its own training fold) would
    be the main way this could still show a suspiciously perfect result
    on such trivially separable data.
    """
    rng = np.random.RandomState(0)
    none_pts = rng.normal(loc=[0, 0], scale=0.1, size=(30, 2))
    flagged_pts = rng.normal(loc=[1, 1], scale=0.1, size=(8, 2))
    X = np.vstack([none_pts, flagged_pts])
    y = ["none"] * 30 + ["flagged"] * 8
    return X, y


def test_loocv_predict_returns_one_prediction_per_example():
    X, y = _toy_embeddings()
    model = _candidates()["logistic_regression"]
    preds = _loocv_predict(model, X, y)
    assert len(preds) == len(y)


def test_loocv_on_well_separated_data_beats_majority_baseline():
    from sklearn.dummy import DummyClassifier

    X, y = _toy_embeddings()
    label_set = ["none", "flagged"]

    model = _candidates()["logistic_regression"]
    model_preds = _loocv_predict(model, X, y)
    model_results = evaluate(y, model_preds, label_set)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline_preds = _loocv_predict(baseline, X, y)
    baseline_results = evaluate(y, baseline_preds, label_set)

    assert model_results["macro_f1"] > baseline_results["macro_f1"]
    # Well-separated clusters -- a real classifier should find nearly all
    # the flagged points, not just get lucky on a couple.
    assert model_results["per_class"]["flagged"]["recall"] > 0.7


def test_candidates_includes_a_parametric_and_similarity_based_model():
    names = set(_candidates().keys())
    assert "logistic_regression" in names
    assert any("nn" in n for n in names)  # at least one nearest-neighbor variant


def test_evaluate_output_is_json_serializable():
    X, y = _toy_embeddings()
    model = _candidates()["1nn"]
    preds = _loocv_predict(model, X, y)
    results = evaluate(y, preds, ["none", "flagged"])
    json.dumps(results)
