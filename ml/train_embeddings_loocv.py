"""Second risk-classifier experiment, fixing the two real methodological
weaknesses the first one (train_risk_classifier.py) surfaced -- not a
retry with different random seeds until something looks better, but two
principled changes:

1. **TF-IDF -> sentence embeddings.** A bag-of-words model needs
   near-exact vocabulary overlap between training and test clauses to
   generalize; with ~10 positive training examples spread across
   different risk categories and phrasings, that's exactly why the first
   experiment learned nothing (see docs/ml_experiment.md's probability-
   ranking diagnosis). Embeddings (the same all-MiniLM-L6-v2 Sanad's own
   retrieval already uses -- see ml/embed.py) let a match fire on
   semantic similarity instead of shared words.

2. **A single 75/25 split -> leave-one-out cross-validation.** With only
   13 positive examples total, a single split leaves ~3 in the test set
   -- one of the least statistically meaningful ways to evaluate this
   dataset (a single unlucky/lucky split swings the result heavily).
   LOO-CV evaluates every one of the 584 clauses exactly once, as the
   held-out example, trained on all 583 others -- using the scarce
   positive examples far more efficiently, and removing single-split luck
   as a variable.

Both changes are applied together; results are compared against the
same majority-class baseline, computed under the identical LOO-CV
procedure so the comparison is apples-to-apples. If this still doesn't
beat baseline, that is reported plainly, same as the first experiment.

Run after `python -m ml.build_dataset` (uses the same dataset file):

    python -m ml.train_embeddings_loocv
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

from ml.embed import embed_texts
from ml.train_risk_classifier import DATA_PATH, load_dataset

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def evaluate(y_true, y_pred, labels: list[str]) -> dict:
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "per_class": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": report[label]["support"],
            }
            for label in labels
        },
        "accuracy": report["accuracy"],
    }


def _loocv_predict(model, X: np.ndarray, y: list[str]) -> np.ndarray:
    """Pooled out-of-fold predictions under leave-one-out CV -- every
    example is predicted exactly once, by a model that never saw it
    during that fold's fit."""
    return cross_val_predict(model, X, y, cv=LeaveOneOut(), n_jobs=-1)


# Three genuinely different candidates, not the same model with cosmetic
# tweaks: a parametric classifier (learns a decision boundary over the
# whole embedding space), and two similarity-based few-shot approaches
# (1-nearest-neighbor: "is the single closest known clause flagged?";
# 5-nearest, distance-weighted: hedges against one mislabeled neighbor).
# Few-shot / similarity search is generally the more appropriate family
# when there are only a handful of positive examples, which is exactly
# this dataset's constraint -- worth testing rather than assuming.
def _candidates() -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "1nn": KNeighborsClassifier(n_neighbors=1),
        "5nn_distance_weighted": KNeighborsClassifier(n_neighbors=5, weights="distance"),
    }


def run() -> dict:
    texts, labels = load_dataset()
    X = embed_texts(texts)

    to_binary = lambda ys: ["none" if y == "none" else "flagged" for y in ys]
    labels_bin = to_binary(labels)
    label_set_4 = sorted(set(labels), key=lambda l: (l != "none", l))
    label_set_bin = ["none", "flagged"]

    results_4, results_bin = {}, {}
    for name, model in _candidates().items():
        results_4[name] = evaluate(labels, _loocv_predict(model, X, labels), label_set_4)
        results_bin[name] = evaluate(labels_bin, _loocv_predict(model, X, labels_bin), label_set_bin)

    baseline = DummyClassifier(strategy="most_frequent")
    results_4["majority_class_baseline"] = evaluate(labels, _loocv_predict(baseline, X, labels), label_set_4)
    results_bin["majority_class_baseline"] = evaluate(
        labels_bin, _loocv_predict(baseline, X, labels_bin), label_set_bin
    )

    results = {
        "method": "sentence-embeddings (all-MiniLM-L6-v2) + leave-one-out cross-validation",
        "dataset": {
            "path": str(DATA_PATH),
            "n_total": len(texts),
            "label_distribution": dict(Counter(labels)),
            "binary_label_distribution": dict(Counter(labels_bin)),
        },
        "results_4class": results_4,
        "results_binary": results_bin,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with (ARTIFACTS_DIR / "results_embeddings_loocv.json").open("w") as f:
        json.dump(results, f, indent=2)

    return results


def _print_report(results: dict) -> None:
    d = results["dataset"]
    print(f"\nDataset: {d['n_total']} clauses (leave-one-out -- every clause is its own held-out test)")
    print(f"4-class label distribution: {d['label_distribution']}")
    print(f"Binary label distribution: {d['binary_label_distribution']}")

    for task_name, task_results in [("4-CLASS", results["results_4class"]), ("BINARY (flagged/none)", results["results_binary"])]:
        print(f"\n=== {task_name} ===")
        for model_name, r in task_results.items():
            print(f"\n{model_name} -- accuracy={r['accuracy']:.3f}, macro-F1={r['macro_f1']:.3f}")
            for label, m in r["per_class"].items():
                print(f"  {label:8s} precision={m['precision']:.2f} recall={m['recall']:.2f} "
                      f"f1={m['f1']:.2f} support={m['support']}")


if __name__ == "__main__":
    results = run()
    _print_report(results)
    print(f"\nSaved results -> ml/artifacts/results_embeddings_loocv.json")
