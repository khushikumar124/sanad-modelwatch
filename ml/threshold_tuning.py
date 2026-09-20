"""Third risk-classifier experiment: is 62% recall / 18% precision (the
default 0.5 probability threshold, see docs/ml_experiment.md's
experiment 2) the only operating point available, or can precision be
raised by simply moving the decision threshold -- no new data, no new
model, same embeddings, same logistic regression?

This is not a new model and not new data: it reuses experiment 2's exact
pooled leave-one-out predicted *probabilities* (not the 0/1 predictions)
and asks what precision/recall look like at different cutoffs applied to
those same probabilities. If a better precision/recall balance exists at
another threshold, it was already latent in experiment 2's output -- this
just makes it visible instead of only ever reading off the default 0.5
cutoff.

Run after ml/build_dataset.py:

    python -m ml.threshold_tuning
"""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from ml.embed import embed_texts
from ml.train_risk_classifier import load_dataset

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

# Coarser than a fine-grained sweep on purpose: with only 13 positive
# examples, adjacent thresholds a few hundredths apart can't move the
# confusion matrix by anything but rounding noise (there just aren't
# enough distinct predicted probabilities to distinguish them) -- these
# ten points already span every operating point this dataset can
# meaningfully support.
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def compute_pooled_probabilities() -> tuple[list[str], list[float]]:
    """The binary "flagged vs none" task, logistic regression on
    embeddings -- the one configuration in experiment 2 that showed a
    real signal (1-NN and 5-NN don't expose a comparably useful
    probability to threshold). Returns (true_labels, P(flagged)) pooled
    across every leave-one-out fold."""
    texts, labels = load_dataset()
    labels_bin = ["none" if y == "none" else "flagged" for y in labels]
    X = embed_texts(texts)

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    proba = cross_val_predict(model, X, labels_bin, cv=LeaveOneOut(), method="predict_proba", n_jobs=-1)
    classes = list(model.fit(X, labels_bin).classes_)  # class order for the proba columns
    flagged_idx = classes.index("flagged")
    return labels_bin, proba[:, flagged_idx].tolist()


def sweep(labels_bin: list[str], p_flagged: list[float]) -> list[dict]:
    rows = []
    for t in THRESHOLDS:
        preds = ["flagged" if p >= t else "none" for p in p_flagged]
        rows.append(
            {
                "threshold": t,
                "precision": precision_score(labels_bin, preds, pos_label="flagged", zero_division=0),
                "recall": recall_score(labels_bin, preds, pos_label="flagged", zero_division=0),
                "f1": f1_score(labels_bin, preds, pos_label="flagged", zero_division=0),
                "n_predicted_flagged": preds.count("flagged"),
            }
        )
    return rows


def run() -> dict:
    labels_bin, p_flagged = compute_pooled_probabilities()
    rows = sweep(labels_bin, p_flagged)
    results = {
        "n_total": len(labels_bin),
        "n_positive": labels_bin.count("flagged"),
        "sweep": rows,
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with (ARTIFACTS_DIR / "results_threshold_sweep.json").open("w") as f:
        json.dump(results, f, indent=2)
    return results


def _print_report(results: dict) -> None:
    print(f"\n{results['n_positive']} positive of {results['n_total']} clauses, "
          f"pooled leave-one-out probabilities, logistic regression on embeddings.\n")
    print(f"{'threshold':>9} {'precision':>10} {'recall':>8} {'f1':>6} {'n_flagged':>10}")
    for r in results["sweep"]:
        print(f"{r['threshold']:>9.2f} {r['precision']:>10.2f} {r['recall']:>8.2f} "
              f"{r['f1']:>6.2f} {r['n_predicted_flagged']:>10}")


if __name__ == "__main__":
    results = run()
    _print_report(results)
    print(f"\nSaved results -> ml/artifacts/results_threshold_sweep.json")
