"""Trains and evaluates a supervised clause-risk-severity classifier as a
genuine train/test experiment, separate from (and evaluated against) the
deterministic rule engine that generated its labels.

Question this actually answers: **can a lightweight learned model
reproduce sanad/features/risk_flagger.py's severity calls from clause
text alone, on clauses it never saw during training?** That's a real,
falsifiable, measured question -- not a claim that this model is "the"
risk detector, or that it's more correct than the rules (it was trained
to imitate them, so it cannot be more correct than its own labels).

Method: TF-IDF (word 1-2 grams) + multinomial logistic regression,
class-balanced (the dataset is heavily skewed toward "none" -- most
clauses in a contract are not risk-relevant). Compared against a
majority-class baseline, because "82% accuracy" is meaningless on its
own when 82% of the data is one class.

Run from the repository root, after `python -m ml.build_dataset`:

    python -m ml.train_risk_classifier
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = Path(__file__).parent / "data" / "clause_risk_dataset_v1.jsonl"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
RANDOM_STATE = 0
TEST_SIZE = 0.25


def load_dataset() -> tuple[list[str], list[str]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} doesn't exist yet -- run `python -m ml.build_dataset` first."
        )
    texts, labels = [], []
    with DATA_PATH.open() as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])
            labels.append(row["label"])
    return texts, labels


def _can_stratify(labels: list[str]) -> bool:
    """train_test_split's stratify needs every class to have at least 2
    members (one for train, one for test). A silver-label dataset built
    from a handful of sample contracts can easily have a severity class
    with a single example -- fall back to a plain random split rather
    than crashing, and say so in the results so the tradeoff is visible,
    not silently swallowed."""
    return min(Counter(labels).values()) >= 2


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def evaluate(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
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


def _fit_eval(X_train, y_train, X_test, y_test, label_set: list[str]) -> tuple[dict, dict, Pipeline]:
    """Fits the real pipeline and a majority-class baseline on the same
    split, returns both evaluations plus the fitted pipeline."""
    model = build_pipeline()
    model.fit(X_train, y_train)
    model_results = evaluate(y_test, model.predict(X_test), label_set)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_results = evaluate(y_test, baseline.predict(X_test), label_set)

    return model_results, baseline_results, model


def run() -> dict:
    texts, labels = load_dataset()
    label_set = sorted(set(labels), key=lambda l: (l != "none", l))  # "none" first for readable tables
    stratify = labels if _can_stratify(labels) else None

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )

    model_results, baseline_results, model = _fit_eval(X_train, y_train, X_test, y_test, label_set)

    # Secondary experiment: the same held-out split, collapsed to a binary
    # "flagged vs none" question. Necessary because the 4-class dataset has
    # only 2-7 examples for "high"/"medium"/"low" each -- too few for any
    # classifier to learn a real decision boundary for those individually
    # (see docs/ml_experiment.md's results: the 4-class model turns out
    # statistically identical to the majority-class baseline). Collapsing
    # severity to a single "flagged" class still leaves a real, if small,
    # positive class (13 of 584 clauses) and answers a coarser but
    # genuinely learnable question. Re-labels the *same* train/test
    # partition rather than re-splitting, so the two experiments are
    # directly comparable rather than evaluated on different held-out sets.
    to_binary = lambda ys: ["none" if y == "none" else "flagged" for y in ys]
    y_train_bin, y_test_bin = to_binary(y_train), to_binary(y_test)
    label_set_bin = ["none", "flagged"]
    model_results_bin, baseline_results_bin, _ = _fit_eval(
        X_train, y_train_bin, X_test, y_test_bin, label_set_bin
    )

    results = {
        "dataset": {
            "path": str(DATA_PATH),
            "n_total": len(texts),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "label_distribution": dict(Counter(labels)),
            "binary_label_distribution": dict(Counter(to_binary(labels))),
            "stratified_split": stratify is not None,
        },
        "model": {
            "pipeline": "TfidfVectorizer(1-2 grams) + LogisticRegression(class_weight=balanced)",
            "random_state": RANDOM_STATE,
        },
        "results": {
            "learned_model": model_results,
            "majority_class_baseline": baseline_results,
            "learned_model_binary": model_results_bin,
            "majority_class_baseline_binary": baseline_results_bin,
        },
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS_DIR / "risk_classifier.joblib")
    with (ARTIFACTS_DIR / "results.json").open("w") as f:
        json.dump(results, f, indent=2)

    return results


def _print_report(results: dict) -> None:
    d = results["dataset"]
    print(f"\nDataset: {d['n_total']} clauses ({d['n_train']} train / {d['n_test']} test)")
    print(f"4-class label distribution: {d['label_distribution']}")
    print(f"Binary label distribution: {d['binary_label_distribution']}")
    if not d["stratified_split"]:
        print("NOTE: at least one class had <2 examples -- split was NOT stratified.")

    sections = [
        ("4-class: learned model", "learned_model"),
        ("4-class: majority-class baseline", "majority_class_baseline"),
        ("Binary (flagged/none): learned model", "learned_model_binary"),
        ("Binary (flagged/none): majority-class baseline", "majority_class_baseline_binary"),
    ]
    for name, key in sections:
        r = results["results"][key]
        print(f"\n{name} -- accuracy={r['accuracy']:.3f}, macro-F1={r['macro_f1']:.3f}")
        for label, m in r["per_class"].items():
            print(f"  {label:8s} precision={m['precision']:.2f} recall={m['recall']:.2f} "
                  f"f1={m['f1']:.2f} support={m['support']}")


if __name__ == "__main__":
    results = run()
    _print_report(results)
    print(f"\nSaved model -> ml/artifacts/risk_classifier.joblib")
    print(f"Saved results -> ml/artifacts/results.json")
