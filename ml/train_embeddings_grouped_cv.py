"""Fourth risk-classifier experiment: does augmenting the 13 real flagged
clauses with LLM-paraphrased variants (ml/augment_dataset.py) improve on
experiment 2's precision (0.18), evaluated in a way that can't leak?

The leakage trap, and why this isn't just experiment 2 with more rows:
a paraphrase of a clause is the same underlying fact restated, not
independent evidence. Plain leave-one-out (experiment 2's method) would
let a real clause's own paraphrase sit in the training fold while that
clause is held out for testing -- the model could match near-duplicate
phrasing and score well without generalizing to anything new. This
script uses **leave-one-GROUP-out** instead: every synthetic row shares
a `group_id` with the real clause it was paraphrased from (see
augment_dataset.py), so a real clause and all of its own paraphrases are
always held out, or trained on, together -- never split.

"none" clauses (571 of them, never augmented) are each their own
singleton group, so for them this is identical to plain leave-one-out.
Only the 13 positive groups actually change shape (1 real + up to 4
synthetic members each, instead of 1).

Run after ml/build_dataset.py and ml/augment_dataset.py:

    python -m ml.train_embeddings_grouped_cv
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

from ml.embed import embed_texts
from ml.train_embeddings_loocv import evaluate
from ml.train_risk_classifier import DATA_PATH, load_dataset

AUGMENTED_PATH = Path(__file__).parent / "data" / "clause_risk_dataset_v1_augmented.jsonl"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def load_combined() -> tuple[list[str], list[str], list[str]]:
    """Returns (texts, labels, group_ids) -- real clauses (group_id =
    "source_document:chunk_index") plus their synthetic paraphrases
    (same group_id as their real origin), if augment_dataset.py has been
    run. Falls back to real data only, each its own group (equivalent to
    plain leave-one-out), if it hasn't -- so this script still runs and
    reports something true rather than crashing."""
    texts, labels = load_dataset()
    real_rows = []
    with DATA_PATH.open() as f:
        for line in f:
            real_rows.append(json.loads(line))
    group_ids = [f"{r['source_document']}:{r['chunk_index']}" for r in real_rows]

    if AUGMENTED_PATH.exists():
        with AUGMENTED_PATH.open() as f:
            for line in f:
                row = json.loads(line)
                texts.append(row["text"])
                labels.append(row["label"])
                group_ids.append(row["group_id"])

    return texts, labels, group_ids


def run() -> dict:
    texts, labels, group_ids = load_combined()
    labels_bin = ["none" if y == "none" else "flagged" for y in labels]
    X = embed_texts(texts)
    cv = LeaveOneGroupOut()

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "1nn": KNeighborsClassifier(n_neighbors=1),
    }
    label_set = ["none", "flagged"]

    results = {}
    for name, model in candidates.items():
        preds = cross_val_predict(model, X, labels_bin, cv=cv, groups=group_ids, n_jobs=-1)
        results[name] = evaluate(labels_bin, preds, label_set)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline_preds = cross_val_predict(baseline, X, labels_bin, cv=cv, groups=group_ids, n_jobs=-1)
    results["majority_class_baseline"] = evaluate(labels_bin, baseline_preds, label_set)

    output = {
        "method": "sentence-embeddings + leave-one-GROUP-out CV (real clause + its own paraphrases held out together)",
        "dataset": {
            "n_total_rows": len(texts),
            "n_groups": len(set(group_ids)),
            "n_synthetic_rows": sum(1 for _ in open(AUGMENTED_PATH)) if AUGMENTED_PATH.exists() else 0,
            "label_distribution": dict(Counter(labels)),
            "binary_label_distribution": dict(Counter(labels_bin)),
        },
        "results": results,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with (ARTIFACTS_DIR / "results_grouped_cv.json").open("w") as f:
        json.dump(output, f, indent=2)
    return output


def _print_report(output: dict) -> None:
    d = output["dataset"]
    print(f"\n{d['n_total_rows']} total rows ({d['n_synthetic_rows']} synthetic), {d['n_groups']} groups, "
          f"leave-one-group-out.")
    print(f"Binary label distribution: {d['binary_label_distribution']}")
    for name, r in output["results"].items():
        print(f"\n{name} -- accuracy={r['accuracy']:.3f}, macro-F1={r['macro_f1']:.3f}")
        for label, m in r["per_class"].items():
            print(f"  {label:8s} precision={m['precision']:.2f} recall={m['recall']:.2f} "
                  f"f1={m['f1']:.2f} support={m['support']}")


if __name__ == "__main__":
    output = run()
    _print_report(output)
    print(f"\nSaved -> ml/artifacts/results_grouped_cv.json")
