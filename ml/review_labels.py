"""Prints the clause-risk dataset's labels for fast human review --
purely a read-only display helper, not a labeling tool: correcting a
label means hand-editing ml/data/clause_risk_dataset_v1.jsonl directly
(each line is one JSON object: text/label/source_document/chunk_index),
then re-running ml/train_embeddings_loocv.py to see whether corrected
labels change the measured results.

Every flagged (high/medium/low) clause is shown in full, since there are
only 13 of them -- the whole point of ml/'s "silver label" caveat (see
build_dataset.py's docstring) is that these came from the rule engine,
not a human, so this is where a human's judgment adds the most value.
A random sample of "none" clauses is also shown, to spot-check for
clauses the rule engine may have missed entirely (a false negative the
classifier experiments have no way to detect on their own, since their
labels ARE the rule engine's output).

Run from the repository root, after `python -m ml.build_dataset`:

    python -m ml.review_labels
    python -m ml.review_labels --none-sample 20 --seed 1
"""
from __future__ import annotations

import argparse
import json
import random

from ml.train_risk_classifier import DATA_PATH


def _print_row(i: int, text: str, label: str, source: str, chunk_index: int) -> None:
    print(f"\n[{i}] label={label}  source={source}  chunk={chunk_index}")
    print("-" * 70)
    print(text.strip())


def run(none_sample: int, seed: int) -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} doesn't exist yet -- run `python -m ml.build_dataset` first.")

    rows = []
    with DATA_PATH.open() as f:
        for line in f:
            rows.append(json.loads(line))

    flagged = [r for r in rows if r["label"] != "none"]
    none_rows = [r for r in rows if r["label"] == "none"]

    print("=" * 70)
    print(f"FLAGGED CLAUSES -- all {len(flagged)}, every one the rule engine fired on.")
    print("Question for each: does the assigned severity (high/medium/low) match")
    print("your own read of the clause? Note the [i] index for any you'd correct.")
    print("=" * 70)
    for i, r in enumerate(flagged):
        _print_row(i, r["text"], r["label"], r["source_document"], r["chunk_index"])

    if none_sample > 0:
        rng = random.Random(seed)
        sample = rng.sample(none_rows, min(none_sample, len(none_rows)))
        print("\n\n" + "=" * 70)
        print(f"RANDOM SAMPLE OF {len(sample)} \"none\" CLAUSES (seed={seed}) -- spot-check for")
        print("anything that looks genuinely risky but the rule engine didn't catch.")
        print("=" * 70)
        for i, r in enumerate(sample):
            _print_row(i, r["text"], r["label"], r["source_document"], r["chunk_index"])

    print(f"\n\nTotal: {len(flagged)} flagged, {len(none_rows)} none, {len(rows)} clauses.")
    print(f"To correct a label: edit {DATA_PATH} directly (one JSON object per line),")
    print("then re-run `python -m ml.train_embeddings_loocv` to see the effect.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--none-sample", type=int, default=15, help="how many 'none' clauses to spot-check (0 to skip)")
    parser.add_argument("--seed", type=int, default=0, help="random seed for the 'none' sample")
    args = parser.parse_args()
    run(args.none_sample, args.seed)
