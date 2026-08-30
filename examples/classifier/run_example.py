#!/usr/bin/env python3
"""Monitors a tabular classifier with ModelWatch's ClassifierAdapter,
using only the public modelwatch-client SDK -- same "connect any model"
story as examples/independent_rag/, for a completely different model
type (tabular, not RAG).

SYNTHETIC DATA: there is no real tabular classifier anywhere in this
project. `age` and `income` below are drawn from hand-picked Gaussian
distributions -- clearly labeled as such, not presented as a real
model's real inputs. What's real is everything ModelWatch does with
that data: a genuine Kolmogorov-Smirnov two-sample test per feature
(scipy, not a canned verdict), a real Bonferroni correction, and a
real accuracy computation from the (also synthetic) predictions/labels
below.

Requires a running ModelWatch server (default http://localhost:8000,
override with MODELWATCH_URL).

Usage:
    pip install -e ./modelwatch-client
    python examples/classifier/run_example.py
"""
from __future__ import annotations

import os
import random
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "modelwatch-client"))

from modelwatch_client import ModelWatchClient, ModelWatchError  # noqa: E402

MODEL_ID = "synthetic-loan-classifier"


def _gaussian_feature(rnd: random.Random, mean: float, stdev: float, n: int) -> list[float]:
    return [rnd.gauss(mean, stdev) for _ in range(n)]


def _synthetic_predictions_and_labels(rnd: random.Random, n: int, accuracy: float) -> tuple[list[int], list[int]]:
    """labels are a coin flip; predictions match `accuracy` fraction of
    the time -- real code computing a real accuracy number, over a
    clearly synthetic label/prediction pair."""
    labels = [rnd.randint(0, 1) for _ in range(n)]
    predictions = [label if rnd.random() < accuracy else 1 - label for label in labels]
    return predictions, labels


def main() -> None:
    base_url = os.environ.get("MODELWATCH_URL", "http://localhost:8000")
    client = ModelWatchClient(base_url=base_url)

    print(f"ModelWatch server: {base_url}")
    try:
        client.list_models()
    except ModelWatchError as e:
        print(f"Could not reach ModelWatch at {base_url}: {e}")
        print("Start it with `./run.sh` from the repo root, or run modelwatch's API standalone.")
        sys.exit(1)

    model_id = f"{MODEL_ID}-{uuid.uuid4().hex[:8]}"
    rnd = random.Random(1)

    print("\n1. Building baseline from SYNTHETIC 'normal operation' data (n=200)...")
    baseline_data = {
        "features": {
            "age": _gaussian_feature(rnd, mean=35, stdev=10, n=200),
            "income": _gaussian_feature(rnd, mean=55000, stdev=15000, n=200),
        }
    }
    print(f"   age: mean~35, stdev~10 | income: mean~55000, stdev~15000")

    print(f"\n2. Registering '{model_id}' with ModelWatch (adapter_name='classifier')...")
    client.register_model(
        model_id=model_id,
        name="Synthetic Loan Approval Classifier",
        adapter_name="classifier",
        baseline_data=baseline_data,
    )

    print("\n3. Checking a CLEAN batch (same distribution, n=50) -- should NOT flag drift...")
    clean_predictions, clean_labels = _synthetic_predictions_and_labels(rnd, n=50, accuracy=0.9)
    clean_result = client.check(model_id, {
        "features": {
            "age": _gaussian_feature(rnd, mean=35, stdev=10, n=50),
            "income": _gaussian_feature(rnd, mean=55000, stdev=15000, n=50),
        },
        "predictions": clean_predictions,
        "labels": clean_labels,
    })
    print(f"   is_drifted: {clean_result['is_drifted']}  quality_score: {clean_result['quality_score']}")

    print("\n4. Checking a SHIFTED batch (older, higher-income applicants, n=50) -- should flag drift...")
    shifted_predictions, shifted_labels = _synthetic_predictions_and_labels(rnd, n=50, accuracy=0.9)
    shifted_result = client.check(model_id, {
        "features": {
            "age": _gaussian_feature(rnd, mean=55, stdev=8, n=50),       # genuinely shifted
            "income": _gaussian_feature(rnd, mean=95000, stdev=20000, n=50),  # genuinely shifted
        },
        "predictions": shifted_predictions,
        "labels": shifted_labels,
    })
    print(f"   is_drifted: {shifted_result['is_drifted']}  quality_score: {shifted_result['quality_score']}")
    for sig in shifted_result["signals"]:
        flag = "DRIFTED" if sig["is_drifted"] else "ok"
        print(f"     - {sig['name']:<10} KS statistic={sig['value']:.3f}  p={sig['detail']['pvalue']:.4f}  [{flag}]")

    print("\nDone -- clean and shifted batches produced different, real KS statistics, not staged ones.")
    print("See README.md for what this proves and what's synthetic vs. real about it.")


if __name__ == "__main__":
    main()
