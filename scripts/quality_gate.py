#!/usr/bin/env python3
"""CI quality gate: runs the Sanad RAG evaluation dataset against the
live model and fails (non-zero exit) if any metric regressed beyond its
configured tolerance versus a stored baseline.

First run (or after a deliberate quality change), capture a baseline:
    python scripts/quality_gate.py --save-baseline

Every subsequent run compares against it:
    python scripts/quality_gate.py

Requires Ollama serving Sanad's configured model -- there is no
fake-LLM mode, since a quality gate that doesn't call the real model
isn't measuring anything real. This mirrors sanad/evaluation/run_eval.py
but is meant for a CI step, not interactive use: it prints a compact
PASS/REGRESSION table and returns exit code 1 on any regression.

Tolerances are deliberately not just "must not decrease at all" -- the
same evaluation run against the same model produces slightly different
numbers between runs (LLM sampling, timing), so a zero-tolerance gate
would fail on noise as often as on a real regression. Widen/narrow via
--tolerance-quality / --tolerance-rate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanad.evaluation.dataset import load_dataset  # noqa: E402
from sanad.evaluation.runner import run_evaluation  # noqa: E402
from sanad.rag.llm_client import OllamaClient  # noqa: E402
from sanad.rag.vector_store import VectorStore  # noqa: E402

DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "datasets" / "sanad_eval" / "sanad_eval_v1.jsonl"
BASELINE_PATH = Path(__file__).resolve().parent.parent / "experiments" / "results" / "quality_gate_baseline.json"

# (metric key on EvalSummary.to_dict(), higher_is_better, default tolerance)
_GATED_METRICS = [
    ("retrieval_hit_rate", True, 0.10),
    ("citation_correctness", True, 0.15),
    ("mean_semantic_similarity", True, 0.10),
    ("refusal_rate", False, 0.15),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--baseline", default=str(BASELINE_PATH))
    parser.add_argument("--save-baseline", action="store_true", help="record this run as the new baseline instead of gating against it")
    parser.add_argument("--tolerance", type=float, default=None, help="override every metric's tolerance at once")
    args = parser.parse_args()

    cases = load_dataset(args.dataset)
    chroma_path = f"/tmp/sanad_quality_gate_{int(time.time())}"
    vector_store = VectorStore(persist_path=chroma_path)
    llm_client = OllamaClient()

    print(f"Running {len(cases)} eval cases against {llm_client.model} ...")
    _results, summary = run_evaluation(cases, vector_store, llm_client)
    current = summary.to_dict()

    baseline_path = Path(args.baseline)
    if args.save_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps({"model": llm_client.model, "summary": current}, indent=2))
        print(f"Saved baseline to {baseline_path}")
        return

    if not baseline_path.exists():
        print(f"No baseline at {baseline_path} -- run with --save-baseline first.")
        sys.exit(2)

    baseline = json.loads(baseline_path.read_text())["summary"]

    print(f"\n{'MODELWATCH QUALITY GATE':^60}")
    print(f"{'metric':<26}{'baseline':>10}{'current':>10}{'status':>14}")
    print("-" * 60)

    any_regression = False
    for key, higher_is_better, default_tol in _GATED_METRICS:
        tol = args.tolerance if args.tolerance is not None else default_tol
        base_val, curr_val = baseline[key], current[key]
        delta = curr_val - base_val
        regressed = (delta < -tol) if higher_is_better else (delta > tol)
        any_regression = any_regression or regressed
        status = "REGRESSION" if regressed else "PASS"
        print(f"{key:<26}{base_val:>10.3f}{curr_val:>10.3f}{status:>14}")

    print("-" * 60)
    if any_regression:
        print("REGRESSION DETECTED -- DEPLOYMENT BLOCKED")
        sys.exit(1)
    print("All metrics within tolerance.")


if __name__ == "__main__":
    main()
