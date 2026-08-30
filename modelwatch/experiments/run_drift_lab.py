"""CLI: run a Drift Lab scenario against Sanad's real pipeline + Ollama
and print the measured, real detection/diagnosis result.

Usage:
    python -m modelwatch.experiments.run_drift_lab retrieval_narrowing
    python -m modelwatch.experiments.run_drift_lab chunk_fragmentation --limit 8
    python -m modelwatch.experiments.run_drift_lab retrieval_narrowing --out experiments/results/retrieval_narrowing.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from modelwatch.experiments.drift_lab import DEFAULT_DATASET, SCENARIOS
from sanad.evaluation.dataset import load_dataset
from sanad.rag.llm_client import OllamaClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=sorted(SCENARIOS))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--limit", type=int, default=None, help="only use the first N cases (each case is 2 LLM calls)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cases = load_dataset(args.dataset)
    if args.limit:
        cases = cases[: args.limit]

    llm_client = OllamaClient()
    print(f"Running scenario '{args.scenario}' on {len(cases)} cases against {llm_client.model} ...")

    result = SCENARIOS[args.scenario](cases, llm_client)
    result.print_report()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
