#!/usr/bin/env python3
"""Runs modelwatch/experiments/benchmark.py and prints/saves the result.

DEMO / SIMULATED DATA -- see benchmark.py's module docstring. This
compares detection methods against synthetic trials with known ground
truth; it is not a live-traffic measurement of Sanad's own quality.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --n-trials 500 --seed 1
    python scripts/run_benchmark.py --register   # also records to modelwatch.db's experiment registry
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modelwatch.experiments.benchmark import run_ablation, run_benchmark  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "experiments" / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--n-events", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--register", action="store_true", help="also record this run in modelwatch.db")
    parser.add_argument("--ablation", action="store_true", help="run the signal-ablation study instead of the method comparison")
    args = parser.parse_args()

    run_fn = run_ablation if args.ablation else run_benchmark
    results = run_fn(n_trials=args.n_trials, n_events=args.n_events, seed=args.seed)
    kind = "ablation" if args.ablation else "benchmark"

    print(f"DEMO/SIMULATED {kind}: {args.n_trials} synthetic trials, seed={args.seed}\n")
    header = f"{'method':<26}{'precision':>10}{'recall':>10}{'f1':>10}{'fpr':>10}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(f"{name:<26}{r.precision:>10.2f}{r.recall:>10.2f}{r.f1:>10.2f}{r.false_positive_rate:>10.2f}")

    print("\nrecall by injected drift type (blind spots show up as low numbers here):")
    for name, r in results.items():
        breakdown = ", ".join(f"{t}={v:.2f}" for t, v in sorted(r.recall_by_drift_type.items()))
        print(f"  {name:<26}{breakdown}")

    payload = {
        "n_trials": args.n_trials,
        "n_events": args.n_events,
        "seed": args.seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": {name: r.to_dict() for name, r in results.items()},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{kind}_seed{args.seed}_n{args.n_trials}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote results to {out_path}")

    if args.register:
        from modelwatch.config import config
        from modelwatch.core.storage import Storage

        storage = Storage(config.db_path)
        experiment_id = storage.create_experiment(
            name=f"{kind}-seed{args.seed}-n{args.n_trials}",
            kind=kind,
            config={"n_trials": args.n_trials, "n_events": args.n_events, "seed": args.seed},
            results=payload["results"],
        )
        storage.close()
        print(f"Recorded as experiment #{experiment_id} in {config.db_path}")


if __name__ == "__main__":
    main()
