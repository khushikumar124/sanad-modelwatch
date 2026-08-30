#!/usr/bin/env python3
"""Recommends a MODELWATCH_DEGRADED_AFTER_CONSECUTIVE value from a real
measured false-positive rate, instead of guessing one.

Runs modelwatch/experiments/benchmark.py's clean (no-drift-injected)
trials for the "rag_adapter_full" method, uses its measured
false_positive_rate as the single-check FPR, and calibrates the smallest
consecutive-drifted-checks threshold that brings the estimated
incident-level false-positive rate at or below --target.

This prints a recommendation; it does NOT modify any running
configuration. Review the number, then set
MODELWATCH_DEGRADED_AFTER_CONSECUTIVE yourself (env var, or .env) if you
want to use it -- see modelwatch/core/calibration.py's docstring for the
independence assumption behind the math, which is why this stays a
human-reviewed recommendation rather than something auto-applied at
startup.

Usage:
    python scripts/calibrate_hysteresis.py
    python scripts/calibrate_hysteresis.py --n-trials 500 --target 0.005
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modelwatch.core.calibration import calibrate_degraded_after_consecutive  # noqa: E402
from modelwatch.experiments.benchmark import run_benchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--n-events", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", type=float, default=0.01, help="target incident-level false-positive rate")
    parser.add_argument("--max-consecutive", type=int, default=10)
    args = parser.parse_args()

    print(f"Running {args.n_trials} synthetic trials to measure rag_adapter_full's single-check FPR...")
    results = run_benchmark(n_trials=args.n_trials, n_events=args.n_events, seed=args.seed)
    single_check_fpr = results["rag_adapter_full"].false_positive_rate

    calibration = calibrate_degraded_after_consecutive(
        single_check_fpr=single_check_fpr,
        target_incident_fpr=args.target,
        max_consecutive=args.max_consecutive,
    )

    print(f"\nMeasured single-check false-positive rate: {single_check_fpr:.4f} "
          f"(over {args.n_trials} synthetic trials, seed={args.seed})")
    print(f"Target incident-level false-positive rate:  {args.target:.4f}")
    print(f"\nRecommended: MODELWATCH_DEGRADED_AFTER_CONSECUTIVE={calibration.recommended_consecutive}")
    print(f"  (estimated incident-level FPR at this threshold: {calibration.achieved_incident_fpr:.6f})")
    if calibration.capped:
        print(
            f"  NOTE: hit max_consecutive={args.max_consecutive} without reaching the target -- "
            "this detector's single-check FPR is high enough that hysteresis alone can't bring the "
            "incident-level rate down to your target; consider raising --max-consecutive (at the cost "
            "of a slower reaction time) or investigating why single-check FPR is high before adjusting "
            "hysteresis at all."
        )
    print(
        "\nThis is a recommendation based on a real measured rate and an independence assumption "
        "documented in modelwatch/core/calibration.py -- review it, then set the env var yourself."
    )


if __name__ == "__main__":
    main()
