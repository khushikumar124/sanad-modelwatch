"""Demonstrates ModelWatch's full self-healing story against Sanad's live
chatbot: clean baseline -> simulate drift by swapping Sanad's configured
Ollama model mid-run -> drift detected and an alert raised -> corrective
action (swap the model back) -> retrain resets the baseline and clears
the alert -> a follow-up check confirms recovery.

Requires Sanad's API, ModelWatch's API, and Ollama all running, plus a
second model genuinely different from Sanad's configured default -- e.g.
`ollama pull mistral:7b` or `ollama pull qwen2.5:0.5b`.

It must be a real second model. Deriving a "degraded" variant of the same
weights with `ollama create` does NOT work here, and the reason is worth
knowing: Sanad sends an explicit system message and explicit `options`
on every request, and those override a Modelfile's SYSTEM and PARAMETER
directives. Measured, such a variant moved the golden-set quality score
by only 0.584 -> 0.565, well inside noise. The script detects this and
tells you the swap wasn't distinguishable rather than reporting a fake
drift event.

Usage:
    python -m modelwatch.examples.simulate_drift_demo --drift-model qwen2.5:0.5b --limit 5
"""
from __future__ import annotations

import argparse
import sys

import requests

from modelwatch.examples.golden_set import GOLDEN_SET
from modelwatch.examples.sanad_golden_set_runner import MODEL_ID, run_once


def get_sanad_model(session: requests.Session, sanad_url: str) -> str:
    res = session.get(f"{sanad_url}/api/admin/model")
    res.raise_for_status()
    return res.json()["model"]


def set_sanad_model(session: requests.Session, sanad_url: str, model: str) -> None:
    res = session.post(f"{sanad_url}/api/admin/model", json={"model": model})
    res.raise_for_status()


def trigger_modelwatch_retrain(session: requests.Session, modelwatch_url: str, golden_set=GOLDEN_SET) -> dict:
    res = session.post(
        f"{modelwatch_url}/models/{MODEL_ID}/retrain",
        json={"new_training_data": [{"prompt": p["prompt"], "expected_answer": p["expected_answer"]} for p in golden_set]},
    )
    res.raise_for_status()
    return res.json()


def get_active_alerts(session: requests.Session, modelwatch_url: str) -> list[dict]:
    res = session.get(f"{modelwatch_url}/alerts", params={"model_id": MODEL_ID, "active_only": True})
    res.raise_for_status()
    return res.json()


def _print_check(label: str, result: dict) -> None:
    print(
        f"[{label}] drift_score={result['drift_score']:.3f} "
        f"quality_score={result['quality_score']} "
        f"is_drifted={result['is_drifted']} "
        f"alert_id={result.get('alert_id')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanad-url", default="http://localhost:8100")
    parser.add_argument("--modelwatch-url", default="http://localhost:8000")
    parser.add_argument(
        "--drift-model",
        required=True,
        help="a different Ollama model to swap to, simulating drift. This does not have to be "
        "a separate download -- `ollama create` can derive a behaviourally different variant "
        "from a model you already have (see the README).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only use the first N golden pairs. The demo makes 3 full passes, so on a local "
        "3B model a full run takes ~30 minutes; limit it when demoing live.",
    )
    args = parser.parse_args()

    golden_set = GOLDEN_SET[: args.limit] if args.limit else GOLDEN_SET
    session = requests.Session()

    print("== step 1: baseline run with the current (good) model ==")
    good_model = get_sanad_model(session, args.sanad_url)
    print(f"Sanad's active model: {good_model}")
    baseline_result = run_once(args.sanad_url, args.modelwatch_url, golden_set=golden_set, session=session)
    _print_check("baseline", baseline_result)
    if baseline_result["is_drifted"]:
        print(
            "warning: baseline run already shows drift -- the golden set may not suit this "
            "model, or the model itself is underperforming. Continuing anyway for the demo."
        )

    print(f"\n== step 2: simulate drift by swapping Sanad to '{args.drift_model}' ==")
    set_sanad_model(session, args.sanad_url, args.drift_model)
    drifted_result = run_once(args.sanad_url, args.modelwatch_url, golden_set=golden_set, session=session)
    _print_check("drifted", drifted_result)
    if not drifted_result["is_drifted"]:
        print(
            "note: the swapped model didn't trigger drift detection -- try a model that's "
            "more different in behavior/quality from the original for a clearer demo."
        )
        sys.exit(0)

    alerts = get_active_alerts(session, args.modelwatch_url)
    print(f"active alerts after drift: {len(alerts)}")

    print(f"\n== step 3: corrective action -- swap Sanad back to '{good_model}' ==")
    set_sanad_model(session, args.sanad_url, good_model)

    print("== step 4: trigger_retrain (resets baseline, bumps version, resolves alerts) ==")
    model_info = trigger_modelwatch_retrain(session, args.modelwatch_url, golden_set)
    print(f"model now at version {model_info['current_version']}")

    print("\n== step 5: follow-up run confirms recovery ==")
    recovered_result = run_once(args.sanad_url, args.modelwatch_url, golden_set=golden_set, session=session)
    _print_check("recovered", recovered_result)
    alerts_after = get_active_alerts(session, args.modelwatch_url)
    print(f"active alerts after recovery: {len(alerts_after)}")


if __name__ == "__main__":
    main()
