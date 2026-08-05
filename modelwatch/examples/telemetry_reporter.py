"""Polls Sanad's telemetry buffer and reports it to ModelWatch.

This is the piece that makes real usage visible. The golden-set runner
answers "are the answers still correct?" and can only ever run against a
fixed test set, because scoring correctness needs to know the correct
answer. This watches the signals that need no ground truth -- refusal
rate, citation rate, latency -- so every question a real user asks
produces a datapoint.

The dependency deliberately points one way. Sanad publishes events at
`/api/telemetry` and knows nothing about who reads them; this reporter
knows about both. Sanad runs perfectly well with nothing watching it.

    # collect a baseline from a period you consider normal, then watch
    python -m modelwatch.examples.telemetry_reporter --baseline
    python -m modelwatch.examples.telemetry_reporter --interval 20

Prints are intentional -- this is an operator script.
"""
from __future__ import annotations

import argparse
import logging
import time

import requests

logger = logging.getLogger(__name__)

MODEL_ID = "sanad-live"
MODEL_NAME = "Sanad Chatbot (live traffic)"


def fetch_events(session: requests.Session, sanad_url: str, drain: bool = True) -> list[dict]:
    res = session.get(f"{sanad_url}/api/telemetry", params={"drain": str(drain).lower()})
    res.raise_for_status()
    return res.json()["events"]


def is_registered(session: requests.Session, modelwatch_url: str) -> bool:
    return session.get(f"{modelwatch_url}/models/{MODEL_ID}").status_code == 200


def register(session: requests.Session, modelwatch_url: str, events: list[dict]) -> None:
    res = session.post(
        f"{modelwatch_url}/models",
        json={
            "model_id": MODEL_ID,
            "name": MODEL_NAME,
            "adapter_name": "live_telemetry",
            "baseline_data": events,
        },
    )
    res.raise_for_status()


def report(session: requests.Session, modelwatch_url: str, events: list[dict]) -> dict:
    res = session.post(f"{modelwatch_url}/models/{MODEL_ID}/check", json={"new_data": events})
    res.raise_for_status()
    return res.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanad-url", default="http://localhost:8100")
    parser.add_argument("--modelwatch-url", default="http://localhost:8000")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="register using the events currently buffered, treating them as normal operation",
    )
    parser.add_argument(
        "--interval", type=float, default=20.0, help="seconds between polls (default 20)"
    )
    parser.add_argument("--once", action="store_true", help="poll a single time and exit")
    parser.add_argument(
        "--min-batch",
        type=int,
        default=5,
        help="wait until this many events are buffered before reporting (default 5). Draining "
        "on every poll produced batches of one or two, which the adapter correctly refuses to "
        "judge, so drift could never fire.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    session = requests.Session()

    if args.baseline:
        events = fetch_events(session, args.sanad_url, drain=False)
        if not events:
            print("No telemetry buffered yet. Ask a few questions in Sanad first.")
            return
        if is_registered(session, args.modelwatch_url):
            print(f"'{MODEL_ID}' is already registered. Use the retrain endpoint to re-baseline.")
            return
        register(session, args.modelwatch_url, events)
        print(f"Registered '{MODEL_ID}' with a baseline of {len(events)} events.")
        return

    print(f"Polling {args.sanad_url} every {args.interval:g}s. Ctrl-C to stop.")
    while True:
        try:
            # Look without consuming: a batch of one or two events cannot
            # support a rate, so hold them until there are enough to judge.
            waiting = fetch_events(session, args.sanad_url, drain=False)
            if len(waiting) < args.min_batch and not args.once:
                print(f"· {len(waiting)}/{args.min_batch} questions buffered, waiting")
                time.sleep(args.interval)
                continue
            events = fetch_events(session, args.sanad_url, drain=True)
            if not events:
                print("· no new questions")
            elif not is_registered(session, args.modelwatch_url):
                register(session, args.modelwatch_url, events)
                print(f"registered baseline from {len(events)} events")
            else:
                r = report(session, args.modelwatch_url, events)
                flag = " DRIFT" if r["is_drifted"] else ""
                print(
                    f"reported {len(events)} event(s) · grounded "
                    f"{r['quality_score']:.0%} · drift {r['drift_score']:.2f}{flag}"
                )
        except requests.exceptions.ConnectionError as e:
            print(f"could not reach a service: {e}")
        except requests.exceptions.HTTPError as e:
            print(f"request failed: {e}")

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
