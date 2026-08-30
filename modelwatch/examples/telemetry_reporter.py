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
import os
import time

import requests

logger = logging.getLogger(__name__)

MODEL_ID = "sanad-live"
MODEL_NAME = "Sanad Chatbot (live traffic)"


def login_if_needed(session: requests.Session, sanad_url: str) -> None:
    """If Sanad has auth on, /api/telemetry (and now, with full traces
    on, potentially real contract text) is behind require_user -- this
    reporter needs its own session cookie just like a browser would.
    Credentials come from env vars, never CLI args (a password on the
    command line lands in shell history and `ps`)."""
    username = os.environ.get("SANAD_REPORTER_USERNAME")
    password = os.environ.get("SANAD_REPORTER_PASSWORD")
    session_info = session.get(f"{sanad_url}/api/auth/session").json()
    if not session_info.get("auth_enabled"):
        return
    if not username or not password:
        raise SystemExit(
            "Sanad has auth enabled but SANAD_REPORTER_USERNAME/SANAD_REPORTER_PASSWORD "
            "are not set -- this reporter can't sign in to read /api/telemetry."
        )
    res = session.post(f"{sanad_url}/api/auth/login", json={"username": username, "password": password})
    res.raise_for_status()
    logger.info("signed in to sanad as %s", username)


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


def report_traces(session: requests.Session, modelwatch_url: str, events: list[dict]) -> int:
    """Forwards each event's full_trace (present only when Sanad has
    SANAD_TELEMETRY_FULL_TRACE on) to ModelWatch's RAG X-Ray store. Runs
    independently of model registration/drift-checking -- a trace is raw
    per-request detail, not a statistical signal, so there's no reason to
    gate it on whether the drift model exists yet."""
    reported = 0
    for event in events:
        trace = event.get("full_trace")
        if not trace:
            continue
        trace_id = trace.get("trace_id") or event.get("trace_id")
        try:
            res = session.post(
                f"{modelwatch_url}/traces",
                json={"trace_id": trace_id, "model_id": MODEL_ID, "data": trace},
            )
            res.raise_for_status()
            reported += 1
        except requests.exceptions.RequestException as e:
            logger.warning("failed to report trace %s: %s", trace_id, e)
    return reported


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
    login_if_needed(session, args.sanad_url)

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
    # Accumulated across polls until there's enough for a judgeable drift
    # batch. Traces are forwarded every poll regardless of this -- a
    # trace is one request's own detail, not a statistical sample, so
    # the RAG X-Ray would otherwise sit empty for --min-batch questions
    # for no reason connected to what it actually shows.
    pending: list[dict] = []
    while True:
        try:
            events = fetch_events(session, args.sanad_url, drain=True)
            if events:
                traced = report_traces(session, args.modelwatch_url, events)
                pending.extend(events)
                if traced:
                    print(f"· {traced} trace(s) sent to RAG X-Ray")

            if len(pending) >= args.min_batch or (args.once and pending):
                batch, pending = pending, []
                if not is_registered(session, args.modelwatch_url):
                    register(session, args.modelwatch_url, batch)
                    print(f"registered baseline from {len(batch)} events")
                else:
                    r = report(session, args.modelwatch_url, batch)
                    flag = " DRIFT" if r["is_drifted"] else ""
                    print(
                        f"reported {len(batch)} event(s) · grounded "
                        f"{r['quality_score']:.0%} · drift {r['drift_score']:.2f}{flag}"
                    )
            elif not events:
                print(f"· no new questions ({len(pending)}/{args.min_batch} buffered for drift check)")
            else:
                print(f"· {len(pending)}/{args.min_batch} questions buffered for drift check")
        except requests.exceptions.ConnectionError as e:
            print(f"could not reach a service: {e}")
        except requests.exceptions.HTTPError as e:
            print(f"request failed: {e}")

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
