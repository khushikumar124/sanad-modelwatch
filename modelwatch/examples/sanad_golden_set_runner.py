"""Runs modelwatch/examples/golden_set.py through Sanad's live chatbot and
reports the results to ModelWatch, so drift/quality of Sanad's own chatbot
shows up on the ModelWatch dashboard.

Requires Sanad's API (default localhost:8100) and ModelWatch's API
(default localhost:8000) both running, plus Ollama serving the model
Sanad is configured to use.

Usage:
    python -m modelwatch.examples.sanad_golden_set_runner
    python -m modelwatch.examples.sanad_golden_set_runner --interval 3600   # re-run hourly

This is a one-off demo/integration script -- prints are intentional here,
unlike the library code elsewhere in this repo.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import requests

from modelwatch.examples.golden_set import GOLDEN_SET, GoldenPair

logger = logging.getLogger(__name__)

MODEL_ID = "sanad-chatbot"
MODEL_NAME = "Sanad RAG Chatbot"


def group_by_source_file(golden_set: list[GoldenPair]) -> dict[str, list[GoldenPair]]:
    grouped: dict[str, list[GoldenPair]] = {}
    for pair in golden_set:
        grouped.setdefault(pair["source_file"], []).append(pair)
    return grouped


def to_baseline_data(golden_set: list[GoldenPair]) -> list[dict]:
    """Shapes the golden set into the {prompt, expected_answer} pairs
    ModelWatch's LLMAdapter expects as a baseline."""
    return [{"prompt": p["prompt"], "expected_answer": p["expected_answer"]} for p in golden_set]


def upload_documents(session: requests.Session, sanad_url: str, golden_set: list[GoldenPair]) -> dict[str, str]:
    """Uploads each unique source contract to Sanad. Returns {source_file: doc_id}."""
    doc_ids: dict[str, str] = {}
    for source_file, pairs in group_by_source_file(golden_set).items():
        contract_type = pairs[0]["contract_type"]
        with open(source_file, "rb") as f:
            res = session.post(
                f"{sanad_url}/api/documents",
                files={"file": (Path(source_file).name, f, "application/pdf")},
                data={"contract_type": contract_type},
            )
        res.raise_for_status()
        doc_ids[source_file] = res.json()["doc_id"]
        logger.info("uploaded to sanad", extra={"source_file": source_file, "doc_id": doc_ids[source_file]})
    return doc_ids


def collect_actual_answers(
    session: requests.Session,
    sanad_url: str,
    golden_set: list[GoldenPair],
    doc_ids: dict[str, str],
) -> list[dict]:
    """Runs every golden prompt through Sanad's live chatbot, one call per pair."""
    results = []
    for pair in golden_set:
        doc_id = doc_ids[pair["source_file"]]
        res = session.post(f"{sanad_url}/api/documents/{doc_id}/chat", json={"question": pair["prompt"]})
        res.raise_for_status()
        results.append({"prompt": pair["prompt"], "actual_answer": res.json()["answer"]})
    return results


def ensure_model_registered(session: requests.Session, modelwatch_url: str, golden_set: list[GoldenPair]) -> None:
    res = session.get(f"{modelwatch_url}/models/{MODEL_ID}")
    if res.status_code == 200:
        return
    res = session.post(
        f"{modelwatch_url}/models",
        json={
            "model_id": MODEL_ID,
            "name": MODEL_NAME,
            "adapter_name": "llm",
            "baseline_data": to_baseline_data(golden_set),
        },
    )
    res.raise_for_status()
    logger.info("registered sanad-chatbot with modelwatch")


def run_check(session: requests.Session, modelwatch_url: str, actual_answers: list[dict]) -> dict:
    res = session.post(f"{modelwatch_url}/models/{MODEL_ID}/check", json={"new_data": actual_answers})
    res.raise_for_status()
    return res.json()


def run_once(
    sanad_url: str,
    modelwatch_url: str,
    golden_set: list[GoldenPair] = GOLDEN_SET,
    session: requests.Session | None = None,
) -> dict:
    session = session or requests.Session()
    doc_ids = upload_documents(session, sanad_url, golden_set)
    actual_answers = collect_actual_answers(session, sanad_url, golden_set, doc_ids)
    ensure_model_registered(session, modelwatch_url, golden_set)
    result = run_check(session, modelwatch_url, actual_answers)
    logger.info(
        "golden-set check complete",
        extra={
            "drift_score": result["drift_score"],
            "quality_score": result["quality_score"],
            "is_drifted": result["is_drifted"],
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanad-url", default="http://localhost:8100")
    parser.add_argument("--modelwatch-url", default="http://localhost:8000")
    parser.add_argument(
        "--interval", type=float, default=None, help="if set, re-run every N seconds instead of once"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    while True:
        try:
            result = run_once(args.sanad_url, args.modelwatch_url)
            print(
                f"drift_score={result['drift_score']:.3f} "
                f"quality_score={result['quality_score']} "
                f"is_drifted={result['is_drifted']} "
                f"alert_id={result.get('alert_id')}"
            )
        except requests.exceptions.ConnectionError as e:
            print(f"could not reach a service: {e}")

        if args.interval is None:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
