#!/usr/bin/env python3
"""Proves ModelWatch is model-agnostic: monitors a RAG pipeline that has
NEVER heard of Sanad, using only the public modelwatch-client SDK and
the same "rag" adapter Sanad's own integration uses.

Requires a running ModelWatch server (default http://localhost:8000,
override with MODELWATCH_URL) -- start one with `./run.sh` from the
repo root, or `uvicorn modelwatch.api.app:app --port 8000` on its own.
Does NOT require Sanad, Ollama, or any LLM API key: rag_pipeline.py's
retrieval and "generation" are both real but fully local (TF-IDF +
template answers), so this example runs anywhere modelwatch-client and
scikit-learn are installed.

Usage:
    pip install -e ./modelwatch-client scikit-learn
    python examples/independent_rag/run_example.py
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Falls back to the repo's own modelwatch-client source if it isn't
# pip-installed -- this example should run straight from a fresh clone.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "modelwatch-client"))

from modelwatch_client import ModelWatchClient, ModelWatchError  # noqa: E402
from rag_pipeline import TinyRAGPipeline  # noqa: E402

MODEL_ID = "independent-faq-rag"

# Baseline: everyday questions the FAQ actually answers -- normal
# operation. Reuses the FAQ's own question phrasing: this pipeline is
# lexical (TF-IDF), not semantic, so that's a real, honest constraint on
# what "normal" input looks like for it -- documented in README.md, not
# hidden.
BASELINE_QUESTIONS = [
    "What is your return policy?",
    "How long does shipping take?",
    "Do you ship internationally?",
    "How do I track my order?",
    "What payment methods are accepted?",
    "Can I change my order after placing it?",
    "Is there a warranty on products?",
    "How do I cancel a subscription?",
    "What is your return policy?",
    "How long does shipping take?",
]

# "Current" traffic: mostly normal, but with several genuinely
# off-topic questions mixed in -- a real, not-staged, refusal-rate
# increase that ModelWatch's RAGAdapter should be able to catch.
CURRENT_QUESTIONS = [
    "What is your return policy?",
    "How long does shipping take?",
    "What's the meaning of life?",
    "Can you recommend a good restaurant nearby?",
    "What's the weather like today?",
    "How do I cancel a subscription?",
    "Do you ship internationally?",
    "What is the capital of France?",
    "Is there a warranty on products?",
    "Tell me a joke.",
]


def event_from_answer(answer) -> dict:
    """The exact telemetry shape modelwatch/adapters/rag_adapter.py's
    RAGAdapter expects -- see its _extract() function. No
    question_embedding here: this toy pipeline uses TF-IDF, not dense
    embeddings, so the embedding_drift signal will honestly report
    insufficient_sample rather than a fabricated vector."""
    return {
        "grounded": answer.grounded,
        "citations": answer.citations,
        "citations_requested": answer.citations_requested,
        "retrieval_scores": answer.retrieval_scores,
        "generation_latency_ms": answer.generation_latency_ms,
    }


def main() -> None:
    base_url = os.environ.get("MODELWATCH_URL", "http://localhost:8000")
    client = ModelWatchClient(base_url=base_url)
    pipeline = TinyRAGPipeline()

    print(f"ModelWatch server: {base_url}")
    try:
        client.list_models()
    except ModelWatchError as e:
        print(f"Could not reach ModelWatch at {base_url}: {e}")
        print("Start it with `./run.sh` from the repo root, or run modelwatch's API standalone.")
        sys.exit(1)

    model_id = f"{MODEL_ID}-{uuid.uuid4().hex[:8]}"  # fresh id per run, so re-running is always safe

    print(f"\n1. Running baseline traffic ({len(BASELINE_QUESTIONS)} normal questions)...")
    baseline_events = [event_from_answer(pipeline.ask(q)) for q in BASELINE_QUESTIONS]
    for q, e in zip(BASELINE_QUESTIONS, baseline_events):
        print(f"   {'✓' if e['grounded'] else '✗'} {q}")

    print(f"\n2. Registering '{model_id}' with ModelWatch (adapter_name='rag')...")
    client.register_model(
        model_id=model_id,
        name="Independent FAQ RAG (no Sanad)",
        adapter_name="rag",
        baseline_data=baseline_events,
    )

    print(f"\n3. Running current traffic ({len(CURRENT_QUESTIONS)} questions, several off-topic)...")
    current_events = [event_from_answer(pipeline.ask(q)) for q in CURRENT_QUESTIONS]
    for q, e in zip(CURRENT_QUESTIONS, current_events):
        print(f"   {'✓' if e['grounded'] else '✗'} {q}")

    print("\n4. Checking for drift...")
    result = client.check(model_id, current_events)
    print(f"   is_drifted:   {result['is_drifted']}")
    print(f"   drift_score:  {result['drift_score']:.3f}")
    print(f"   quality_score:{result['quality_score']}")
    print(f"   health_state: {result['health_state']}")
    print("   signals:")
    for sig in result["signals"]:
        flag = "DRIFTED" if sig["is_drifted"] else "ok"
        print(f"     - {sig['name']:<20} value={sig['value']:.3f}  [{flag}]")

    print("\nDone -- this used only modelwatch_client's public API against a pipeline")
    print("that has never imported anything from sanad/. See README.md for what this proves.")


if __name__ == "__main__":
    main()
