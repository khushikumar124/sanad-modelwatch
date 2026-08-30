"""CLI: run the Sanad RAG evaluation dataset against the live Ollama model
and print/save real, measured metrics.

Requires Ollama serving the configured model (sanad.config.config.ollama_model)
-- there is no fake-LLM mode here, since the whole point is to measure the
actual model's actual answers. For fast, deterministic, no-Ollama testing
of the evaluation *engine* itself, see sanad/tests/test_evaluation.py,
which uses a FakeLLMClient.

Usage:
    python -m sanad.evaluation.run_eval
    python -m sanad.evaluation.run_eval --dataset datasets/sanad_eval/sanad_eval_v1.jsonl
    python -m sanad.evaluation.run_eval --out experiments/results/eval_run.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sanad.evaluation.dataset import load_dataset
from sanad.evaluation.runner import run_evaluation
from sanad.rag.llm_client import OllamaClient
from sanad.rag.vector_store import VectorStore

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "datasets" / "sanad_eval" / "sanad_eval_v1.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--out", default=None, help="write full JSON results here")
    parser.add_argument("--chroma-path", default=None, help="scratch chroma dir (default: tmp)")
    args = parser.parse_args()

    cases = load_dataset(args.dataset)
    print(f"Loaded {len(cases)} cases from {args.dataset}")

    chroma_path = args.chroma_path or f"/tmp/sanad_eval_chroma_{int(time.time())}"
    vector_store = VectorStore(persist_path=chroma_path)
    llm_client = OllamaClient()

    print(f"Running evaluation against {llm_client.model} ...")
    results, summary = run_evaluation(cases, vector_store, llm_client)

    print()
    print("RESULTS (deterministic + embedding-based, no fabricated numbers)")
    print(f"  cases:                    {summary.n_cases}")
    print(f"  retrieval hit rate:       {summary.retrieval_hit_rate:.1%}")
    print(f"  citation correctness:     {summary.citation_correctness:.1%}")
    print(f"  refusal rate:             {summary.refusal_rate:.1%}")
    print(f"  parse error rate:         {summary.parse_error_rate:.1%}")
    print(f"  mean semantic similarity: {summary.mean_semantic_similarity:.3f}")
    print(f"  latency p95:              {summary.latency_p95_ms:.0f}ms")
    print(f"    retrieval p95:          {summary.retrieval_latency_p95_ms:.0f}ms")
    print(f"    generation p95:         {summary.generation_latency_p95_ms:.0f}ms")
    print()
    print("  by category:")
    for category, stats in sorted(summary.by_category.items()):
        print(
            f"    {category:<22} n={stats['n_cases']:<3} "
            f"hit_rate={stats['retrieval_hit_rate']:.0%} "
            f"sim={stats['mean_semantic_similarity']:.3f}"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": llm_client.model,
            "dataset": args.dataset,
            "summary": summary.to_dict(),
            "cases": [r.to_dict() for r in results],
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote full results to {out_path}")


if __name__ == "__main__":
    main()
