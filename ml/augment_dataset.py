"""Generates paraphrased variants of the 13 real flagged clauses using
the same local Ollama model Sanad's own features already depend on --
data augmentation, not new real documents. Every generated row is
explicitly marked `"is_synthetic": true`; nothing here is passed off as
real evidence, matching this repo's established labeling discipline for
anything not human/document-verified.

Why augment only the flagged (positive) clauses, not the 571 "none"
ones: the bottleneck identified in docs/ml_experiment.md is precision on
the *rare* class -- there's no shortage of "none" examples, only of
positive ones. Paraphrasing preserves the clause's legal meaning (the
same duration, the same restriction, the same party affected) in
different wording, so the model sees more of *how* a given risk gets
phrased without inventing a new risk category that was never in a real
document.

Critical methodological point, spelled out because getting it wrong
would produce a fake-looking improvement: a paraphrase of a clause is
NOT independent evidence of that risk pattern -- it's the same
underlying fact restated. If a real clause is held out for testing while
its own paraphrase sits in the training set, a model could "recognize"
near-duplicate phrasing and score well without having learned anything
that generalizes -- classic data leakage. Every synthetic row here
carries a `group_id` identical to its real origin clause's, and
ml/train_embeddings_grouped_cv.py evaluates with leave-one-GROUP-out
(not leave-one-example-out) specifically so a real clause and every one
of its own paraphrases are always held out or trained on *together*,
never split across the train/test boundary.

Run after ml/build_dataset.py:

    python -m ml.augment_dataset
"""
from __future__ import annotations

import json
from pathlib import Path

from sanad.rag.llm_client import LLMConnectionError, OllamaClient

from ml.train_risk_classifier import DATA_PATH

OUT_PATH = Path(__file__).parent / "data" / "clause_risk_dataset_v1_augmented.jsonl"
VARIANTS_PER_CLAUSE = 4

PARAPHRASE_SYSTEM_PROMPT = """You rewrite a single clause from an Indian legal contract (rental, employment, or freelance/service agreement) in different wording, while preserving its exact legal meaning -- the same obligation, the same duration, the same restriction, the same party affected. Do not add, remove, or change any concrete fact (numbers, durations, party roles). Do not add commentary. Respond with ONLY the rewritten clause text, nothing else."""

PARAPHRASE_SCHEMA = {
    "type": "object",
    "properties": {"rewritten_clause": {"type": "string"}},
    "required": ["rewritten_clause"],
}


def _group_id(row: dict) -> str:
    return f"{row['source_document']}:{row['chunk_index']}"


def load_flagged_rows() -> list[dict]:
    rows = []
    with DATA_PATH.open() as f:
        for line in f:
            row = json.loads(line)
            if row["label"] != "none":
                rows.append(row)
    return rows


def paraphrase(llm_client: OllamaClient, clause_text: str) -> str | None:
    raw = llm_client.generate(
        system_prompt=PARAPHRASE_SYSTEM_PROMPT,
        user_prompt=f"Clause:\n\n{clause_text}",
        response_schema=PARAPHRASE_SCHEMA,
        timeout=120,
    )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    rewritten = data.get("rewritten_clause")
    return rewritten.strip() if isinstance(rewritten, str) and rewritten.strip() else None


def run(variants_per_clause: int = VARIANTS_PER_CLAUSE) -> list[dict]:
    flagged = load_flagged_rows()
    llm_client = OllamaClient()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    augmented = []
    # Written one row at a time (flush + fsync after each), not
    # accumulated and written once at the end: this makes ~52
    # sequential local-LLM calls, and a crash partway through used to
    # mean losing every already-generated paraphrase along with it. A
    # rerun now resumes -- see _already_done below -- instead of
    # re-paying for variants that were already generated.
    already_done = _existing_group_counts()
    with OUT_PATH.open("a") as f:
        for row in flagged:
            group_id = _group_id(row)
            have = already_done.get(group_id, 0)
            if have >= variants_per_clause:
                print(f"Skipping {group_id}: already has {have}/{variants_per_clause} variants")
                continue
            print(f"Paraphrasing {group_id} (label={row['label']}, {have}/{variants_per_clause} so far)...")
            made = have
            attempts = 0
            while made < variants_per_clause and attempts < variants_per_clause + 3:
                attempts += 1
                try:
                    rewritten = paraphrase(llm_client, row["text"])
                except LLMConnectionError as e:
                    # A slow/timed-out Ollama call is a transient hiccup,
                    # not a reason to lose every already-written variant --
                    # retry this one attempt instead of crashing the run.
                    print(f"  attempt {attempts} failed ({e}), retrying...")
                    continue
                if rewritten is None or rewritten == row["text"]:
                    continue
                record = {
                    "text": rewritten,
                    "label": row["label"],
                    "source_document": row["source_document"],
                    "chunk_index": row["chunk_index"],
                    "group_id": group_id,
                    "is_synthetic": True,
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                augmented.append(record)
                made += 1
            if made < variants_per_clause:
                print(f"  only got {made}/{variants_per_clause} usable variants")

    return augmented


def _existing_group_counts() -> dict[str, int]:
    """How many synthetic variants each group already has in
    OUT_PATH, for resuming an interrupted run instead of redoing it."""
    if not OUT_PATH.exists():
        return {}
    counts: dict[str, int] = {}
    with OUT_PATH.open() as f:
        for line in f:
            row = json.loads(line)
            counts[row["group_id"]] = counts.get(row["group_id"], 0) + 1
    return counts


if __name__ == "__main__":
    augmented = run()
    print(f"\nGenerated {len(augmented)} synthetic variants from {len(load_flagged_rows())} real flagged clauses.")
    print(f"Saved -> {OUT_PATH}")
