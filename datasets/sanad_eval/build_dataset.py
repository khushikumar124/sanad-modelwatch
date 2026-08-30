"""Builds datasets/sanad_eval/sanad_eval_v1.jsonl from the existing
GOLDEN_SET (modelwatch/examples/golden_set.py).

GOLDEN_SET already has real questions with expected_answers hand-verified
against real sample contracts. What it lacks (because it was built for
LLMAdapter, which never looks at retrieval) is a *retrieval* ground
truth: which chunk(s) of the document actually support the answer.

This script derives that automatically: it re-chunks each source
document exactly as the live pipeline does (sanad.ingestion.chunking,
current config), then finds the chunk(s) with the highest word overlap
against expected_answer. That is a heuristic, not a human annotation --
it can be wrong for an answer that paraphrases heavily instead of
echoing contract vocabulary. It's clearly labelled DEMO/SYNTHETIC ground
truth for exactly that reason (see dataset.py's docstring).

Run this whenever GOLDEN_SET changes:
    python -m datasets.sanad_eval.build_dataset
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modelwatch.examples.golden_set import GOLDEN_SET, GoldenPair  # noqa: E402
from sanad.evaluation.dataset import EvalCase, save_dataset  # noqa: E402
from sanad.ingestion.chunking import chunk_document  # noqa: E402
from sanad.ingestion.extraction import extract_document  # noqa: E402

OUT_PATH = Path(__file__).parent / "sanad_eval_v1.jsonl"

_STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "to", "and", "or", "in", "on", "for",
    "by", "with", "this", "that", "be", "will", "shall", "at", "as", "it",
    "not", "no", "any", "may", "can", "does", "do", "who", "what", "which",
}

# Simple keyword -> category rules, checked in order against the lowercased
# question text. Deterministic and reproducible, unlike hand-labeling --
# but a crude proxy for real categorization, so treat "category" as a demo
# convenience for slicing results, not a validated taxonomy.
_CATEGORY_RULES: list[tuple[str, str]] = [
    ("probation", "probation"),
    ("deposit", "security_deposit"),
    ("intellectual property", "intellectual_property"),
    ("employee", "employment_status"),
    ("employer-employee", "employment_status"),
    ("dispute", "dispute_resolution"),
    ("law govern", "governing_law"),
    ("jurisdiction", "governing_law"),
    ("notice", "notice_period"),
    ("terminat", "termination"),
    ("repair", "obligations"),
    ("vacate", "termination"),
    ("term of this lease", "duration"),
    ("other client", "working_restrictions"),
    ("remuneration", "compensation"),
    ("rent due", "compensation"),
    ("paid late", "compensation"),
    ("lease amount", "compensation"),
]


def _categorize(question: str) -> str:
    q = question.lower()
    for keyword, category in _CATEGORY_RULES:
        if keyword in q:
            return category
    return "general"


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _best_matching_chunks(expected_answer: str, chunks: list, top_n: int = 2, min_overlap: int = 2) -> list[int]:
    """Chunk indices with the most word overlap against expected_answer,
    keeping only chunks that clear a minimum absolute overlap (protects
    against returning a chunk that shares only stopword-adjacent noise)."""
    target_words = _tokenize(expected_answer)
    scored = []
    for chunk in chunks:
        overlap = len(target_words & _tokenize(chunk.text))
        if overlap >= min_overlap:
            scored.append((overlap, chunk.index))
    scored.sort(key=lambda x: -x[0])
    return [idx for _, idx in scored[:top_n]]


def build() -> list[EvalCase]:
    cases: list[EvalCase] = []
    doc_cache: dict[str, list] = {}
    counters: dict[str, int] = {}

    for pair in GOLDEN_SET:
        source_file = pair["source_file"]
        if source_file not in doc_cache:
            doc = extract_document(source_file)
            doc_cache[source_file] = chunk_document(doc.text)
        chunks = doc_cache[source_file]

        relevant_chunks = _best_matching_chunks(pair["expected_answer"], chunks)
        if not relevant_chunks:
            print(f"WARNING: no chunk matched for {pair['prompt']!r} in {source_file}", file=sys.stderr)

        stem = Path(source_file).stem
        counters[stem] = counters.get(stem, 0) + 1
        case_id = f"{stem}-{counters[stem]:02d}"

        cases.append(
            EvalCase(
                id=case_id,
                question=pair["prompt"],
                expected_answer=pair["expected_answer"],
                relevant_document=source_file,
                category=_categorize(pair["prompt"]),
                relevant_chunks=relevant_chunks,
                expected_citations=relevant_chunks,
            )
        )
    return cases


def main() -> None:
    cases = build()
    save_dataset(cases, OUT_PATH)
    print(f"wrote {len(cases)} cases to {OUT_PATH}")


if __name__ == "__main__":
    main()
