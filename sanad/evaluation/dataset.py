"""Loading/saving the Sanad RAG evaluation dataset (JSONL).

Each case names a real question about a real sample contract, together
with a human-written expected_answer and a *retrieval ground truth*
(relevant_chunks / expected_citations): the chunk index(es) of that
contract's real chunked text which actually support the answer.

The ground truth in datasets/sanad_eval/sanad_eval_v1.jsonl is DEMO /
SYNTHETIC data, auto-derived (see build_dataset.py) by lexical-overlap
matching between each expected_answer and the document's real chunks --
not manually annotated by a human reader of the contract. That makes it
useful for exercising the evaluation engine and for regression-testing
"did retrieval/citation quality change", but it is not a substitute for
human-annotated ground truth if this is ever used to make a quality
claim in a paper. See docs/evaluation.md.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalCase:
    id: str
    question: str
    expected_answer: str
    relevant_document: str
    category: str
    #: chunk indices (from sanad.ingestion.chunking.chunk_document on the
    #: current config) that support expected_answer. Ground truth for
    #: retrieval-hit-rate / retrieval-coverage metrics.
    relevant_chunks: list[int] = field(default_factory=list)
    #: excerpt numbers (1-indexed, matching the chatbot's citation scheme)
    #: a correct grounded answer should cite. Currently identical to
    #: relevant_chunks re-expressed against a specific retrieval call's
    #: hit list -- kept as a separate field because a real annotator might
    #: legitimately consider a broader/narrower citation set correct than
    #: what "supports the answer" means for retrieval.
    expected_citations: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "relevant_document": self.relevant_document,
            "category": self.category,
            "relevant_chunks": self.relevant_chunks,
            "expected_citations": self.expected_citations,
        }

    @staticmethod
    def from_dict(d: dict) -> "EvalCase":
        return EvalCase(
            id=d["id"],
            question=d["question"],
            expected_answer=d["expected_answer"],
            relevant_document=d["relevant_document"],
            category=d["category"],
            relevant_chunks=list(d.get("relevant_chunks", [])),
            expected_citations=list(d.get("expected_citations", [])),
        )


def load_dataset(path: str | Path) -> list[EvalCase]:
    path = Path(path)
    cases = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(EvalCase.from_dict(json.loads(line)))
    return cases


def save_dataset(cases: list[EvalCase], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for case in cases:
            f.write(json.dumps(case.to_dict()) + "\n")
