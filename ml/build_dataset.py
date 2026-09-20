"""Builds a labeled clause-level dataset for the risk-severity classifier
from every sample contract already in the repo.

Labels come from `sanad/features/risk_flagger.py`'s deterministic regex
rules, not human annotation -- this is a **silver-label** dataset, and
`docs/ml_experiment.md` says so up front. That has a real, specific
consequence for what the resulting classifier can and can't prove: it
tests whether a learned model can reproduce the *existing rules'*
decisions from clause text alone (useful as a fast pre-filter, or for
generalizing beyond wording the fixed regexes don't anticipate) -- it
does NOT independently validate that the rules themselves are correct,
since the labels and the thing being approximated are the same rules.

Run from the repository root:

    python -m ml.build_dataset
"""
from __future__ import annotations

import json
from pathlib import Path

from sanad.features.risk_flagger import flag_risks
from sanad.ingestion.chunking import chunk_document
from sanad.ingestion.extraction import extract_document

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DOCS_DIR = ROOT / "sanad" / "sample_docs"
OUT_PATH = Path(__file__).parent / "data" / "clause_risk_dataset_v1.jsonl"

SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1}


def _label_chunks(pdf_path: Path) -> list[dict]:
    """Extracts, chunks, and rule-flags one document; returns one labeled
    record per clause (chunk), using the highest-severity finding that
    touches that chunk, or "none" if the rule set didn't fire on it."""
    extracted = extract_document(str(pdf_path))
    chunks = chunk_document(extracted.text)
    report = flag_risks(chunks)

    label_by_index: dict[int, str] = {}
    for finding in report.findings:
        current = label_by_index.get(finding.chunk_index)
        if current is None or SEVERITY_RANK[finding.severity] > SEVERITY_RANK[current]:
            label_by_index[finding.chunk_index] = finding.severity

    records = []
    for chunk in chunks:
        text = chunk.text.strip()
        if not text:
            continue
        records.append(
            {
                "text": text,
                "label": label_by_index.get(chunk.index, "none"),
                "source_document": str(pdf_path.relative_to(ROOT)),
                "chunk_index": chunk.index,
            }
        )
    return records


def build() -> list[dict]:
    pdf_paths = sorted(SAMPLE_DOCS_DIR.rglob("*.pdf"))
    if not pdf_paths:
        raise RuntimeError(f"no sample PDFs found under {SAMPLE_DOCS_DIR}")

    all_records: list[dict] = []
    for pdf_path in pdf_paths:
        try:
            records = _label_chunks(pdf_path)
        except Exception as e:  # noqa: BLE001 -- one bad/scanned sample shouldn't kill the build
            print(f"  skipped {pdf_path.relative_to(ROOT)}: {e}")
            continue
        print(f"  {pdf_path.relative_to(ROOT)}: {len(records)} clauses")
        all_records.extend(records)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    return all_records


if __name__ == "__main__":
    print(f"Extracting + chunking + rule-flagging every PDF under {SAMPLE_DOCS_DIR}...")
    records = build()
    counts: dict[str, int] = {}
    for r in records:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    print(f"\nWrote {len(records)} labeled clauses to {OUT_PATH}")
    print("Label distribution:", dict(sorted(counts.items())))
