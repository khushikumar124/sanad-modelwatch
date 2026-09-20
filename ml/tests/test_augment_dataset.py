"""Tests augment_dataset.py's resume logic against a temp file -- not the
real LLM paraphrase call, which needs a live Ollama and is exercised for
real by actually running the script (see ml/README.md)."""
from __future__ import annotations

import json

import ml.augment_dataset as augment_dataset


def test_existing_group_counts_is_empty_when_no_file_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(augment_dataset, "OUT_PATH", tmp_path / "does_not_exist.jsonl")
    assert augment_dataset._existing_group_counts() == {}


def test_existing_group_counts_tallies_per_group(tmp_path, monkeypatch):
    path = tmp_path / "partial.jsonl"
    rows = [
        {"text": "a", "group_id": "doc.pdf:1", "is_synthetic": True},
        {"text": "b", "group_id": "doc.pdf:1", "is_synthetic": True},
        {"text": "c", "group_id": "doc.pdf:5", "is_synthetic": True},
    ]
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    monkeypatch.setattr(augment_dataset, "OUT_PATH", path)

    counts = augment_dataset._existing_group_counts()

    assert counts == {"doc.pdf:1": 2, "doc.pdf:5": 1}


def test_group_id_combines_source_document_and_chunk_index():
    row = {"source_document": "sanad/sample_docs/x.pdf", "chunk_index": 7}
    assert augment_dataset._group_id(row) == "sanad/sample_docs/x.pdf:7"
