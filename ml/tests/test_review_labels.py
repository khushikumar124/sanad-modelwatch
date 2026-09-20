"""Tests review_labels.py's display logic against a small temp dataset --
this is a read-only reporting tool, so the only things worth testing are
that it doesn't crash, includes every flagged row, and respects the
requested sample size, not the real dataset's actual content."""
from __future__ import annotations

import json

import ml.review_labels as review_labels


def _write_dataset(tmp_path, rows):
    path = tmp_path / "dataset.jsonl"
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def _row(text, label, chunk_index):
    return {"text": text, "label": label, "source_document": "doc.pdf", "chunk_index": chunk_index}


def test_run_prints_every_flagged_row_and_respects_sample_size(tmp_path, monkeypatch, capsys):
    rows = (
        [_row(f"flagged clause {i}", "high", i) for i in range(3)]
        + [_row(f"none clause {i}", "none", 10 + i) for i in range(20)]
    )
    monkeypatch.setattr(review_labels, "DATA_PATH", _write_dataset(tmp_path, rows))

    review_labels.run(none_sample=5, seed=0)

    out = capsys.readouterr().out
    for i in range(3):
        assert f"flagged clause {i}" in out
    assert out.count("label=none") == 5
    assert "Total: 3 flagged, 20 none, 23 clauses." in out


def test_run_with_zero_none_sample_shows_only_flagged(tmp_path, monkeypatch, capsys):
    rows = [_row("the one flagged clause", "medium", 0), _row("an unflagged clause", "none", 1)]
    monkeypatch.setattr(review_labels, "DATA_PATH", _write_dataset(tmp_path, rows))

    review_labels.run(none_sample=0, seed=0)

    out = capsys.readouterr().out
    assert "the one flagged clause" in out
    assert "an unflagged clause" not in out


def test_run_raises_a_clear_error_when_dataset_missing(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setattr(review_labels, "DATA_PATH", tmp_path / "does_not_exist.jsonl")
    with pytest.raises(FileNotFoundError, match="build_dataset"):
        review_labels.run(none_sample=5, seed=0)
