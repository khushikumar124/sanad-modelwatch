"""Clause/section-aware chunking for legal documents.

Legal contracts carry real structural signal -- numbered clauses, lettered
sub-clauses, ARTICLE/SECTION headers -- that a fixed-length sliding-window
splitter would ignore, cutting clauses in half and mixing unrelated
obligations into one chunk. This splits on detected clause/section
boundaries first, then merges runs that are too small and splits chunks
that are too large, so each chunk stays close to one coherent clause.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from sanad.config import config

# Matches the start of a line that looks like a clause/section boundary:
#   "1. ", "2.3 ", "(a) ", "(iv) ", "ARTICLE 3", "Section 4:", "Clause 5 -"
_BOUNDARY_PATTERNS = [
    re.compile(r"^\s*\d{1,2}(?:\.\d{1,2}){0,2}\.?\s+\S"),  # 1.  1.1  2.3.4
    re.compile(r"^\s*\([a-zA-Z0-9]{1,4}\)\s+\S"),  # (a)  (iv)  (1)
    re.compile(r"^\s*(ARTICLE|Article|SECTION|Section|Clause|CLAUSE)\s+[IVXLCM\d]+", re.IGNORECASE),
    re.compile(r"^[A-Z][A-Z \-&]{6,60}$"),  # ALL CAPS HEADER LINE
]


@dataclass
class Chunk:
    index: int
    text: str
    heading: str | None


def _is_boundary(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return any(p.match(stripped) for p in _BOUNDARY_PATTERNS)


def _split_on_boundaries(text: str) -> list[str]:
    lines = text.split("\n")
    raw_chunks: list[list[str]] = []
    for line in lines:
        if _is_boundary(line) or not raw_chunks:
            raw_chunks.append([line])
        else:
            raw_chunks[-1].append(line)
    return ["\n".join(lines).strip() for lines in raw_chunks if "\n".join(lines).strip()]


def _split_oversized(text: str, max_chars: int) -> list[str]:
    """Fallback sub-split for a chunk that's still too large: break on
    sentence boundaries, greedily packing sentences up to max_chars."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            pieces.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current.strip():
        pieces.append(current.strip())
    return pieces or [text]


def _merge_undersized(raw_chunks: list[str], min_chars: int, max_chars: int) -> list[str]:
    merged: list[str] = []
    buffer = ""
    for chunk in raw_chunks:
        candidate = f"{buffer}\n\n{chunk}".strip() if buffer else chunk
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            if buffer:
                merged.append(buffer)
            buffer = chunk
        if len(buffer) >= min_chars:
            merged.append(buffer)
            buffer = ""
    if buffer:
        if merged and len(buffer) < min_chars:
            merged[-1] = f"{merged[-1]}\n\n{buffer}".strip()
        else:
            merged.append(buffer)
    return merged


def _extract_heading(chunk_text: str) -> str | None:
    first_line = chunk_text.split("\n", 1)[0].strip()
    return first_line if _is_boundary(first_line) else None


def chunk_document(text: str, max_chars: int | None = None, min_chars: int | None = None) -> list[Chunk]:
    max_chars = max_chars or config.chunk_max_chars
    min_chars = min_chars or config.chunk_min_chars

    raw_chunks = _split_on_boundaries(text)
    if len(raw_chunks) <= 1:
        # No structural signal detected (e.g. an unusual layout) -- fall
        # back to sentence-packed fixed-size chunking rather than one
        # giant chunk.
        raw_chunks = _split_oversized(text, max_chars)
    else:
        sized: list[str] = []
        for c in raw_chunks:
            if len(c) > max_chars:
                sized.extend(_split_oversized(c, max_chars))
            else:
                sized.append(c)
        raw_chunks = sized

    merged = _merge_undersized(raw_chunks, min_chars, max_chars)

    return [
        Chunk(index=i, text=c, heading=_extract_heading(c))
        for i, c in enumerate(merged)
    ]
