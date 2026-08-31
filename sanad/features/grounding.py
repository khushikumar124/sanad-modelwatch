"""Shared grounding-check helpers: does a model-claimed quote actually
appear in the document, and if so, which clause?

Extracted from sanad/features/obligations.py (the original home of this
logic) so sanad/features/overview.py can reuse the exact same, already
real-data-tuned grounding check rather than re-implementing a second,
possibly-drifting copy of it. Behavior is unchanged from before the
extraction -- obligations.py's own tests (written against this logic
before the move) still pass against the relocated version unmodified.

Grounding uses an exact substring match first, falling back to a
content-word-overlap check (stopwords excluded) at an 0.85 threshold --
see _find_evidence_chunk's own docstring for the real rental-contract
case ("by local" vs "by the local") that made the fallback necessary.
"""
from __future__ import annotations

import re

from sanad.ingestion.chunking import Chunk


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "to", "and", "or", "in", "on", "for",
    "by", "with", "this", "that", "be", "will", "shall", "at", "as", "it",
    "not", "no", "any", "may", "can", "does", "do", "who", "what", "which",
    "under", "from", "such", "if", "would", "than", "then",
}

#: Content-word-overlap ratio above which a quote counts as grounded even
#: without an exact substring match. Real-data finding: a genuine,
#: correct extraction ("Owner shall pay for all taxes/cesses levied on
#: the premises by the local or government authorities...") failed an
#: exact substring check because the model's quote read "by local or
#: government authorities" (one word off from "by the local..."). Only
#: content words (stopwords excluded) count toward the ratio -- a short
#: fabricated quote built mostly of common words ("the tenant shall pay
#: a...") would otherwise score a deceptively high overlap purely from
#: words any contract sentence contains, without actually matching the
#: quote's substantive content.
OVERLAP_THRESHOLD = 0.85
MIN_CONTENT_WORDS = 3


def content_words(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1]


def _overlap_ratio(quote_words: list[str], chunk_words: set[str]) -> float:
    if not quote_words:
        return 0.0
    matched = sum(1 for w in quote_words if w in chunk_words)
    return matched / len(quote_words)


def find_evidence_chunk(source_quote: str, chunks: list[Chunk], normalized_doc: str) -> tuple[bool, int | None]:
    """Checks source_quote against each chunk individually (not just the
    whole document at once) so a grounded claim also records *which*
    clause supports it -- this is what lets the UI jump straight to the
    clause instead of only saying "somewhere in this document".

    Falls back to a whole-document check (chunk_index left None) for the
    rare quote that straddles a chunk boundary -- clause-aware chunking
    (sanad/ingestion/chunking.py) keeps this uncommon, but a boundary
    case should still be recognised as grounded, just not localized.
    """
    if not source_quote:
        return False, None
    normalized_quote = normalize(source_quote)

    for chunk in chunks:
        if normalized_quote in normalize(chunk.text):
            return True, chunk.index

    quote_words = content_words(source_quote)
    if len(quote_words) >= MIN_CONTENT_WORDS:
        best_ratio, best_index = 0.0, None
        for chunk in chunks:
            ratio = _overlap_ratio(quote_words, set(content_words(chunk.text)))
            if ratio > best_ratio:
                best_ratio, best_index = ratio, chunk.index
        if best_ratio >= OVERLAP_THRESHOLD:
            return True, best_index

    # Boundary-spanning fallback: not localizable to one chunk, but still
    # check whether it's grounded in the document as a whole.
    if normalized_quote in normalized_doc:
        return True, None
    if len(quote_words) >= MIN_CONTENT_WORDS:
        doc_ratio = _overlap_ratio(quote_words, set(content_words(normalized_doc)))
        if doc_ratio >= OVERLAP_THRESHOLD:
            return True, None
    return False, None
