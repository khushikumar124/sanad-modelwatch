"""Checks a contract for standard clause categories that are commonly
expected but easy to overlook: termination, notice, confidentiality, IP
ownership, dispute resolution, liability, payment, renewal, governing law.

Rule-based, like risk_flagger.py, and for the same reason: "this
category's keywords don't appear anywhere in the document" is a
deterministic, testable claim. "This contract is missing a
confidentiality clause" is not -- a rule scan can only ever say a
pattern wasn't matched, never that a real lawyer reading the whole
document wouldn't find equivalent language phrased differently. Every
result is therefore reported as `found` or `not_found`, never
`missing`: `not_found` means honestly what it says, "this scan's
patterns didn't match anything," not a claim about what is or isn't in
the contract.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Pattern

from sanad.ingestion.chunking import Chunk

IMPORTANCE_HIGH = "high"
IMPORTANCE_MEDIUM = "medium"


def _p(*expressions: str) -> tuple[Pattern[str], ...]:
    return tuple(re.compile(e, re.IGNORECASE | re.DOTALL) for e in expressions)


@dataclass(frozen=True)
class CoverageCategory:
    category_id: str
    label: str
    importance: str  # "high" | "medium" -- how consequential a real gap here tends to be
    patterns: tuple[Pattern[str], ...]


COVERAGE_CATEGORIES: list[CoverageCategory] = [
    CoverageCategory("termination", "Termination", IMPORTANCE_HIGH, _p(
        r"terminat", r"end(s|ing)? this agreement",
    )),
    CoverageCategory("notice_period", "Notice period", IMPORTANCE_HIGH, _p(
        r"notice period", r"\bnotice\b[^.]{0,60}?(day|days|month|months|week|weeks)",
        r"(day|days|month|months|week|weeks)[^.]{0,40}?\bnotice\b",
    )),
    CoverageCategory("confidentiality", "Confidentiality", IMPORTANCE_MEDIUM, _p(
        r"confidential", r"non[-\s]?disclosure",
    )),
    CoverageCategory("ip_ownership", "IP ownership", IMPORTANCE_MEDIUM, _p(
        r"intellectual property", r"work for hire", r"\bcopyright\b",
    )),
    CoverageCategory("dispute_resolution", "Dispute resolution", IMPORTANCE_HIGH, _p(
        r"dispute", r"arbitrat", r"jurisdiction of",
    )),
    CoverageCategory("liability", "Liability", IMPORTANCE_MEDIUM, _p(
        r"liabilit", r"indemnif",
    )),
    CoverageCategory("payment", "Payment terms", IMPORTANCE_HIGH, _p(
        r"\brent\b", r"\bsalary\b", r"\bremuneration\b", r"\bfee(s)?\b[^.]{0,40}?payable",
        r"\bpaid\b[^.]{0,40}?(monthly|annually|per month|per annum)",
    )),
    CoverageCategory("renewal", "Renewal", IMPORTANCE_MEDIUM, _p(
        r"renew", r"extend(ed|s)? (this|the) (agreement|term|lease)",
    )),
    CoverageCategory("governing_law", "Governing law", IMPORTANCE_MEDIUM, _p(
        r"governing law", r"govern(ed)? by[^.]{0,40}?laws of", r"laws of india",
    )),
]


@dataclass
class CoverageResult:
    category_id: str
    label: str
    importance: str
    status: str  # "found" | "not_found"
    evidence_chunk_index: int | None
    evidence_preview: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "category_id": self.category_id,
            "label": self.label,
            "importance": self.importance,
            "status": self.status,
            "evidence_chunk_index": self.evidence_chunk_index,
            "evidence_preview": self.evidence_preview,
        }


@dataclass
class CoverageReport:
    results: list[CoverageResult] = field(default_factory=list)

    @property
    def not_found(self) -> list[CoverageResult]:
        return [r for r in self.results if r.status == "not_found"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "not_found_count": len(self.not_found),
        }


def check_coverage(chunks: list[Chunk], categories: list[CoverageCategory] | None = None) -> CoverageReport:
    categories = categories if categories is not None else COVERAGE_CATEGORIES
    results = []
    for category in categories:
        match_chunk, match_text = None, None
        for chunk in chunks:
            if any(p.search(chunk.text) for p in category.patterns):
                match_chunk, match_text = chunk.index, chunk.text
                break
        if match_chunk is not None:
            preview = " ".join(match_text.split())[:200]
            results.append(CoverageResult(category.category_id, category.label, category.importance, "found", match_chunk, preview))
        else:
            results.append(CoverageResult(category.category_id, category.label, category.importance, "not_found", None, None))
    return CoverageReport(results=results)
