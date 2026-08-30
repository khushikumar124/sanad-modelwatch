"""Finds duration-based contradictions among extracted obligations: two
obligations in the same category (typically notice or renewal) whose
deadlines state different numbers of days/weeks/months/years.

Deliberately narrow scope, not general contradiction detection: telling
whether two clauses are *semantically* inconsistent is an open NLU
problem well beyond what this heuristic attempts. What's actually
checked is much smaller and fully mechanical -- parse a duration
("30 days", "one month") out of each obligation's deadline text within
a category, normalize to days, and flag when a category has more than
one distinct value. That catches the concrete, common real-world case
this feature is aimed at (Clause 4 says 30-day notice, Clause 18 says
60-day notice) without claiming to catch anything subtler.

Input is sanad.features.obligations.ObligationsReport -- run
extract_obligations() first.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from sanad.features.obligations import Obligation

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_UNIT_DAYS = {"day": 1, "days": 1, "week": 7, "weeks": 7, "month": 30, "months": 30, "year": 365, "years": 365}

_DURATION_RE = re.compile(
    r"\b(\d+|" + "|".join(_NUMBER_WORDS) + r")\s*[-\s]?\s*(day|days|week|weeks|month|months|year|years)\b",
    re.IGNORECASE,
)

#: Categories where multiple distinct durations most plausibly indicate a
#: real drafting inconsistency rather than two legitimately different
#: obligations (e.g. "termination" often legitimately has different
#: notice lengths for cause vs. without cause, so it's excluded).
_CHECKED_CATEGORIES = {"notice", "renewal"}


def _parse_duration_days(text: str) -> int | None:
    match = _DURATION_RE.search(text or "")
    if not match:
        return None
    raw_number, unit = match.group(1).lower(), match.group(2).lower()
    number = _NUMBER_WORDS.get(raw_number, None)
    if number is None:
        try:
            number = int(raw_number)
        except ValueError:
            return None
    return number * _UNIT_DAYS[unit]


@dataclass
class Contradiction:
    category: str
    values_days: list[int]
    obligations: list[Obligation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "values_days": self.values_days,
            "obligations": [o.to_dict() for o in self.obligations],
        }


@dataclass
class ContradictionReport:
    contradictions: list[Contradiction] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"contradictions": [c.to_dict() for c in self.contradictions]}


def find_contradictions(obligations: list[Obligation]) -> ContradictionReport:
    by_category: dict[str, list[tuple[int, Obligation]]] = defaultdict(list)
    for o in obligations:
        if o.category not in _CHECKED_CATEGORIES or not o.grounded:
            continue
        days = _parse_duration_days(o.deadline or "") or _parse_duration_days(o.obligation)
        if days is not None:
            by_category[o.category].append((days, o))

    contradictions = []
    for category, entries in by_category.items():
        distinct_values = sorted({days for days, _ in entries})
        if len(distinct_values) > 1:
            contradictions.append(
                Contradiction(
                    category=category,
                    values_days=distinct_values,
                    obligations=[o for _, o in entries],
                )
            )
    return ContradictionReport(contradictions=contradictions)
