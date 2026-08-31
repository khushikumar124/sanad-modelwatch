"""Synthesizes a "Review Contract" report from findings that are already
computed elsewhere: risk_flagger's rule-based findings, coverage.py's
missing-category scan, and contradictions.py's duration conflicts.

Deliberately makes no new LLM call and adds no new judgment: everything
here is a re-ranking/re-phrasing of findings another module already
produced deterministically. That keeps the "review" framing honest --
this surfaces and organizes real findings for a human to act on, it does
not generate a legal opinion. Every item traces back to one of the three
source reports; nothing is synthesized from whole cloth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sanad.features.contradictions import ContradictionReport
from sanad.features.coverage import CoverageReport
from sanad.features.risk_flagger import RiskReport

#: Templated prompts a reader might raise with the other party, keyed by
#: the risk_flagger rule_id that triggered them. Deliberately phrased as
#: questions to ask, not as a legal conclusion to assert.
_NEGOTIATION_QUESTIONS: dict[str, str] = {
    "unilateral_termination": "Can termination rights be made mutual, or a minimum notice period added for both sides?",
    "termination_without_notice": "What minimum notice period would apply if this clause were negotiated?",
    "penalty_multiplier": "What specifically triggers this penalty, and is the multiplier negotiable?",
    "unilateral_change_of_terms": "Can changes to terms require mutual written consent instead of unilateral notice?",
    "non_compete": "How broad is 'competing business' defined, and can the duration/scope be narrowed?",
    "deposit_forfeiture": "Under exactly which conditions is the deposit forfeited, and can that be tightened?",
    "ip_assignment": "Does this cover pre-existing work, and can that be carved out?",
    "lock_in_period": "Is there an exit option (e.g. paying a fee) before the lock-in period ends?",
    "auto_renewal": "Can the auto-renewal notice window be made longer or opt-in instead of opt-out?",
    "broad_indemnity": "Can indemnity be limited to losses caused by the indemnifying party's own fault?",
    "liquidated_damages": "Is the fixed sum proportionate to a realistic estimate of actual loss?",
}

_COVERAGE_CLARIFICATION_TEMPLATE = "This document doesn't appear to address {label} -- worth clarifying with the other party before signing."


@dataclass
class ReviewItem:
    source: str  # "risk" | "contradiction" | "coverage"
    severity: str  # "high" | "medium" | "low"
    title: str
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source, "severity": self.severity,
            "title": self.title, "detail": self.detail, "evidence": self.evidence,
        }


@dataclass
class ReviewReport:
    top_issues: list[ReviewItem] = field(default_factory=list)
    negotiable_clauses: list[ReviewItem] = field(default_factory=list)
    questions_to_ask: list[str] = field(default_factory=list)
    clarification_areas: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_issues": [i.to_dict() for i in self.top_issues],
            "negotiable_clauses": [i.to_dict() for i in self.negotiable_clauses],
            "questions_to_ask": self.questions_to_ask,
            "clarification_areas": self.clarification_areas,
        }


def build_review(risk_report: RiskReport, coverage_report: CoverageReport, contradiction_report: ContradictionReport) -> ReviewReport:
    top_issues: list[ReviewItem] = []
    negotiable: list[ReviewItem] = []
    questions: list[str] = []
    clarifications: list[str] = []

    for finding in risk_report.findings:
        item = ReviewItem(
            source="risk", severity=finding.severity, title=finding.label,
            detail=finding.explanation,
            evidence={"rule_id": finding.rule_id, "excerpt": finding.excerpt, "chunk_index": finding.chunk_index},
        )
        if finding.severity == "high":
            top_issues.append(item)
        if finding.severity in ("high", "medium"):
            negotiable.append(item)
        question = _NEGOTIATION_QUESTIONS.get(finding.rule_id)
        if question and question not in questions:
            questions.append(question)

    for contradiction in contradiction_report.contradictions:
        # "Potential conflict requiring review", not "contradiction" or
        # "conflict" stated as fact -- this heuristic only knows that two
        # clauses in the same category state different durations, not that
        # they're actually inconsistent (e.g. one could legitimately be an
        # exception case the heuristic doesn't parse out). Overclaiming a
        # real logical contradiction here is exactly the kind of confident
        # wrong statement the spec's grounding rules exist to prevent.
        values = ", ".join(f"{v} days" for v in contradiction.values_days)
        item = ReviewItem(
            source="contradiction", severity="high",
            title=f"Potential conflict requiring review: {contradiction.category} durations differ",
            detail=f"Found {len(contradiction.obligations)} clause(s) in '{contradiction.category}' stating "
                    f"different durations: {values}. This may be a genuine drafting inconsistency, or the "
                    "clauses may legitimately cover different situations -- worth checking which one governs.",
            evidence={"category": contradiction.category, "values_days": contradiction.values_days},
        )
        top_issues.append(item)

    for result in coverage_report.not_found:
        if result.importance == "high":
            top_issues.append(ReviewItem(
                source="coverage", severity="medium", title=f"No {result.label} clause found",
                detail=f"A scan for {result.label.lower()} language found nothing in this document. "
                        "That does not prove it's absent -- only that this scan didn't find it.",
                evidence={"category_id": result.category_id},
            ))
        clarifications.append(_COVERAGE_CLARIFICATION_TEMPLATE.format(label=result.label.lower()))

    severity_order = {"high": 0, "medium": 1, "low": 2}
    top_issues.sort(key=lambda i: severity_order.get(i.severity, 9))
    negotiable.sort(key=lambda i: severity_order.get(i.severity, 9))

    return ReviewReport(
        top_issues=top_issues, negotiable_clauses=negotiable,
        questions_to_ask=questions, clarification_areas=clarifications,
    )
