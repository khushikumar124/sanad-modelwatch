"""Compares two contracts' risk scans side by side.

Deliberately built on the same rule-based RiskReport the single-document
risk scan already produces (sanad/features/risk_flagger.py), not a new
LLM-based comparison: a rule either matched a document's clauses or it
didn't, so "does contract A have a risk contract B doesn't" is a
deterministic, testable set operation -- not something that needs a
model call, and inherits the risk flagger's own documented limitations
(rule-based, conservative, not legal advice) rather than adding new ones.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sanad.features.risk_flagger import RiskReport


@dataclass
class RuleComparison:
    rule_id: str
    label: str
    severity: str
    in_a: bool
    in_b: bool
    #: Why this difference (or overlap) actually matters -- who it
    #: affects and what it means that it's present in one document,
    #: the other, or both. Sourced directly from the same RiskRule
    #: metadata the single-document scan already shows, not a new
    #: judgement -- a real explanation of a real presence/absence fact,
    #: not a claim about the documents' overall similarity.
    affects: str
    impact: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "label": self.label,
            "severity": self.severity,
            "in_a": self.in_a,
            "in_b": self.in_b,
            "affects": self.affects,
            "impact": self.impact,
        }


@dataclass
class ComparisonResult:
    counts_a: dict[str, int]
    counts_b: dict[str, int]
    only_in_a: list[RuleComparison]
    only_in_b: list[RuleComparison]
    shared: list[RuleComparison]

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts_a": self.counts_a,
            "counts_b": self.counts_b,
            "only_in_a": [r.to_dict() for r in self.only_in_a],
            "only_in_b": [r.to_dict() for r in self.only_in_b],
            "shared": [r.to_dict() for r in self.shared],
        }


_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def compare_risk_reports(report_a: RiskReport, report_b: RiskReport) -> ComparisonResult:
    # First occurrence per rule_id: label/severity are fixed per rule, so
    # which specific clause matched doesn't matter for a presence
    # comparison -- only whether the rule fired at all in each document.
    by_rule_a = {f.rule_id: f for f in report_a.findings}
    by_rule_b = {f.rule_id: f for f in report_b.findings}

    all_rule_ids = sorted(
        set(by_rule_a) | set(by_rule_b),
        key=lambda rid: _SEVERITY_ORDER.get((by_rule_a.get(rid) or by_rule_b[rid]).severity, 9),
    )

    only_in_a, only_in_b, shared = [], [], []
    for rule_id in all_rule_ids:
        finding = by_rule_a.get(rule_id) or by_rule_b.get(rule_id)
        in_a, in_b = rule_id in by_rule_a, rule_id in by_rule_b
        if in_a and in_b:
            impact = ("Appears in both documents. Whether this is actually the same risk depends on the "
                      "clause wording -- amounts, durations, and triggers can still differ; open both "
                      "clauses to compare them directly.")
        elif in_a:
            impact = f"Present in Document A only, absent from Document B -- this specific risk to {finding.affects} does not apply under Document B's stated terms."
        else:
            impact = f"Present in Document B only, absent from Document A -- this specific risk to {finding.affects} does not apply under Document A's stated terms."
        comparison = RuleComparison(
            rule_id=rule_id,
            label=finding.label,
            severity=finding.severity,
            in_a=in_a,
            in_b=in_b,
            affects=finding.affects,
            impact=impact,
        )
        if comparison.in_a and comparison.in_b:
            shared.append(comparison)
        elif comparison.in_a:
            only_in_a.append(comparison)
        else:
            only_in_b.append(comparison)

    return ComparisonResult(
        counts_a=report_a.counts,
        counts_b=report_b.counts,
        only_in_a=only_in_a,
        only_in_b=only_in_b,
        shared=shared,
    )
