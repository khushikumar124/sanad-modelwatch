"""Tests for sanad/features/review.py -- a pure synthesis of already-
computed findings, so these tests build controlled reports directly
rather than running the whole pipeline."""
from sanad.features.contradictions import Contradiction, ContradictionReport
from sanad.features.coverage import CoverageReport, CoverageResult
from sanad.features.review import build_review
from sanad.features.risk_flagger import RiskFinding, RiskReport


def _risk(rule_id, severity, label="label"):
    return RiskFinding(
        rule_id=rule_id, label=label, severity=severity, explanation="explanation",
        affects="someone", clause_heading=None, excerpt="excerpt", chunk_index=0,
    )


def test_high_severity_risk_is_a_top_issue_and_negotiable():
    risk_report = RiskReport(findings=[_risk("termination_without_notice", "high")], clauses_scanned=10)
    review = build_review(risk_report, CoverageReport(), ContradictionReport())

    assert len(review.top_issues) == 1
    assert review.top_issues[0].source == "risk"
    assert len(review.negotiable_clauses) == 1


def test_low_severity_risk_is_not_a_top_issue_but_may_be_negotiable():
    risk_report = RiskReport(findings=[_risk("assignment_restriction", "low")], clauses_scanned=10)
    review = build_review(risk_report, CoverageReport(), ContradictionReport())

    assert review.top_issues == []
    assert review.negotiable_clauses == []  # low severity is neither top issue nor negotiable


def test_medium_severity_risk_is_negotiable_but_not_top_issue():
    risk_report = RiskReport(findings=[_risk("ip_assignment", "medium")], clauses_scanned=10)
    review = build_review(risk_report, CoverageReport(), ContradictionReport())

    assert review.top_issues == []
    assert len(review.negotiable_clauses) == 1


def test_known_rule_id_generates_a_negotiation_question():
    risk_report = RiskReport(findings=[_risk("non_compete", "high")], clauses_scanned=10)
    review = build_review(risk_report, CoverageReport(), ContradictionReport())

    assert len(review.questions_to_ask) == 1
    assert "competing business" in review.questions_to_ask[0]


def test_contradiction_becomes_a_top_issue():
    contradiction_report = ContradictionReport(contradictions=[
        Contradiction(category="notice", values_days=[30, 60], obligations=[])
    ])
    review = build_review(RiskReport(), CoverageReport(), contradiction_report)

    assert len(review.top_issues) == 1
    assert review.top_issues[0].source == "contradiction"
    assert "30" in review.top_issues[0].detail and "60" in review.top_issues[0].detail


def test_high_importance_not_found_category_is_top_issue_and_clarification():
    coverage_report = CoverageReport(results=[
        CoverageResult("dispute_resolution", "Dispute resolution", "high", "not_found", None, None)
    ])
    review = build_review(RiskReport(), coverage_report, ContradictionReport())

    assert len(review.top_issues) == 1
    assert review.top_issues[0].source == "coverage"
    assert len(review.clarification_areas) == 1


def test_medium_importance_not_found_category_is_clarification_only():
    coverage_report = CoverageReport(results=[
        CoverageResult("liability", "Liability", "medium", "not_found", None, None)
    ])
    review = build_review(RiskReport(), coverage_report, ContradictionReport())

    assert review.top_issues == []
    assert len(review.clarification_areas) == 1


def test_found_category_produces_no_clarification():
    coverage_report = CoverageReport(results=[
        CoverageResult("payment", "Payment terms", "high", "found", 2, "preview text")
    ])
    review = build_review(RiskReport(), coverage_report, ContradictionReport())

    assert review.top_issues == []
    assert review.clarification_areas == []


def test_top_issues_are_sorted_by_severity():
    risk_report = RiskReport(
        findings=[_risk("ip_assignment", "medium", "medium-issue")],
        clauses_scanned=5,
    )
    contradiction_report = ContradictionReport(contradictions=[
        Contradiction(category="notice", values_days=[30, 60], obligations=[])
    ])
    review = build_review(risk_report, CoverageReport(), contradiction_report)
    # the "medium" risk finding never enters top_issues (only high does),
    # so the high-severity contradiction should be the only top issue
    assert len(review.top_issues) == 1
    assert review.top_issues[0].severity == "high"


def test_result_is_json_serializable():
    import json
    risk_report = RiskReport(findings=[_risk("non_compete", "high")], clauses_scanned=10)
    json.dumps(build_review(risk_report, CoverageReport(), ContradictionReport()).to_dict())
