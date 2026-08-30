"""Tests for sanad/features/comparison.py against controlled RiskReports
(constructed directly, not via a real contract) so the set logic is
exercised exactly."""
from sanad.features.comparison import compare_risk_reports
from sanad.features.risk_flagger import RiskFinding, RiskReport


def _finding(rule_id, severity="high", label=None):
    return RiskFinding(
        rule_id=rule_id,
        label=label or rule_id.replace("_", " "),
        severity=severity,
        explanation="explanation",
        affects="someone",
        clause_heading=None,
        excerpt="excerpt",
        chunk_index=0,
    )


def test_shared_rule_appears_in_shared_not_only_in_either():
    report_a = RiskReport(findings=[_finding("non_compete")], clauses_scanned=10)
    report_b = RiskReport(findings=[_finding("non_compete")], clauses_scanned=8)

    result = compare_risk_reports(report_a, report_b)

    assert [r.rule_id for r in result.shared] == ["non_compete"]
    assert result.only_in_a == []
    assert result.only_in_b == []


def test_rule_only_in_a_is_reported_correctly():
    report_a = RiskReport(findings=[_finding("deposit_forfeiture")], clauses_scanned=10)
    report_b = RiskReport(findings=[], clauses_scanned=8)

    result = compare_risk_reports(report_a, report_b)

    assert [r.rule_id for r in result.only_in_a] == ["deposit_forfeiture"]
    assert result.only_in_a[0].in_a is True
    assert result.only_in_a[0].in_b is False
    assert result.only_in_b == []
    assert result.shared == []


def test_rule_only_in_b_is_reported_correctly():
    report_a = RiskReport(findings=[], clauses_scanned=10)
    report_b = RiskReport(findings=[_finding("lock_in_period")], clauses_scanned=8)

    result = compare_risk_reports(report_a, report_b)

    assert [r.rule_id for r in result.only_in_b] == ["lock_in_period"]
    assert result.only_in_a == []


def test_counts_are_preserved_from_each_report():
    report_a = RiskReport(findings=[_finding("a", "high"), _finding("b", "medium")], clauses_scanned=5)
    report_b = RiskReport(findings=[_finding("c", "low")], clauses_scanned=5)

    result = compare_risk_reports(report_a, report_b)

    assert result.counts_a == {"high": 1, "medium": 1, "low": 0}
    assert result.counts_b == {"high": 0, "medium": 0, "low": 1}


def test_results_are_ordered_by_severity():
    report_a = RiskReport(
        findings=[_finding("low_risk", "low"), _finding("high_risk", "high"), _finding("med_risk", "medium")],
        clauses_scanned=5,
    )
    report_b = RiskReport(findings=[], clauses_scanned=5)

    result = compare_risk_reports(report_a, report_b)

    assert [r.rule_id for r in result.only_in_a] == ["high_risk", "med_risk", "low_risk"]


def test_multiple_findings_of_the_same_rule_count_as_one_comparison_row():
    """Two clauses tripping the same rule in report A shouldn't produce
    two rows -- presence is what's compared, not occurrence count."""
    report_a = RiskReport(
        findings=[_finding("non_compete"), RiskFinding(
            rule_id="non_compete", label="non compete", severity="high", explanation="e",
            affects="a", clause_heading=None, excerpt="e2", chunk_index=5,
        )],
        clauses_scanned=10,
    )
    report_b = RiskReport(findings=[], clauses_scanned=5)

    result = compare_risk_reports(report_a, report_b)

    assert len(result.only_in_a) == 1


def test_identical_empty_reports_produce_no_differences():
    report_a = RiskReport(findings=[], clauses_scanned=5)
    report_b = RiskReport(findings=[], clauses_scanned=5)

    result = compare_risk_reports(report_a, report_b)

    assert result.only_in_a == result.only_in_b == result.shared == []


def test_to_dict_is_json_serializable():
    import json

    report_a = RiskReport(findings=[_finding("non_compete")], clauses_scanned=10)
    report_b = RiskReport(findings=[_finding("lock_in_period")], clauses_scanned=8)
    json.dumps(compare_risk_reports(report_a, report_b).to_dict())
