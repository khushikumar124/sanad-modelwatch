"""Risk flagger tests.

Rule-based detection is deterministic, so it can be tested the way the
drift detectors are: construct a clause that should fire, construct one
that shouldn't, and assert the detector agrees. Real sample contracts are
used as the integration case, where the expected findings were confirmed
by reading the clauses.
"""
from sanad.features.risk_flagger import RISK_RULES, flag_risks
from sanad.ingestion.chunking import Chunk, chunk_document
from sanad.ingestion.extraction import extract_document

RENTAL_1 = "sanad/sample_docs/rental/rental_agreement_sample_1.pdf"
RENTAL_2 = "sanad/sample_docs/rental/rental_agreement_sample2.pdf"


def _chunk(text: str, index: int = 0, heading: str | None = None) -> Chunk:
    return Chunk(index=index, text=text, heading=heading)


# -- individual rules, controlled inputs -----------------------------------


def test_flags_penalty_multiplier():
    chunks = [_chunk("The Tenant will pay damages calculated at two times the rent for any period of occupation.")]
    findings = flag_risks(chunks).findings
    assert [f.rule_id for f in findings] == ["penalty_multiplier"]
    assert findings[0].severity == "high"


def test_flags_unilateral_termination():
    chunks = [_chunk("Notwithstanding any other provision, the Lessor shall have the right to terminate this agreement at any point of time during the lease.")]
    assert "unilateral_termination" in {f.rule_id for f in flag_risks(chunks).findings}


def test_flags_non_compete():
    chunks = [_chunk("During your employment and for 1 (one) year thereafter, you agree not to engage in any business which is directly or indirectly competing with the business of the company.")]
    assert "non_compete" in {f.rule_id for f in flag_risks(chunks).findings}


def test_flags_ip_assignment():
    chunks = [_chunk("The Deliverables shall be deemed to be 'work for hire' and all Intellectual Property Rights shall vest solely with the Company.")]
    assert "ip_assignment" in {f.rule_id for f in flag_risks(chunks).findings}


def test_benign_clause_is_not_flagged():
    """A clause with no unfavourable term must produce nothing. Guards
    against rules broad enough to fire on ordinary contract language."""
    chunks = [
        _chunk("This Agreement shall be interpreted in accordance with the substantive laws of the Republic of India."),
        _chunk("Each Party shall bear its own costs in connection with the negotiation of this Agreement."),
        _chunk("All notices under this Agreement shall be written in English and sent by hand or by courier."),
    ]
    assert flag_risks(chunks).findings == []


def test_one_finding_per_rule_per_clause():
    """A clause matching several phrasings of the same rule is one finding,
    not one per phrasing."""
    chunks = [_chunk("The Company may terminate without notice, and may terminate without any prior notice at any time.")]
    ids = [f.rule_id for f in flag_risks(chunks).findings]
    assert ids.count("termination_without_notice") == 1


def test_findings_sorted_by_severity():
    chunks = [
        _chunk("The Tenant shall not sublet the premises.", index=0),
        _chunk("The Lessor shall have the right to terminate this agreement at any point of time.", index=1),
    ]
    severities = [f.severity for f in flag_risks(chunks).findings]
    assert severities == sorted(severities, key=lambda s: {"high": 0, "medium": 1, "low": 2}[s])


def test_non_compete_factors_report_duration_and_geography_when_present():
    """The spec's own worked example: a non-compete clause's severity
    isn't one flat label -- whether it states a duration and a geographic
    scope are separate, checkable facts, each with its own quote."""
    chunks = [_chunk(
        "During your employment and for a period of 2 years thereafter, you agree not to engage "
        "in any business which is directly or indirectly competing with the business of the "
        "company within a radius of 50 kilometres of any company office."
    )]
    finding = next(f for f in flag_risks(chunks).findings if f.rule_id == "non_compete")
    by_key = {f.key: f for f in finding.factors}
    assert by_key["duration_specified"].present is True
    assert "2 years" in by_key["duration_specified"].quote
    assert by_key["geographic_scope_specified"].present is True
    assert by_key["consideration_mentioned"].present is False
    assert by_key["consideration_mentioned"].quote is None


def test_non_compete_factors_all_absent_on_bare_clause():
    """An unqualified non-compete clause -- no stated duration, scope, or
    consideration -- should show all three factors absent rather than
    silently omitting them; the absence is itself the useful signal."""
    chunks = [_chunk("You agree not to engage in any business which is directly or indirectly competing with the business of the company.")]
    finding = next(f for f in flag_risks(chunks).findings if f.rule_id == "non_compete")
    assert all(not f.present and f.quote is None for f in finding.factors)
    assert {f.key for f in finding.factors} == {"duration_specified", "geographic_scope_specified", "consideration_mentioned"}


def test_rules_without_factor_checks_produce_no_factors():
    """Most rules have no factor_checks defined -- their findings must
    carry an empty factors list, not a crash or a fabricated breakdown."""
    chunks = [_chunk("The Deliverables shall be deemed to be 'work for hire' and vest solely with the Company.")]
    finding = next(f for f in flag_risks(chunks).findings if f.rule_id == "ip_assignment")
    assert finding.factors == []


def test_finding_to_dict_includes_factors_key():
    chunks = [_chunk("You agree not to engage in any business which is directly or indirectly competing with the business of the company for 1 year.")]
    finding = next(f for f in flag_risks(chunks).findings if f.rule_id == "non_compete")
    d = finding.to_dict()
    assert "factors" in d
    assert d["factors"][0]["key"] == "duration_specified"


def test_every_rule_has_explanation_and_valid_severity():
    for rule in RISK_RULES:
        assert rule.severity in {"high", "medium", "low"}, rule.rule_id
        assert len(rule.explanation) > 40, rule.rule_id
        assert rule.patterns, rule.rule_id


# -- real documents ---------------------------------------------------------


def test_real_rental_contract_flags_known_clauses():
    """rental_agreement_sample_1 contains a 2x-rent overstay penalty and
    puts minor repairs on the tenant -- both confirmed by reading it."""
    report = flag_risks(chunk_document(extract_document(RENTAL_1).text))
    ids = {f.rule_id for f in report.findings}
    assert "penalty_multiplier" in ids
    assert "maintenance_on_weaker_party" in ids
    assert report.clauses_scanned > 5


def test_real_commercial_lease_flags_unilateral_termination():
    """rental_agreement_sample2 lets the Lessor terminate at any time."""
    report = flag_risks(chunk_document(extract_document(RENTAL_2).text))
    findings = [f for f in report.findings if f.rule_id == "unilateral_termination"]
    assert findings, "expected the at-will termination clause to be flagged"
    assert "terminate" in findings[0].excerpt.lower()


def test_report_counts_match_findings():
    report = flag_risks(chunk_document(extract_document(RENTAL_1).text))
    assert sum(report.counts.values()) == len(report.findings)
