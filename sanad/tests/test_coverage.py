"""Tests for sanad/features/coverage.py against both controlled chunks
and a real sample contract."""
from sanad.features.coverage import CoverageCategory, check_coverage
from sanad.ingestion.chunking import Chunk, chunk_document
from sanad.ingestion.extraction import extract_document


def _chunk(text, index=0):
    return Chunk(index=index, text=text, heading=None)


def test_category_present_is_found_with_evidence():
    chunks = [_chunk("The tenant shall pay a monthly rent of Rs. 10,000.")]
    report = check_coverage(chunks)
    payment = next(r for r in report.results if r.category_id == "payment")
    assert payment.status == "found"
    assert payment.evidence_chunk_index == 0
    assert "rent" in payment.evidence_preview.lower()


def test_category_absent_is_not_found_never_missing():
    chunks = [_chunk("This document only discusses the weather forecast.")]
    report = check_coverage(chunks)
    assert all(r.status == "not_found" for r in report.results)
    # deliberately checking the wording itself: "missing" is never used
    assert all(r.status != "missing" for r in report.results)


def test_not_found_list_matches_results():
    chunks = [_chunk("Nothing relevant here.")]
    report = check_coverage(chunks)
    assert len(report.not_found) == len(report.results)


def test_high_importance_categories_are_labeled_correctly():
    chunks = [_chunk("no relevant content")]
    report = check_coverage(chunks)
    termination = next(r for r in report.results if r.category_id == "termination")
    assert termination.importance == "high"


def test_custom_category_list_is_respected():
    custom = [CoverageCategory("custom", "Custom", "high", ())]
    chunks = [_chunk("anything")]
    report = check_coverage(chunks, categories=custom)
    assert len(report.results) == 1
    assert report.results[0].category_id == "custom"


def test_result_is_json_serializable():
    import json
    chunks = [_chunk("The tenant shall pay a monthly rent.")]
    json.dumps(check_coverage(chunks).to_dict())


def test_against_real_rental_contract_finds_payment_and_termination():
    """A real rental contract should have payment/termination/notice
    language somewhere -- this isn't testing the regex in isolation, it's
    testing the feature against real extracted text."""
    doc = extract_document("sanad/sample_docs/rental/rental_agreement_sample_1.pdf")
    chunks = chunk_document(doc.text)
    report = check_coverage(chunks)

    found_ids = {r.category_id for r in report.results if r.status == "found"}
    assert "payment" in found_ids
    assert "termination" in found_ids
    assert "notice_period" in found_ids
