"""Tests for sanad/features/document_quality.py against controlled
ExtractedDocument/PageExtraction instances, the same pattern as the risk
flagger tests -- deterministic logic, tested against constructed input."""
from sanad.features.document_quality import (
    LOW_TEXT_CHAR_THRESHOLD,
    DocumentQualityReport,
    PageQuality,
    assess_document_quality,
)
from sanad.ingestion.extraction import ExtractedDocument, PageExtraction


def _doc(pages: list[PageExtraction]) -> ExtractedDocument:
    return ExtractedDocument(source_path="test.pdf", pages=pages)


def test_native_text_pages_are_not_flagged():
    doc = _doc([
        PageExtraction(page_number=1, text="A" * 500, used_ocr=False),
        PageExtraction(page_number=2, text="B" * 300, used_ocr=False),
    ])
    report = assess_document_quality(doc)
    assert report.used_ocr is False
    assert report.low_text_page_count == 0
    assert report.detailed_available is True
    assert "no quality concerns" in report.summary


def test_ocr_pages_are_counted():
    doc = _doc([
        PageExtraction(page_number=1, text="A" * 500, used_ocr=True),
        PageExtraction(page_number=2, text="B" * 500, used_ocr=False),
    ])
    report = assess_document_quality(doc)
    assert report.used_ocr is True
    assert report.ocr_page_count == 1
    assert report.total_pages == 2


def test_low_text_page_is_flagged():
    doc = _doc([PageExtraction(page_number=1, text="x" * (LOW_TEXT_CHAR_THRESHOLD - 1), used_ocr=True)])
    report = assess_document_quality(doc)
    assert report.low_text_page_count == 1
    assert report.pages[0].flagged_low_text is True
    assert "worth a manual check" in report.summary


def test_page_at_threshold_is_not_flagged():
    doc = _doc([PageExtraction(page_number=1, text="x" * LOW_TEXT_CHAR_THRESHOLD, used_ocr=False)])
    report = assess_document_quality(doc)
    assert report.low_text_page_count == 0


def test_empty_document_reports_zero_pages():
    report = assess_document_quality(_doc([]))
    assert report.total_pages == 0
    assert report.summary == "No pages were extracted."


def test_to_dict_and_from_dict_round_trip():
    doc = _doc([
        PageExtraction(page_number=1, text="A" * 500, used_ocr=True),
        PageExtraction(page_number=2, text="x", used_ocr=True),
    ])
    report = assess_document_quality(doc)
    restored = DocumentQualityReport.from_dict(report.to_dict())
    assert restored.to_dict() == report.to_dict()


def test_undetailed_report_reports_unavailable_summary():
    report = DocumentQualityReport(detailed_available=False)
    assert report.detailed_available is False
    assert "isn't available" in report.summary
    assert report.total_pages == 0


def test_page_quality_to_dict_shape():
    page = PageQuality(page_number=3, used_ocr=True, char_count=12, flagged_low_text=True)
    assert page.to_dict() == {
        "page_number": 3, "used_ocr": True, "char_count": 12, "flagged_low_text": True,
    }
