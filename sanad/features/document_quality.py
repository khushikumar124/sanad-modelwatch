"""Assesses extraction quality of an uploaded document: how much of it
came from OCR, and whether any page looks like an extraction failure
rather than genuinely short content.

Deliberately built only from signals sanad/ingestion/extraction.py
already produces for real -- per page, whether OCR was used and how many
characters came out. No OCR confidence score is fabricated (Tesseract's
own per-word confidence isn't captured by the extraction pipeline today,
so this doesn't pretend to have it). A low-text page is flagged as worth
a manual check, not declared unreadable or broken: it could legitimately
be a short signature page or cover sheet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sanad.ingestion.extraction import ExtractedDocument

#: Pages with fewer characters than this are flagged as worth a manual
#: check. A cleanly extracted contract page runs into the hundreds of
#: characters; a page that comes back near-empty after OCR usually means
#: a failed or garbled pass -- though a real short page is also possible,
#: which is why this is a flag to look at, not a verdict.
LOW_TEXT_CHAR_THRESHOLD = 40


@dataclass
class PageQuality:
    page_number: int
    used_ocr: bool
    char_count: int
    flagged_low_text: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "used_ocr": self.used_ocr,
            "char_count": self.char_count,
            "flagged_low_text": self.flagged_low_text,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PageQuality":
        return cls(
            page_number=d["page_number"], used_ocr=d["used_ocr"],
            char_count=d["char_count"], flagged_low_text=d["flagged_low_text"],
        )


@dataclass
class DocumentQualityReport:
    pages: list[PageQuality] = field(default_factory=list)
    #: False only for documents uploaded before this feature existed --
    #: their DB row has no persisted per-page breakdown to reconstruct
    #: (extraction happens once, at upload time, and isn't re-run just to
    #: backfill this). Never fabricated from the joined document text.
    detailed_available: bool = True

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def ocr_page_count(self) -> int:
        return sum(1 for p in self.pages if p.used_ocr)

    @property
    def low_text_page_count(self) -> int:
        return sum(1 for p in self.pages if p.flagged_low_text)

    @property
    def used_ocr(self) -> bool:
        return self.ocr_page_count > 0

    @property
    def summary(self) -> str:
        if not self.detailed_available:
            return "Per-page quality detail isn't available for this document (uploaded before this feature existed)."
        if self.total_pages == 0:
            return "No pages were extracted."
        if self.low_text_page_count == 0 and not self.used_ocr:
            return "Extracted cleanly from native text; no quality concerns detected."
        parts = []
        if self.used_ocr:
            parts.append(f"{self.ocr_page_count} of {self.total_pages} page(s) required OCR (no native text layer)")
        if self.low_text_page_count:
            plural = "s" if self.low_text_page_count != 1 else ""
            parts.append(f"{self.low_text_page_count} page{plural} extracted unusually little text -- worth a manual check")
        return "; ".join(parts) + "."

    def to_dict(self) -> dict[str, Any]:
        return {
            "pages": [p.to_dict() for p in self.pages],
            "total_pages": self.total_pages,
            "ocr_page_count": self.ocr_page_count,
            "low_text_page_count": self.low_text_page_count,
            "used_ocr": self.used_ocr,
            "summary": self.summary,
            "detailed_available": self.detailed_available,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DocumentQualityReport":
        return cls(
            pages=[PageQuality.from_dict(p) for p in d.get("pages", [])],
            detailed_available=d.get("detailed_available", True),
        )


def assess_document_quality(document: ExtractedDocument) -> DocumentQualityReport:
    pages = [
        PageQuality(
            page_number=p.page_number,
            used_ocr=p.used_ocr,
            char_count=len(p.text),
            flagged_low_text=len(p.text) < LOW_TEXT_CHAR_THRESHOLD,
        )
        for p in document.pages
    ]
    return DocumentQualityReport(pages=pages)
