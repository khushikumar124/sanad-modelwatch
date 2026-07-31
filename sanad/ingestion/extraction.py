"""Text extraction for uploaded contracts.

Text-based PDFs are read directly via PyMuPDF. Scanned PDFs (or individual
scanned pages within an otherwise text-based PDF) have no text layer, so
each such page falls back to Tesseract OCR. Plain image uploads (a single
photographed page) are OCR'd directly.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

import fitz
import pytesseract
from PIL import Image

from sanad.config import config

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}


@dataclass
class PageExtraction:
    page_number: int
    text: str
    used_ocr: bool


@dataclass
class ExtractedDocument:
    source_path: str
    pages: list[PageExtraction]

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text)

    @property
    def used_ocr(self) -> bool:
        return any(p.used_ocr for p in self.pages)


def _ocr_image(image: Image.Image) -> str:
    return pytesseract.image_to_string(image, lang=config.ocr_language).strip()


def _extract_pdf(path: str) -> list[PageExtraction]:
    pages: list[PageExtraction] = []
    doc = fitz.open(path)
    try:
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            used_ocr = False
            if not text:
                pix = page.get_pixmap(dpi=config.ocr_dpi)
                image = Image.open(io.BytesIO(pix.tobytes("png")))
                text = _ocr_image(image)
                used_ocr = True
                logger.info("OCR fallback used for page", extra={"path": path, "page": i + 1})
            pages.append(PageExtraction(page_number=i + 1, text=text, used_ocr=used_ocr))
    finally:
        doc.close()
    return pages


def _extract_image(path: str) -> list[PageExtraction]:
    image = Image.open(path)
    text = _ocr_image(image)
    return [PageExtraction(page_number=1, text=text, used_ocr=True)]


def extract_document(file_path: str) -> ExtractedDocument:
    """Extract text from a PDF or image file, using OCR only where a page
    has no native text layer."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        pages = _extract_pdf(file_path)
    elif ext in IMAGE_EXTENSIONS:
        pages = _extract_image(file_path)
    else:
        raise ValueError(f"unsupported file type '{ext}' for {file_path}")

    logger.info(
        "document extracted",
        extra={"path": file_path, "pages": len(pages), "chars": sum(len(p.text) for p in pages)},
    )
    return ExtractedDocument(source_path=file_path, pages=pages)
