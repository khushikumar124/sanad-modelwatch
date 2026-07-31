"""Extraction tests: native-text PDFs go through PyMuPDF directly; pages
with no text layer fall back to Tesseract OCR. The OCR case is exercised
by rasterizing a real sample page to a flat image (stripping its text
layer) rather than relying on a scanned sample document.
"""
import fitz

from sanad.ingestion.extraction import extract_document

SAMPLE_DOC = "sanad/sample_docs/rental/rental_agreement_sample_1.pdf"


def test_native_text_pdf_does_not_use_ocr():
    doc = extract_document(SAMPLE_DOC)
    assert doc.used_ocr is False
    assert "RENTAL AGREEMENT" in doc.text


def test_scanned_image_falls_back_to_ocr(tmp_path):
    src = fitz.open(SAMPLE_DOC)
    pix = src[0].get_pixmap(dpi=300)
    image_path = tmp_path / "scanned_page.png"
    pix.save(str(image_path))
    src.close()

    doc = extract_document(str(image_path))

    assert doc.used_ocr is True
    assert "RENTAL AGREEMENT" in doc.text
