"""End-to-end test of the shared ingestion pipeline (extraction ->
chunking -> embedding -> vector store) against a real sample contract,
verifying a topically relevant chunk is retrieved for a test query.
"""
import uuid

import pytest

from sanad.ingestion.chunking import chunk_document
from sanad.ingestion.extraction import extract_document
from sanad.rag.vector_store import VectorStore

SAMPLE_DOC = "sanad/sample_docs/freelance/freelance_agreement_sample2.pdf"


@pytest.fixture(scope="module")
def extracted_chunks():
    doc = extract_document(SAMPLE_DOC)
    return doc, chunk_document(doc.text)


def test_extraction_produces_text(extracted_chunks):
    doc, _ = extracted_chunks
    assert len(doc.text) > 500
    assert doc.used_ocr is False  # this sample is a native-text PDF


def test_chunking_splits_on_clause_structure(extracted_chunks):
    _, chunks = extracted_chunks
    assert len(chunks) > 5
    headings = [c.heading for c in chunks if c.heading]
    assert any("Payment" in h for h in headings)
    assert any("Intellectual Property" in h for h in headings)


def test_retrieval_finds_relevant_clause(extracted_chunks, tmp_path):
    _, chunks = extracted_chunks
    store = VectorStore(persist_path=str(tmp_path / "chroma"))
    doc_id = f"test-{uuid.uuid4()}"
    store.add_document(doc_id, chunks)

    hits = store.query(doc_id, "What are the payment terms for the consultant?", top_k=3)

    assert len(hits) == 3
    assert any("Payment" in h["metadata"]["heading"] for h in hits)


def test_retrieval_is_scoped_per_document(extracted_chunks, tmp_path):
    """A query against one doc_id must never surface another document's chunks."""
    _, chunks = extracted_chunks
    store = VectorStore(persist_path=str(tmp_path / "chroma"))
    store.add_document("doc-a", chunks[:3])
    store.add_document("doc-b", chunks[3:6])

    hits = store.query("doc-a", "payment terms", top_k=10)

    assert len(hits) <= 3
    assert all(h["metadata"]["doc_id"] == "doc-a" for h in hits)
