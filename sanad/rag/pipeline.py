"""Shared ingestion entry point.

The summarizer and the chatbot both need the same extract -> chunk ->
index steps for an uploaded contract; this is the one place that logic
lives, so neither feature re-derives it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from sanad.ingestion.chunking import Chunk, chunk_document
from sanad.ingestion.extraction import extract_document
from sanad.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestedDocument:
    doc_id: str
    source_path: str
    text: str
    chunks: list[Chunk]
    used_ocr: bool


def ingest_document(file_path: str, doc_id: str, vector_store: VectorStore) -> IngestedDocument:
    """Extract, chunk, and index an uploaded contract under doc_id."""
    extracted = extract_document(file_path)
    chunks = chunk_document(extracted.text)
    vector_store.add_document(doc_id, chunks)
    logger.info("document ingested", extra={"doc_id": doc_id, "chunk_count": len(chunks)})
    return IngestedDocument(
        doc_id=doc_id,
        source_path=file_path,
        text=extracted.text,
        chunks=chunks,
        used_ocr=extracted.used_ocr,
    )
