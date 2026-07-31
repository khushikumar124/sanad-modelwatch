"""ChromaDB-backed vector store for contract chunks.

All documents share one persistent Chroma collection; each chunk is
scoped by a doc_id metadata field so retrieval for one uploaded contract
never surfaces chunks from another.
"""
from __future__ import annotations

import logging

import chromadb

from sanad.config import config
from sanad.ingestion.chunking import Chunk
from sanad.rag.embeddings import Embedder

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(
        self,
        persist_path: str | None = None,
        collection_name: str | None = None,
        embedder: Embedder | None = None,
    ):
        self._client = chromadb.PersistentClient(path=persist_path or config.chroma_db_path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name or config.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = embedder or Embedder()

    def add_document(self, doc_id: str, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        ids = [f"{doc_id}::{c.index}" for c in chunks]
        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed(texts)
        metadatas = [
            {"doc_id": doc_id, "chunk_index": c.index, "heading": c.heading or ""} for c in chunks
        ]
        self._collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        logger.info("chunks indexed", extra={"doc_id": doc_id, "chunk_count": len(chunks)})

    def query(self, doc_id: str, query_text: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or config.retrieval_top_k
        query_embedding = self._embedder.embed_one(query_text)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"doc_id": doc_id},
        )
        hits = []
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []
        for text, metadata, distance in zip(documents, metadatas, distances):
            hits.append({"text": text, "metadata": metadata, "distance": distance})
        return hits

    def delete_document(self, doc_id: str) -> None:
        self._collection.delete(where={"doc_id": doc_id})
