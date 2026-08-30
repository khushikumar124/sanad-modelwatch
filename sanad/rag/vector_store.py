"""ChromaDB-backed vector store for contract chunks.

All documents share one persistent Chroma collection; each chunk is
scoped by a doc_id metadata field so retrieval for one uploaded contract
never surfaces chunks from another.
"""
from __future__ import annotations

import logging

import chromadb
from rank_bm25 import BM25Okapi

from sanad.config import config
from sanad.ingestion.chunking import Chunk
from sanad.rag.embeddings import Embedder
from sanad.rag.hybrid_retrieval import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


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

    @property
    def embedder(self) -> Embedder:
        """Exposes the store's own Embedder so other callers (e.g. the
        RAG trace) reuse the already-loaded model instead of loading a
        second copy of it."""
        return self._embedder

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
        if config.retrieval_mode == "dense":
            return self._dense_query(doc_id, query_text, top_k)
        return self._hybrid_query(doc_id, query_text, top_k)

    def _dense_query(self, doc_id: str, query_text: str, top_k: int) -> list[dict]:
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

    def _hybrid_query(self, doc_id: str, query_text: str, top_k: int) -> list[dict]:
        """Combines dense (embedding) and sparse (BM25 lexical) retrieval
        via reciprocal rank fusion, so a query using exact contract
        terminology (e.g. "force majeure", a defined term, a clause
        number) that the embedding model doesn't weight highly can still
        surface the right clause via keyword match, without losing dense
        retrieval's ability to match paraphrases.

        Both rankings run over the *entire* corpus for this doc_id, not
        just its own top-k slice: contract documents chunk into at most a
        few dozen pieces here, so this stays cheap, and it means every
        candidate that could be fused in has a real cosine distance
        rather than a fabricated one.
        """
        corpus = self._collection.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
        ids = corpus["ids"]
        if not ids:
            return []
        chunk_indices = [m["chunk_index"] for m in corpus["metadatas"]]
        text_by_index = dict(zip(chunk_indices, corpus["documents"]))
        metadata_by_index = dict(zip(chunk_indices, corpus["metadatas"]))
        n = len(ids)

        dense_hits = self._dense_query(doc_id, query_text, top_k=n)
        dense_ranking = [h["metadata"]["chunk_index"] for h in dense_hits]
        distance_by_index = {h["metadata"]["chunk_index"]: h["distance"] for h in dense_hits}

        bm25 = BM25Okapi([_tokenize(t) for t in corpus["documents"]])
        bm25_scores = bm25.get_scores(_tokenize(query_text))
        bm25_ranking = [
            idx for idx, _score in sorted(zip(chunk_indices, bm25_scores), key=lambda pair: pair[1], reverse=True)
        ]

        fused = reciprocal_rank_fusion([dense_ranking, bm25_ranking])
        top_indices = sorted(fused, key=fused.get, reverse=True)[:top_k]

        return [
            {
                "text": text_by_index[idx],
                "metadata": metadata_by_index[idx],
                # Falls back to the maximum cosine distance only for a
                # chunk BM25 surfaced that the dense ranking's own
                # n_results=n call somehow omitted -- shouldn't happen
                # since both run over the same corpus, but a hit must
                # never be missing the field trace.py reads.
                "distance": distance_by_index.get(idx, 2.0),
            }
            for idx in top_indices
        ]

    def delete_document(self, doc_id: str) -> None:
        self._collection.delete(where={"doc_id": doc_id})
