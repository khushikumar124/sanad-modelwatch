"""Tests for VectorStore's dense and hybrid retrieval, against a real
ChromaDB instance and the real sentence-transformers embedder (already
required for the app to run at all -- there's no lighter-weight fake
that would actually exercise retrieval quality)."""
from dataclasses import replace

import pytest

from sanad.config import config as base_config
from sanad.ingestion.chunking import Chunk
from sanad.rag import vector_store as vector_store_module
from sanad.rag.vector_store import VectorStore

# One doc whose chunks are deliberately distinguishable both by exact
# vocabulary (for BM25) and by meaning (for embeddings), so a hybrid-mode
# test can tell the two retrieval paths apart.
_CHUNKS = [
    Chunk(index=0, text="The tenant shall pay a security deposit of INR 50000 before move-in.", heading="Deposit"),
    Chunk(index=1, text="Either party may terminate this agreement with 30 days written notice.", heading="Termination"),
    Chunk(index=2, text="This lease shall remain confidential and shall not be disclosed to third parties.", heading="Confidentiality"),
    Chunk(index=3, text="Force majeure events include flood, earthquake, and government-imposed lockdown.", heading="Force Majeure"),
]


@pytest.fixture
def store(tmp_path):
    return VectorStore(persist_path=str(tmp_path / "chroma"), collection_name="test_hybrid")


def test_dense_query_returns_hits_with_required_fields(store):
    store.add_document("doc-1", _CHUNKS)
    results = store._dense_query("doc-1", "how much is the security deposit", top_k=2)
    assert len(results) == 2
    assert all({"text", "metadata", "distance"} <= set(h) for h in results)
    assert results[0]["metadata"]["chunk_index"] == 0  # the deposit clause


def test_hybrid_query_finds_exact_defined_term_via_bm25(store):
    """'force majeure' is a specific legal term that a small embedding
    model can under-rank against more generic clauses; BM25's exact
    lexical match should pull it to the top regardless."""
    store.add_document("doc-1", _CHUNKS)
    results = store._hybrid_query("doc-1", "force majeure", top_k=1)
    assert results[0]["metadata"]["chunk_index"] == 3


def test_hybrid_query_every_hit_has_a_real_distance(store):
    store.add_document("doc-1", _CHUNKS)
    results = store._hybrid_query("doc-1", "termination notice period", top_k=4)
    assert len(results) == 4
    for hit in results:
        assert 0.0 <= hit["distance"] <= 2.0  # a real cosine distance, not the 2.0 fallback


def test_hybrid_query_scopes_to_the_requested_doc_id(store):
    store.add_document("doc-1", _CHUNKS)
    store.add_document("doc-2", [Chunk(index=0, text="An unrelated document about something else entirely.", heading=None)])
    results = store._hybrid_query("doc-1", "deposit", top_k=10)
    assert all(h["metadata"]["doc_id"] == "doc-1" for h in results)
    assert len(results) == len(_CHUNKS)


def test_hybrid_query_on_empty_document_returns_no_hits(store):
    assert store._hybrid_query("never-uploaded", "anything", top_k=5) == []


def test_public_query_dispatches_by_configured_retrieval_mode(store, monkeypatch):
    store.add_document("doc-1", _CHUNKS)

    monkeypatch.setattr(vector_store_module, "config", replace(base_config, retrieval_mode="dense"))
    dense_results = store.query("doc-1", "force majeure", top_k=1)

    monkeypatch.setattr(vector_store_module, "config", replace(base_config, retrieval_mode="hybrid"))
    hybrid_results = store.query("doc-1", "force majeure", top_k=1)

    # BM25 should win the "force majeure" query in hybrid mode even if
    # dense mode alone doesn't rank it first -- that's the whole point of
    # adding the sparse signal.
    assert hybrid_results[0]["metadata"]["chunk_index"] == 3
