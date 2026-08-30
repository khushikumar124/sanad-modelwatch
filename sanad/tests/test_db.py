"""Tests for sanad/db.py's document registry, against a real SQLite
file per test (tmp_path) -- exercises the real SQLAlchemy engine and
schema creation, not a mock.

sanad.config.config is a frozen dataclass, so tests swap the
module-level `config` name that db.py imported, via dataclasses.replace()
-- same pattern used for ModelWatch's config in other test files."""
from dataclasses import replace

import pytest

from sanad import db
from sanad.config import config as base_config


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Points db.py at a fresh SQLite file for this test only, and
    resets the cached engine so the next get_engine() call picks up the
    new URL instead of reusing whatever a previous test connected to."""
    monkeypatch.setattr(db, "config", replace(base_config, database_url=f"sqlite:///{tmp_path}/test.db"))
    db.reset_engine()
    yield
    db.reset_engine()


def _record(doc_id="doc-1", **overrides):
    defaults = dict(
        doc_id=doc_id, filename="lease.pdf", contract_type="rental", chunk_count=12,
        used_ocr=False, uploaded_at="2026-01-01T00:00:00+00:00", text="full contract text",
        source_path="/tmp/lease.pdf",
    )
    defaults.update(overrides)
    return db.DocumentRecord(**defaults)


def test_save_and_get_document_round_trips(fresh_db):
    db.save_document(_record())
    fetched = db.get_document("doc-1")
    assert fetched == _record()


def test_get_unknown_document_returns_none(fresh_db):
    assert db.get_document("does-not-exist") is None


def test_list_documents_returns_all_saved_ones(fresh_db):
    db.save_document(_record("doc-1", uploaded_at="2026-01-01T00:00:00+00:00"))
    db.save_document(_record("doc-2", uploaded_at="2026-01-02T00:00:00+00:00"))
    docs = db.list_documents()
    assert {d.doc_id for d in docs} == {"doc-1", "doc-2"}


def test_list_documents_orders_by_uploaded_at(fresh_db):
    db.save_document(_record("later", uploaded_at="2026-01-02T00:00:00+00:00"))
    db.save_document(_record("earlier", uploaded_at="2026-01-01T00:00:00+00:00"))
    docs = db.list_documents()
    assert [d.doc_id for d in docs] == ["earlier", "later"]


def test_delete_document_removes_it(fresh_db):
    db.save_document(_record())
    db.delete_document("doc-1")
    assert db.get_document("doc-1") is None


def test_delete_unknown_document_does_not_raise(fresh_db):
    db.delete_document("does-not-exist")  # must not raise


def test_save_document_overwrites_existing_row_with_same_id(fresh_db):
    db.save_document(_record(filename="v1.pdf"))
    db.save_document(_record(filename="v2.pdf"))
    docs = db.list_documents()
    assert len(docs) == 1
    assert docs[0].filename == "v2.pdf"


def test_to_response_excludes_full_text():
    """The response shape must never leak the full contract text over
    the wire for a document-list/get call -- only DocumentResponse's
    metadata fields."""
    response = _record().to_response()
    assert "text" not in response
    assert response["doc_id"] == "doc-1"
    assert response["chunk_count"] == 12


def test_contract_type_can_be_none(fresh_db):
    db.save_document(_record(contract_type=None))
    fetched = db.get_document("doc-1")
    assert fetched.contract_type is None


def test_engine_persists_across_calls_within_same_url(fresh_db):
    """Two independent calls against the same configured URL must see
    the same data -- i.e. the engine/connection isn't silently creating
    a fresh, empty database each time."""
    db.save_document(_record())
    assert db.get_document("doc-1") is not None
    assert db.get_document("doc-1") is not None
