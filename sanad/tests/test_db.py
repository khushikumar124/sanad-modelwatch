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
    # version_group_id/version_number are backfilled at read time when
    # not explicitly set at save time (a standalone document is version
    # 1 of its own one-document chain) -- see db.py's _row_to_record.
    assert fetched == _record(version_group_id="doc-1", version_number=1)


def test_version_group_id_and_number_round_trip_when_explicitly_set(fresh_db):
    db.save_document(_record(doc_id="doc-2", version_group_id="doc-1", version_number=2))
    fetched = db.get_document("doc-2")
    assert fetched.version_group_id == "doc-1"
    assert fetched.version_number == 2


def test_list_version_group_returns_the_whole_chain_oldest_first(fresh_db):
    db.save_document(_record(doc_id="doc-1", version_group_id="doc-1", version_number=1))
    db.save_document(_record(doc_id="doc-2", version_group_id="doc-1", version_number=2))
    db.save_document(_record(doc_id="doc-3", version_group_id="doc-1", version_number=3))
    db.save_document(_record(doc_id="unrelated"))  # a different, standalone chain

    chain = db.list_version_group("doc-1")
    assert [r.doc_id for r in chain] == ["doc-1", "doc-2", "doc-3"]


def test_list_version_group_of_a_standalone_legacy_document_is_just_itself(fresh_db):
    """A row saved with no version columns at all (as every row was
    before this feature existed) must still resolve to a one-document
    chain of itself, via the version_group_id IS NULL fallback."""
    db.save_document(_record(doc_id="legacy-doc", version_group_id=None, version_number=None))
    chain = db.list_version_group("legacy-doc")
    assert [r.doc_id for r in chain] == ["legacy-doc"]


def test_list_version_group_skips_a_deleted_version(fresh_db):
    """version_group_id is just a stored string, not a foreign key that
    requires the referenced row to still exist -- deleting one version
    (even the root, whose own doc_id the others' version_group_id
    stores) must not break looking up the rest of the chain."""
    db.save_document(_record(doc_id="doc-1", version_group_id="doc-1", version_number=1))
    db.save_document(_record(doc_id="doc-2", version_group_id="doc-1", version_number=2))
    db.save_document(_record(doc_id="doc-3", version_group_id="doc-1", version_number=3))

    db.delete_document("doc-2")  # delete a middle version
    chain = db.list_version_group("doc-1")
    assert [r.doc_id for r in chain] == ["doc-1", "doc-3"]

    db.delete_document("doc-1")  # delete the root version itself
    chain = db.list_version_group("doc-1")  # group id string still resolves
    assert [r.doc_id for r in chain] == ["doc-3"]


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


def test_owner_round_trips(fresh_db):
    db.save_document(_record(owner="alice"))
    assert db.get_document("doc-1").owner == "alice"


def test_owner_defaults_to_none(fresh_db):
    db.save_document(_record())
    assert db.get_document("doc-1").owner is None


def test_migration_adds_owner_column_to_a_pre_existing_database(tmp_path, monkeypatch):
    """Simulates a real upgrade: a database created before the `owner`
    column existed (documents_table minus that column, saved and
    populated the old way), then db.py's normal startup path run against
    it. Must add the column without losing the row already there."""
    from sqlalchemy import Boolean, Column, Integer, MetaData, String, Table, Text, create_engine, insert

    db_path = tmp_path / "pre_owner_column.db"
    old_metadata = MetaData()
    old_documents_table = Table(
        "documents", old_metadata,
        Column("doc_id", String, primary_key=True),
        Column("filename", String, nullable=False),
        Column("contract_type", String, nullable=True),
        Column("chunk_count", Integer, nullable=False),
        Column("used_ocr", Boolean, nullable=False),
        Column("uploaded_at", String, nullable=False),
        Column("text", Text, nullable=False),
        Column("source_path", String, nullable=False),
    )
    old_engine = create_engine(f"sqlite:///{db_path}")
    old_metadata.create_all(old_engine)
    with old_engine.begin() as conn:
        conn.execute(insert(old_documents_table).values(
            doc_id="legacy-doc", filename="old.pdf", contract_type=None, chunk_count=3,
            used_ocr=False, uploaded_at="2025-01-01T00:00:00+00:00", text="old text",
            source_path="/tmp/old.pdf",
        ))
    old_engine.dispose()

    monkeypatch.setattr(db, "config", replace(base_config, database_url=f"sqlite:///{db_path}"))
    db.reset_engine()

    record = db.get_document("legacy-doc")
    assert record is not None
    assert record.owner is None  # migrated column, not backfilled
    assert record.filename == "old.pdf"  # pre-existing data survived

    # And the migrated database now behaves like a fresh one going forward.
    db.save_document(_record(doc_id="new-doc", owner="alice"))
    assert db.get_document("new-doc").owner == "alice"
    db.reset_engine()
