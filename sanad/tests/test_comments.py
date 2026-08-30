"""Tests for db.py's comment storage and the /comments API, including
real end-to-end multi-user authorization checks (reusing the same
reload-the-app-with-auth-on pattern as test_authorization.py)."""
import importlib
import os
import uuid
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from sanad import db
from sanad.api import auth
from sanad.config import config as base_config

RENTAL_DOC = "sanad/sample_docs/rental/rental_agreement_sample_1.pdf"


# -- db.py storage, no auth involved -----------------------------------


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "config", replace(base_config, database_url=f"sqlite:///{tmp_path}/test.db"))
    db.reset_engine()
    yield
    db.reset_engine()


def _comment(comment_id="c1", **overrides):
    defaults = dict(
        comment_id=comment_id, doc_id="doc-1", chunk_index=0, author="alice",
        text="This clause seems unfair.", created_at="2026-01-01T00:00:00+00:00",
    )
    defaults.update(overrides)
    return db.Comment(**defaults)


def test_add_and_list_comments(fresh_db):
    db.add_comment(_comment())
    comments = db.list_comments("doc-1")
    assert len(comments) == 1
    assert comments[0].text == "This clause seems unfair."


def test_list_comments_scoped_to_doc_id(fresh_db):
    db.add_comment(_comment(comment_id="c1", doc_id="doc-1"))
    db.add_comment(_comment(comment_id="c2", doc_id="doc-2"))
    assert [c.comment_id for c in db.list_comments("doc-1")] == ["c1"]


def test_list_comments_ordered_by_created_at(fresh_db):
    db.add_comment(_comment(comment_id="later", created_at="2026-02-01T00:00:00+00:00"))
    db.add_comment(_comment(comment_id="earlier", created_at="2026-01-01T00:00:00+00:00"))
    assert [c.comment_id for c in db.list_comments("doc-1")] == ["earlier", "later"]


def test_get_comment_returns_none_for_unknown_id(fresh_db):
    assert db.get_comment("does-not-exist") is None


def test_delete_comment_removes_it(fresh_db):
    db.add_comment(_comment())
    db.delete_comment("c1")
    assert db.list_comments("doc-1") == []


def test_deleting_a_document_deletes_its_comments(fresh_db):
    db.save_document(db.DocumentRecord(
        doc_id="doc-1", filename="x.pdf", contract_type=None, chunk_count=1, used_ocr=False,
        uploaded_at="2026-01-01T00:00:00+00:00", text="text", source_path="/tmp/x.pdf",
    ))
    db.add_comment(_comment())
    db.delete_document("doc-1")
    assert db.list_comments("doc-1") == []


# -- API, real end-to-end multi-user authorization -----------------------


@pytest.fixture(scope="module")
def multi_user_app():
    test_id = uuid.uuid4().hex
    os.environ["SANAD_AUTH_ENABLED"] = "true"
    os.environ["SANAD_SESSION_SECRET"] = "unit-test-secret"
    os.environ["SANAD_USERS"] = (
        f"alice:{auth.hash_password('alice-pw')},"
        f"bob:{auth.hash_password('bob-pw')},"
        f"admin:{auth.hash_password('admin-pw')}"
    )
    os.environ["SANAD_ADMIN_USERS"] = "admin"
    os.environ["SANAD_CHROMA_DB_PATH"] = f"/tmp/sanad_comments_chroma_{test_id}"
    os.environ["SANAD_UPLOAD_DIR"] = f"/tmp/sanad_comments_uploads_{test_id}"
    os.environ["SANAD_DATABASE_URL"] = f"sqlite:////tmp/sanad_comments_documents_{test_id}.db"

    import sanad.config
    import sanad.api.auth
    import sanad.db
    import sanad.api.app

    importlib.reload(sanad.config)
    importlib.reload(sanad.api.auth)
    importlib.reload(sanad.db)
    app_module = importlib.reload(sanad.api.app)
    yield app_module

    for key in ("SANAD_AUTH_ENABLED", "SANAD_SESSION_SECRET", "SANAD_USERS", "SANAD_ADMIN_USERS",
                "SANAD_CHROMA_DB_PATH", "SANAD_UPLOAD_DIR", "SANAD_DATABASE_URL"):
        os.environ.pop(key, None)
    importlib.reload(sanad.config)
    importlib.reload(sanad.api.auth)
    importlib.reload(sanad.db)
    importlib.reload(sanad.api.app)


def _logged_in_client(app_module, username: str, password: str) -> TestClient:
    client = TestClient(app_module.app)
    res = client.post("/api/auth/login", json={"username": username, "password": password})
    assert res.status_code == 200, res.text
    return client


@pytest.fixture
def clients(multi_user_app):
    return {
        "alice": _logged_in_client(multi_user_app, "alice", "alice-pw"),
        "bob": _logged_in_client(multi_user_app, "bob", "bob-pw"),
        "admin": _logged_in_client(multi_user_app, "admin", "admin-pw"),
    }


def _upload_as(client: TestClient) -> str:
    with open(RENTAL_DOC, "rb") as f:
        res = client.post("/api/documents", files={"file": ("rental.pdf", f, "application/pdf")})
    assert res.status_code == 201, res.text
    return res.json()["doc_id"]


def test_owner_can_comment_on_their_own_document(clients):
    doc_id = _upload_as(clients["alice"])
    res = clients["alice"].post(f"/api/documents/{doc_id}/comments", json={"chunk_index": 0, "text": "note"})
    assert res.status_code == 201
    body = res.json()
    assert body["author"] == "alice"
    assert body["chunk_index"] == 0


def test_comment_rejects_an_out_of_range_chunk_index(clients):
    doc_id = _upload_as(clients["alice"])
    res = clients["alice"].post(f"/api/documents/{doc_id}/comments", json={"chunk_index": 99999, "text": "note"})
    assert res.status_code == 400


def test_comment_rejects_empty_text(clients):
    doc_id = _upload_as(clients["alice"])
    res = clients["alice"].post(f"/api/documents/{doc_id}/comments", json={"chunk_index": 0, "text": "   "})
    assert res.status_code == 400


def test_other_user_cannot_comment_on_someone_else_s_document(clients):
    doc_id = _upload_as(clients["alice"])
    res = clients["bob"].post(f"/api/documents/{doc_id}/comments", json={"chunk_index": 0, "text": "note"})
    assert res.status_code == 404


def test_other_user_cannot_list_comments(clients):
    doc_id = _upload_as(clients["alice"])
    clients["alice"].post(f"/api/documents/{doc_id}/comments", json={"chunk_index": 0, "text": "note"})
    res = clients["bob"].get(f"/api/documents/{doc_id}/comments")
    assert res.status_code == 404


def test_admin_can_comment_and_list_on_another_user_s_document(clients):
    doc_id = _upload_as(clients["alice"])
    res = clients["admin"].post(f"/api/documents/{doc_id}/comments", json={"chunk_index": 0, "text": "admin note"})
    assert res.status_code == 201
    listed = clients["admin"].get(f"/api/documents/{doc_id}/comments").json()["comments"]
    assert len(listed) == 1


def test_author_can_delete_their_own_comment(clients):
    doc_id = _upload_as(clients["alice"])
    comment_id = clients["alice"].post(
        f"/api/documents/{doc_id}/comments", json={"chunk_index": 0, "text": "note"}
    ).json()["comment_id"]
    res = clients["alice"].delete(f"/api/documents/{doc_id}/comments/{comment_id}")
    assert res.status_code == 204
    assert clients["alice"].get(f"/api/documents/{doc_id}/comments").json()["comments"] == []


def test_document_owner_can_delete_anyones_comment_on_their_document(clients):
    """Once admin has access to alice's document and comments on it,
    alice (the doc owner) can still moderate it."""
    doc_id = _upload_as(clients["alice"])
    comment_id = clients["admin"].post(
        f"/api/documents/{doc_id}/comments", json={"chunk_index": 0, "text": "admin note"}
    ).json()["comment_id"]
    res = clients["alice"].delete(f"/api/documents/{doc_id}/comments/{comment_id}")
    assert res.status_code == 204


def test_unrelated_admin_comment_cannot_be_deleted_by_a_non_owner_non_author(clients):
    doc_id = _upload_as(clients["alice"])
    bob_own_doc = _upload_as(clients["bob"])
    comment_id = clients["alice"].post(
        f"/api/documents/{doc_id}/comments", json={"chunk_index": 0, "text": "alice's note"}
    ).json()["comment_id"]
    # bob has no access to alice's document at all -> 404, not 403
    res = clients["bob"].delete(f"/api/documents/{doc_id}/comments/{comment_id}")
    assert res.status_code == 404


def test_deleting_a_document_removes_its_comments_via_the_api(clients):
    doc_id = _upload_as(clients["alice"])
    clients["alice"].post(f"/api/documents/{doc_id}/comments", json={"chunk_index": 0, "text": "note"})
    clients["alice"].delete(f"/api/documents/{doc_id}")
    # the document is gone, so even asking about its comments 404s
    assert clients["alice"].get(f"/api/documents/{doc_id}/comments").status_code == 404
