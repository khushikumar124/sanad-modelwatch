"""Document-ownership / RBAC tests: with auth on, one user must not be
able to see or act on another user's uploaded documents, an admin must
be able to see everyone's, and a document uploaded before this feature
existed (owner=NULL) must stay visible to any authenticated user.

Real end-to-end HTTP tests against a real running app (module reloaded
with auth actually enabled, same pattern as test_auth.py's authed_client)
with two independent TestClient cookie jars for two real logged-in
users, not mocked authorization checks.
"""
import importlib
import os
import uuid

import pytest
from fastapi.testclient import TestClient

from sanad.api import auth

ALICE_PASSWORD = "alice-password-123"
BOB_PASSWORD = "bob-password-456"
ADMIN_PASSWORD = "admin-password-789"

RENTAL_DOC = "sanad/sample_docs/rental/rental_agreement_sample_1.pdf"


@pytest.fixture(scope="module")
def multi_user_app():
    """Three real users: alice, bob, and an admin (listed in
    SANAD_ADMIN_USERS). Returns (app_module,) so tests can build their
    own TestClient per user -- one shared cookie jar would log everyone
    in as the same person."""
    test_id = uuid.uuid4().hex
    os.environ["SANAD_AUTH_ENABLED"] = "true"
    os.environ["SANAD_SESSION_SECRET"] = "unit-test-secret"
    os.environ["SANAD_USERS"] = (
        f"alice:{auth.hash_password(ALICE_PASSWORD)},"
        f"bob:{auth.hash_password(BOB_PASSWORD)},"
        f"admin:{auth.hash_password(ADMIN_PASSWORD)}"
    )
    os.environ["SANAD_ADMIN_USERS"] = "admin"
    os.environ["SANAD_CHROMA_DB_PATH"] = f"/tmp/sanad_authz_chroma_{test_id}"
    os.environ["SANAD_UPLOAD_DIR"] = f"/tmp/sanad_authz_uploads_{test_id}"
    os.environ["SANAD_DATABASE_URL"] = f"sqlite:////tmp/sanad_authz_documents_{test_id}.db"

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
        "alice": _logged_in_client(multi_user_app, "alice", ALICE_PASSWORD),
        "bob": _logged_in_client(multi_user_app, "bob", BOB_PASSWORD),
        "admin": _logged_in_client(multi_user_app, "admin", ADMIN_PASSWORD),
    }


def _upload_as(client: TestClient) -> str:
    with open(RENTAL_DOC, "rb") as f:
        res = client.post("/api/documents", files={"file": ("rental.pdf", f, "application/pdf")})
    assert res.status_code == 201, res.text
    return res.json()["doc_id"]


def test_owner_can_see_their_own_document(clients):
    doc_id = _upload_as(clients["alice"])
    assert clients["alice"].get(f"/api/documents/{doc_id}").status_code == 200


def test_other_user_gets_404_not_403(clients):
    """404, not 403 -- a document belonging to someone else should not
    even confirm it exists to an unauthorized caller."""
    doc_id = _upload_as(clients["alice"])
    res = clients["bob"].get(f"/api/documents/{doc_id}")
    assert res.status_code == 404


def test_other_user_cannot_delete(clients):
    doc_id = _upload_as(clients["alice"])
    res = clients["bob"].delete(f"/api/documents/{doc_id}")
    assert res.status_code == 404
    assert clients["alice"].get(f"/api/documents/{doc_id}").status_code == 200  # still there


def test_other_user_cannot_chat(clients):
    doc_id = _upload_as(clients["alice"])
    res = clients["bob"].post(f"/api/documents/{doc_id}/chat", json={"question": "anything"})
    assert res.status_code == 404


def test_other_user_cannot_get_risks_or_clauses(clients):
    doc_id = _upload_as(clients["alice"])
    assert clients["bob"].get(f"/api/documents/{doc_id}/risks").status_code == 404
    assert clients["bob"].get(f"/api/documents/{doc_id}/clauses").status_code == 404
    assert clients["bob"].get(f"/api/documents/{doc_id}/coverage").status_code == 404


def test_list_documents_only_shows_the_caller_s_own(clients):
    alice_doc = _upload_as(clients["alice"])
    bob_doc = _upload_as(clients["bob"])

    alice_ids = {d["doc_id"] for d in clients["alice"].get("/api/documents").json()}
    bob_ids = {d["doc_id"] for d in clients["bob"].get("/api/documents").json()}

    assert alice_doc in alice_ids and alice_doc not in bob_ids
    assert bob_doc in bob_ids and bob_doc not in alice_ids


def test_admin_can_see_and_access_another_user_s_document(clients):
    doc_id = _upload_as(clients["alice"])
    assert clients["admin"].get(f"/api/documents/{doc_id}").status_code == 200
    admin_ids = {d["doc_id"] for d in clients["admin"].get("/api/documents").json()}
    assert doc_id in admin_ids


def test_cross_chat_requires_access_to_every_document(clients):
    alice_doc = _upload_as(clients["alice"])
    bob_doc = _upload_as(clients["bob"])
    res = clients["bob"].post(
        "/api/documents/cross-chat", json={"doc_ids": [alice_doc, bob_doc], "question": "compare"}
    )
    assert res.status_code == 404  # alice_doc isn't bob's to include


def test_compare_requires_access_to_both_documents(clients):
    alice_doc = _upload_as(clients["alice"])
    bob_doc = _upload_as(clients["bob"])
    assert clients["bob"].get(f"/api/documents/{bob_doc}/compare/{alice_doc}").status_code == 404


def test_legacy_document_with_no_owner_is_visible_to_any_authenticated_user(clients, multi_user_app):
    """A document saved before the owner column existed (or by any older
    client that didn't set it) has owner=NULL -- must stay reachable
    rather than becoming permanently inaccessible to everyone the moment
    this feature ships."""
    doc_id = _upload_as(clients["alice"])
    # Simulate "legacy" by clearing the owner directly in the registry --
    # this is exactly the state a pre-existing row is in after migration.
    record = multi_user_app.db.get_document(doc_id)
    from dataclasses import replace
    multi_user_app.db.save_document(replace(record, owner=None))

    assert clients["bob"].get(f"/api/documents/{doc_id}").status_code == 200
