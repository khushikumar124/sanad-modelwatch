"""Authentication tests.

Covers the pieces that are dangerous to get wrong: password verification,
session signature forgery, expiry, and that protected routes actually
refuse an unauthenticated caller. The app is re-imported with auth turned
on rather than mocked, so these exercise the real dependency wiring.
"""
import importlib
import os
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from sanad.api import auth

PASSWORD = "correct horse battery staple"


# -- password hashing -------------------------------------------------------


def test_hash_verifies_and_rejects():
    stored = auth.hash_password(PASSWORD)
    assert auth.verify_password(PASSWORD, stored)
    assert not auth.verify_password("wrong", stored)


def test_same_password_hashes_differently():
    """Distinct salts, so identical passwords don't produce identical
    hashes and a leaked file doesn't reveal shared passwords."""
    assert auth.hash_password(PASSWORD) != auth.hash_password(PASSWORD)


def test_malformed_stored_hash_is_rejected_not_crashed():
    for junk in ["", "nonsense", "pbkdf2_sha256$notanint$a$b", "md5$1$a$b"]:
        assert auth.verify_password(PASSWORD, junk) is False


# -- admin list ---------------------------------------------------------


@pytest.fixture
def admin_users(monkeypatch):
    from dataclasses import replace
    from sanad.config import config as base_config

    def _use(csv: str):
        monkeypatch.setattr(auth, "config", replace(base_config, admin_users=csv))
    return _use


def test_is_admin_true_for_a_listed_user(admin_users):
    admin_users("alice, bob")
    assert auth.is_admin("alice") is True
    assert auth.is_admin("bob") is True


def test_is_admin_false_for_an_unlisted_user(admin_users):
    admin_users("alice")
    assert auth.is_admin("mallory") is False


def test_is_admin_false_when_no_admins_configured(admin_users):
    admin_users("")
    assert auth.is_admin("anyone") is False


def test_is_admin_false_for_none_username(admin_users):
    admin_users("alice")
    assert auth.is_admin(None) is False


# -- session tokens ---------------------------------------------------------


@pytest.fixture
def secret(monkeypatch):
    """Config is a frozen dataclass, so pin the key by replacing the
    accessor rather than mutating config."""
    def _use(key: bytes):
        monkeypatch.setattr(auth, "_secret", lambda: key)
    _use(b"test-secret")
    return _use


def test_session_roundtrip(secret):
    assert auth.read_session(auth.create_session("alice")) == "alice"


def test_tampered_payload_is_rejected(secret):
    """The whole point of signing: editing the username must invalidate it."""
    body, signature = auth.create_session("alice").split(".")
    forged_body = auth._b64(b'{"sub":"admin","exp":9999999999}')
    assert auth.read_session(f"{forged_body}.{signature}") is None


def test_expired_session_is_rejected(secret):
    assert auth.read_session(auth.create_session("alice", ttl_seconds=-1)) is None


def test_session_signed_with_another_secret_is_rejected(secret):
    token = auth.create_session("alice")
    secret(b"a-different-secret")
    assert auth.read_session(token) is None


def test_garbage_token_is_rejected(secret):
    for junk in ["", "no-dot", "a.b.c", "...."]:
        assert auth.read_session(junk) is None


# -- protected routes, with auth actually enabled ---------------------------


@pytest.fixture(scope="module")
def authed_client():
    test_id = uuid.uuid4().hex
    os.environ["SANAD_AUTH_ENABLED"] = "true"
    os.environ["SANAD_SESSION_SECRET"] = "unit-test-secret"
    os.environ["SANAD_USERS"] = f"tester:{auth.hash_password(PASSWORD)}"
    os.environ["SANAD_CHROMA_DB_PATH"] = f"/tmp/sanad_auth_chroma_{test_id}"
    os.environ["SANAD_UPLOAD_DIR"] = f"/tmp/sanad_auth_uploads_{test_id}"

    import sanad.config
    import sanad.api.auth
    import sanad.api.app

    importlib.reload(sanad.config)
    importlib.reload(sanad.api.auth)
    app_module = importlib.reload(sanad.api.app)
    yield TestClient(app_module.app)

    for key in ("SANAD_AUTH_ENABLED", "SANAD_SESSION_SECRET", "SANAD_USERS"):
        os.environ.pop(key, None)
    importlib.reload(sanad.config)
    importlib.reload(sanad.api.auth)
    importlib.reload(sanad.api.app)


def test_protected_routes_reject_anonymous(authed_client):
    for method, path in [
        ("get", "/api/documents"),
        ("get", "/api/admin/model"),
        ("get", "/api/documents/whatever/risks"),
        ("post", "/api/documents/whatever/summarize"),
    ]:
        res = getattr(authed_client, method)(path)
        assert res.status_code == 401, f"{method} {path} was not protected"


def test_login_rejects_bad_credentials(authed_client):
    assert authed_client.post("/api/auth/login", json={"username": "tester", "password": "nope"}).status_code == 401
    assert authed_client.post("/api/auth/login", json={"username": "ghost", "password": PASSWORD}).status_code == 401


def test_login_then_access_then_logout(authed_client):
    res = authed_client.post("/api/auth/login", json={"username": "tester", "password": PASSWORD})
    assert res.status_code == 200
    assert res.json()["username"] == "tester"

    # cookie is now on the client, so protected routes open up
    assert authed_client.get("/api/documents").status_code == 200
    assert authed_client.get("/api/auth/session").json()["username"] == "tester"

    authed_client.post("/api/auth/logout")
    assert authed_client.get("/api/documents").status_code == 401


def test_session_cookie_is_httponly(authed_client):
    res = authed_client.post("/api/auth/login", json={"username": "tester", "password": PASSWORD})
    cookie_header = res.headers.get("set-cookie", "")
    assert "httponly" in cookie_header.lower(), "session cookie must not be readable from JavaScript"
    assert "samesite=lax" in cookie_header.lower().replace(" ", "")
    authed_client.post("/api/auth/logout")
