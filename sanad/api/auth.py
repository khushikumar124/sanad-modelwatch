"""Local username/password authentication with signed session cookies.

Deliberately has no external identity provider. Sanad's whole position is
that a confidential contract never leaves the machine it was uploaded to,
and routing sign-in through Google would put an internet dependency in
front of an otherwise offline app. The seam for adding one later is
`_SESSION_SUBJECT`: a session records *who* you are, not *how* you proved
it, so an OAuth callback can mint the same session without touching route
protection or the frontend.

Design notes:

* Passwords are stored as PBKDF2-HMAC-SHA256 with a per-user random salt.
  scrypt or argon2 would be stronger, but PBKDF2 is in the standard
  library and this avoids adding a native dependency to a project that
  must install cleanly on a laptop.
* Sessions are signed with `itsdangerous`-style HMAC over a payload we
  generate, using a secret from the environment. No secret in the
  repository, and a missing secret in a non-local deployment is a hard
  error rather than a silent default.
* Auth is off by default (`SANAD_AUTH_ENABLED`) so the test suite and a
  local demo don't need credentials. That is a deliberate trade: it keeps
  the failure mode "no login screen" rather than "app is inaccessible
  because the operator hasn't set a password yet".
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass

from fastapi import HTTPException, Request

from sanad.config import config

logger = logging.getLogger(__name__)

_PBKDF2_ROUNDS = 240_000
_SESSION_SUBJECT = "sub"
COOKIE_NAME = "sanad_session"


# -- password hashing -------------------------------------------------------


def hash_password(password: str, salt: bytes | None = None) -> str:
    """Return a `pbkdf2_sha256$rounds$salt$hash` string."""
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ROUNDS)
    return "$".join(
        [
            "pbkdf2_sha256",
            str(_PBKDF2_ROUNDS),
            base64.b64encode(salt).decode(),
            base64.b64encode(digest).decode(),
        ]
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, rounds, salt_b64, digest_b64 = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        expected = base64.b64decode(digest_b64)
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), base64.b64decode(salt_b64), int(rounds)
        )
    except (ValueError, TypeError):
        return False
    # Constant-time: a length-or-content comparison that short-circuits
    # leaks how much of the hash matched.
    return hmac.compare_digest(expected, actual)


# -- session tokens ---------------------------------------------------------


def _secret() -> bytes:
    secret = config.session_secret
    if secret:
        return secret.encode()
    # Only reachable with auth disabled; a random per-process secret means
    # any stray cookie is simply invalid rather than trusted.
    return _EPHEMERAL_SECRET


_EPHEMERAL_SECRET = secrets.token_bytes(32)


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def create_session(username: str, ttl_seconds: int | None = None) -> str:
    ttl = ttl_seconds if ttl_seconds is not None else config.session_ttl_seconds
    payload = {_SESSION_SUBJECT: username, "exp": int(time.time()) + ttl}
    body = _b64(json.dumps(payload, separators=(",", ":")).encode())
    signature = _b64(hmac.new(_secret(), body.encode(), hashlib.sha256).digest())
    return f"{body}.{signature}"


def read_session(token: str) -> str | None:
    """Return the username in a valid, unexpired token, else None."""
    try:
        body, signature = token.split(".")
    except ValueError:
        return None
    expected = _b64(hmac.new(_secret(), body.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(expected, signature):
        return None
    try:
        payload = json.loads(_unb64(body))
    except (ValueError, TypeError):
        return None
    if payload.get("exp", 0) < time.time():
        return None
    subject = payload.get(_SESSION_SUBJECT)
    return subject if isinstance(subject, str) else None


# -- users ------------------------------------------------------------------


@dataclass(frozen=True)
class User:
    username: str
    password_hash: str


def _load_users() -> dict[str, User]:
    """Users come from the environment, not a database.

    This is a single-operator personal app; a users table would imply a
    registration flow and account management that nothing here needs.
    Format: SANAD_USERS="alice:<pbkdf2 hash>,bob:<pbkdf2 hash>".
    """
    raw = os.environ.get("SANAD_USERS", "").strip()
    users: dict[str, User] = {}
    if not raw:
        return users
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        username, _, password_hash = entry.partition(":")
        if not username or not password_hash:
            logger.warning("ignoring malformed SANAD_USERS entry")
            continue
        users[username] = User(username=username, password_hash=password_hash)
    return users


def authenticate(username: str, password: str) -> User | None:
    user = _load_users().get(username)
    if user is None:
        # Hash anyway so a missing username and a wrong password take
        # comparable time, rather than the fast path revealing which
        # usernames exist.
        hash_password(password)
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


# -- request guard ----------------------------------------------------------


def current_user(request: Request) -> str | None:
    if not config.auth_enabled:
        return None
    token = request.cookies.get(COOKIE_NAME)
    return read_session(token) if token else None


def require_user(request: Request) -> str | None:
    """FastAPI dependency: 401 unless signed in. A no-op when auth is off."""
    if not config.auth_enabled:
        return None
    user = current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="authentication required")
    return user
