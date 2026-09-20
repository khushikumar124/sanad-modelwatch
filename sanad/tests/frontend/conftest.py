"""Boots the real Sanad app (real FastAPI routes, real ingestion
pipeline, a fake LLM client) on a real local port, so Playwright can
drive the actual frontend/index.html exactly as a person would in a
browser -- clicking tabs, typing into the search box, reading rendered
DOM state -- rather than only exercising the API layer the way
sanad/tests/test_api.py does.

Isolated the same way test_api.py is: unique chroma/upload/db paths per
test session, set before sanad.api.app is imported (VectorStore() and
the upload dir are created at module import time). Auth is left off
(SANAD_AUTH_ENABLED defaults to false) so tests don't need to drive the
login screen first.
"""
from __future__ import annotations

import os
import socket
import threading
import time
import uuid

_TEST_ID = uuid.uuid4().hex
os.environ.setdefault("SANAD_CHROMA_DB_PATH", f"/tmp/sanad_e2e_chroma_{_TEST_ID}")
os.environ.setdefault("SANAD_UPLOAD_DIR", f"/tmp/sanad_e2e_uploads_{_TEST_ID}")
os.environ.setdefault("SANAD_DATABASE_URL", f"sqlite:////tmp/sanad_e2e_documents_{_TEST_ID}.db")

import pytest
import uvicorn

from sanad.api.app import app
import sanad.api.app as sanad_app_module
from sanad.rag.llm_client import LLMClient


class FakeLLMClient(LLMClient):
    """Returns a fixed, well-formed answer for every call. Good enough
    for exercising the frontend's rendering of a grounded chat answer;
    not a substitute for sanad/tests/test_chatbot.py's per-scenario
    (refusal, ungrounded-citation, no-context) coverage, which stays
    against the real `ask()` function directly."""

    # The topbar's model chip calls GET /api/admin/model, which reads
    # this attribute directly off whatever's assigned to the module-level
    # llm_client -- not part of the LLMClient interface itself, but real
    # OllamaClient instances have it, so this fake needs it too.
    model = "fake-model-for-tests"

    def generate(self, system_prompt: str, user_prompt: str, response_schema: dict | None = None, timeout: float = 180) -> str:
        import json

        return json.dumps(
            {
                "grounded": True,
                "answer": "The agreement is governed by the laws of India, per the governing law clause.",
                "cited_excerpts": [1],
            }
        )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server():
    """Starts the real app in a background thread for the whole test
    module (uploads/state accumulate across tests in this file, same
    tradeoff sanad/tests/test_api.py's module-scoped fixtures make).

    Restores the real llm_client on teardown -- `sanad.api.app.llm_client`
    is a module-level global, so leaving the swap in place after this
    fixture tears down would leak into whatever test module happens to
    run next in the same process. Caught for real, twice: first with no
    restore at all, then with a restore that only ran at *session*
    teardown (this fixture was scope="session") -- by which point
    sanad/tests/test_api.py's "returns 503 when Ollama is unreachable"
    tests (which sorts alphabetically after this directory) had already
    run against the fake, always-succeeding client instead of a real
    unreachable one. Module scope means the restore happens right after
    this file's own tests finish, before pytest moves on to the next
    module."""
    original_llm_client = sanad_app_module.llm_client
    sanad_app_module.llm_client = FakeLLMClient()

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("live_server didn't start within 10s")

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=5)
    sanad_app_module.llm_client = original_llm_client
