"""Sanad API tests. Uses a real TestClient against the real app -- uploads
go through the real extraction/chunking/embedding pipeline. Ollama isn't
installed in this environment, so the summarize/chat 503 tests exercise
the real connection-refused path rather than a mock.

Env vars for chroma/upload paths must be set before sanad.api.app is
imported, since VectorStore() and the upload dir are created at module
import time.
"""
import os
import uuid

_TEST_ID = uuid.uuid4().hex
os.environ.setdefault("SANAD_CHROMA_DB_PATH", f"/tmp/sanad_test_chroma_{_TEST_ID}")
os.environ.setdefault("SANAD_UPLOAD_DIR", f"/tmp/sanad_test_uploads_{_TEST_ID}")

import pytest
from fastapi.testclient import TestClient

from sanad.api.app import app

client = TestClient(app)

RENTAL_DOC = "sanad/sample_docs/rental/rental_agreement_sample_1.pdf"


@pytest.fixture(scope="module")
def uploaded_doc_id():
    with open(RENTAL_DOC, "rb") as f:
        res = client.post(
            "/api/documents",
            files={"file": ("rental.pdf", f, "application/pdf")},
            data={"contract_type": "rental"},
        )
    assert res.status_code == 201
    return res.json()["doc_id"]


def test_upload_logs_at_info_without_crashing(caplog):
    """Regression test: logger.info(..., extra={...}) previously crashed
    with `KeyError: "Attempt to overwrite 'filename'"` because 'filename'
    collides with a reserved LogRecord attribute. Pytest's default log
    capture level (WARNING) silently swallowed this in every other test
    here, so it only surfaced via manual live-server testing -- this test
    forces INFO-level logging on so the collision can't hide again.
    """
    with caplog.at_level("INFO"):
        with open(RENTAL_DOC, "rb") as f:
            res = client.post("/api/documents", files={"file": ("regression.pdf", f, "application/pdf")})
    assert res.status_code == 201


def test_upload_returns_chunk_count_and_metadata(uploaded_doc_id):
    res = client.get(f"/api/documents/{uploaded_doc_id}")
    body = res.json()
    assert res.status_code == 200
    assert body["chunk_count"] > 0
    assert body["used_ocr"] is False
    assert body["contract_type"] == "rental"
    assert body["filename"] == "rental.pdf"


def test_list_documents_includes_uploaded(uploaded_doc_id):
    res = client.get("/api/documents")
    assert res.status_code == 200
    ids = [d["doc_id"] for d in res.json()]
    assert uploaded_doc_id in ids


def test_upload_unsupported_extension_returns_400():
    res = client.post(
        "/api/documents",
        files={"file": ("contract.docx", b"not a real docx", "application/octet-stream")},
    )
    assert res.status_code == 400


def test_get_unknown_document_returns_404():
    res = client.get("/api/documents/does-not-exist")
    assert res.status_code == 404


def test_summarize_unknown_document_returns_404():
    res = client.post("/api/documents/does-not-exist/summarize")
    assert res.status_code == 404


def test_chat_unknown_document_returns_404():
    res = client.post("/api/documents/does-not-exist/chat", json={"question": "hi"})
    assert res.status_code == 404


def test_risks_endpoint_returns_a_report(uploaded_doc_id):
    res = client.get(f"/api/documents/{uploaded_doc_id}/risks")
    assert res.status_code == 200
    body = res.json()
    assert "findings" in body and "counts" in body and "clauses_scanned" in body


FREELANCE_DOC = "sanad/sample_docs/freelance/freelance_agreement_sample2.pdf"


@pytest.fixture(scope="module")
def second_doc_id():
    with open(FREELANCE_DOC, "rb") as f:
        res = client.post(
            "/api/documents",
            files={"file": ("freelance.pdf", f, "application/pdf")},
            data={"contract_type": "freelance"},
        )
    assert res.status_code == 201
    return res.json()["doc_id"]


def test_compare_endpoint_returns_a_comparison(uploaded_doc_id, second_doc_id):
    res = client.get(f"/api/documents/{uploaded_doc_id}/compare/{second_doc_id}")
    assert res.status_code == 200
    body = res.json()
    for key in ("counts_a", "counts_b", "only_in_a", "only_in_b", "shared"):
        assert key in body


def test_compare_with_unknown_document_returns_404(uploaded_doc_id):
    res = client.get(f"/api/documents/{uploaded_doc_id}/compare/does-not-exist")
    assert res.status_code == 404


@pytest.fixture
def llm_unreachable():
    """Point the app's LLM client at a port with no listener.

    These tests previously relied on Ollama simply not being installed on
    the machine, which meant they passed by accident and started failing
    the moment Ollama was installed. Forcing an unreachable URL makes the
    unavailable-backend path deterministic either way.
    """
    from sanad.api import app as app_module

    original = app_module.llm_client.base_url
    app_module.llm_client.base_url = "http://127.0.0.1:9"  # discard port, no listener
    yield
    app_module.llm_client.base_url = original


def test_summarize_returns_503_when_llm_unreachable(uploaded_doc_id, llm_unreachable):
    res = client.post(f"/api/documents/{uploaded_doc_id}/summarize")
    assert res.status_code == 503
    assert "ollama" in res.json()["detail"].lower()


def test_chat_returns_503_when_llm_unreachable(uploaded_doc_id, llm_unreachable):
    res = client.post(f"/api/documents/{uploaded_doc_id}/chat", json={"question": "What is the notice period?"})
    assert res.status_code == 503
    assert "ollama" in res.json()["detail"].lower()


def test_admin_can_swap_active_model():
    original = client.get("/api/admin/model").json()["model"]

    res = client.post("/api/admin/model", json={"model": "mistral:7b"})
    assert res.status_code == 200
    assert res.json()["model"] == "mistral:7b"
    assert client.get("/api/admin/model").json()["model"] == "mistral:7b"

    client.post("/api/admin/model", json={"model": original})  # restore for other tests


def test_delete_document_removes_it():
    with open(RENTAL_DOC, "rb") as f:
        res = client.post("/api/documents", files={"file": ("to_delete.pdf", f, "application/pdf")})
    doc_id = res.json()["doc_id"]

    del_res = client.delete(f"/api/documents/{doc_id}")
    assert del_res.status_code == 204

    get_res = client.get(f"/api/documents/{doc_id}")
    assert get_res.status_code == 404


def test_frontend_is_served_at_root():
    res = client.get("/")
    assert res.status_code == 200
    assert "Sanad" in res.text
