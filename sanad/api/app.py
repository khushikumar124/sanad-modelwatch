"""Sanad REST API: upload a contract, summarize it, chat with it.

Thin HTTP layer over the shared ingestion pipeline and the two features.

Ingested documents live in an in-memory registry keyed by doc_id, backed
by a small JSON record written next to each uploaded file. The registry is
rebuilt from those records at startup, so a restart doesn't invalidate a
doc_id that a browser is still holding -- previously that surfaced as
"document ... not found" on a document the user had just uploaded
successfully. Chunks and embeddings already persist in ChromaDB, so
nothing is re-extracted or re-embedded on restore.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from sanad.api.auth import COOKIE_NAME, authenticate, create_session, current_user, require_user
from sanad.api import telemetry
from sanad.api.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentResponse,
    LoginRequest,
    RiskResponse,
    SessionResponse,
    SetModelRequest,
    SetModelResponse,
    SummaryResponse,
)
from sanad.config import config
from sanad.features.chatbot import ask
from sanad.features.risk_flagger import flag_risks
from sanad.features.summarizer import summarize
from sanad.ingestion.chunking import chunk_document
from sanad.ingestion.extraction import IMAGE_EXTENSIONS
from sanad.rag.llm_client import LLMConnectionError, OllamaClient
from sanad.rag.pipeline import IngestedDocument, ingest_document
from sanad.rag.vector_store import VectorStore

logging.basicConfig(
    level=getattr(logging, config.log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf"} | IMAGE_EXTENSIONS

vector_store = VectorStore()
llm_client = OllamaClient()

Path(config.upload_dir).mkdir(parents=True, exist_ok=True)

if config.auth_enabled and not config.session_secret:
    raise RuntimeError(
        "SANAD_AUTH_ENABLED is on but SANAD_SESSION_SECRET is unset. Sessions would be "
        "signed with an ephemeral key, so every restart would silently log everyone out. "
        "Generate one with: python -c 'import secrets; print(secrets.token_hex(32))'"
    )


@dataclass
class DocumentRecord:
    """What the API needs to serve an already-ingested document.

    Kept flat and JSON-serializable rather than wrapping IngestedDocument,
    so a record can be written to disk on upload and read back on startup
    without re-running extraction or re-embedding.
    """

    doc_id: str
    filename: str
    contract_type: str | None
    chunk_count: int
    used_ocr: bool
    uploaded_at: str
    text: str
    source_path: str

    def to_response(self) -> DocumentResponse:
        return DocumentResponse(
            doc_id=self.doc_id,
            filename=self.filename,
            contract_type=self.contract_type,
            chunk_count=self.chunk_count,
            used_ocr=self.used_ocr,
            uploaded_at=self.uploaded_at,
        )

    @classmethod
    def from_ingested(
        cls, ingested: IngestedDocument, filename: str, contract_type: str | None
    ) -> "DocumentRecord":
        return cls(
            doc_id=ingested.doc_id,
            filename=filename,
            contract_type=contract_type,
            chunk_count=len(ingested.chunks),
            used_ocr=ingested.used_ocr,
            uploaded_at=datetime.now(timezone.utc).isoformat(),
            text=ingested.text,
            source_path=ingested.source_path,
        )


documents: dict[str, DocumentRecord] = {}


def _sidecar_path(doc_id: str) -> Path:
    return Path(config.upload_dir) / f"{doc_id}.json"


def _persist(record: DocumentRecord) -> None:
    """Write a record beside its uploaded file.

    Without this the registry is process-memory only, so restarting the
    server silently invalidates every doc_id a browser is already holding
    -- the user sees "document ... not found" on a document they just
    uploaded successfully. The chunks themselves already persist in
    ChromaDB; this is the metadata and extracted text that went with them.
    """
    try:
        _sidecar_path(record.doc_id).write_text(json.dumps(asdict(record)))
    except OSError as e:
        logger.warning("could not persist document record", extra={"doc_id": record.doc_id, "err": str(e)})


def _restore_documents() -> None:
    upload_dir = Path(config.upload_dir)
    if not upload_dir.is_dir():
        return
    for sidecar in upload_dir.glob("*.json"):
        try:
            documents[sidecar.stem] = DocumentRecord(**json.loads(sidecar.read_text()))
        except (OSError, ValueError, TypeError) as e:
            logger.warning("skipping unreadable record", extra={"file": sidecar.name, "err": str(e)})
    if documents:
        logger.info("restored documents from disk", extra={"count": len(documents)})

_restore_documents()

app = FastAPI(title="Sanad API")
# The frontend is served from this same origin, so no cross-origin access is
# needed. A wildcard here would let any page you visit drive this API with
# your session cookie attached.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"http://localhost:{config.api_port}", f"http://127.0.0.1:{config.api_port}"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_record(doc_id: str) -> DocumentRecord:
    record = documents.get(doc_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"document '{doc_id}' not found")
    return record


@app.get("/api/auth/session", response_model=SessionResponse)
def get_session(request: Request):
    """Who am I, and is auth even switched on? The frontend uses this to
    decide whether to show a login screen at all."""
    return {"auth_enabled": config.auth_enabled, "username": current_user(request)}


@app.post("/api/auth/login", response_model=SessionResponse)
def login(req: LoginRequest, response: Response):
    if not config.auth_enabled:
        return {"auth_enabled": False, "username": None}
    user = authenticate(req.username, req.password)
    if user is None:
        # One message for both unknown-user and wrong-password: saying which
        # was wrong tells an attacker whether a username exists.
        logger.warning("failed login attempt", extra={"username_attempted": req.username})
        raise HTTPException(status_code=401, detail="invalid username or password")
    response.set_cookie(
        COOKIE_NAME,
        create_session(user.username),
        httponly=True,          # not readable from JavaScript, so XSS cannot steal it
        samesite="lax",         # not sent on cross-site POSTs
        secure=config.session_cookie_secure,
        max_age=config.session_ttl_seconds,
        path="/",
    )
    logger.info("login succeeded", extra={"user": user.username})
    return {"auth_enabled": True, "username": user.username}


@app.post("/api/auth/logout", response_model=SessionResponse)
def logout(response: Response):
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"auth_enabled": config.auth_enabled, "username": None}


@app.post("/api/documents", response_model=DocumentResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    contract_type: str | None = Form(None),
    _user: str | None = Depends(require_user),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported file type '{ext}', expected one of {sorted(SUPPORTED_EXTENSIONS)}",
        )

    doc_id = str(uuid.uuid4())
    upload_dir = Path(config.upload_dir)
    # Recreate per-request rather than only at import: the directory can go
    # away under a long-running server (temp cleaners, a manual rm), and a
    # missing directory shouldn't turn every later upload into a 500.
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f"{doc_id}{ext}"
    dest.write_bytes(await file.read())

    ingested = ingest_document(str(dest), doc_id, vector_store)
    record = DocumentRecord.from_ingested(
        ingested, filename=file.filename or dest.name, contract_type=contract_type
    )
    documents[doc_id] = record
    _persist(record)

    logger.info("document uploaded", extra={"doc_id": doc_id, "doc_filename": record.filename})
    return record.to_response()


@app.get("/api/documents", response_model=list[DocumentResponse])
def list_documents(_user: str | None = Depends(require_user)):
    return [r.to_response() for r in documents.values()]


@app.get("/api/documents/{doc_id}", response_model=DocumentResponse)
def get_document(doc_id: str, _user: str | None = Depends(require_user)):
    return _get_record(doc_id).to_response()


@app.delete("/api/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str, _user: str | None = Depends(require_user)):
    record = _get_record(doc_id)
    vector_store.delete_document(doc_id)
    Path(record.source_path).unlink(missing_ok=True)
    _sidecar_path(doc_id).unlink(missing_ok=True)
    del documents[doc_id]


@app.post("/api/documents/{doc_id}/summarize", response_model=SummaryResponse)
def summarize_document(doc_id: str, _user: str | None = Depends(require_user)):
    record = _get_record(doc_id)
    try:
        result = summarize(record.text, llm_client)
    except LLMConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return result.to_dict()


@app.get("/api/documents/{doc_id}/risks", response_model=RiskResponse)
def get_risks(doc_id: str, _user: str | None = Depends(require_user)):
    """Rule-based scan for unfavourable clauses. No LLM call, so this is
    instant and its output is reproducible -- re-chunking from the stored
    text is cheap next to re-extracting the PDF."""
    record = _get_record(doc_id)
    return flag_risks(chunk_document(record.text)).to_dict()


@app.post("/api/documents/{doc_id}/chat", response_model=ChatResponse)
def chat_with_document(
    doc_id: str, req: ChatRequest, _user: str | None = Depends(require_user)
):
    _get_record(doc_id)  # 404 if unknown, even though vector_store would just return no hits
    started = time.perf_counter()
    try:
        result = ask(doc_id, req.question, vector_store, llm_client)
    except LLMConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    # Operational facts only -- no question or answer text. See telemetry.py.
    telemetry.record_chat(
        grounded=result.grounded,
        citations=len(result.cited_chunks),
        latency_ms=(time.perf_counter() - started) * 1000,
        parse_error=result.parse_error,
        retrieved=len(result.retrieved_chunks),
    )
    return result.to_dict()


@app.get("/api/telemetry")
def get_telemetry(drain: bool = False, _user: str | None = Depends(require_user)):
    """Recent operational events, for an external monitor to collect.

    Sanad does not push anywhere -- it publishes, and something else
    decides to look. `drain=true` consumes the buffer so a polling
    collector sees each question once.
    """
    return {"events": telemetry.snapshot(drain=drain)}


@app.get("/api/admin/model", response_model=SetModelResponse)
def get_active_model(_user: str | None = Depends(require_user)):
    return {"model": llm_client.model}


@app.post("/api/admin/model", response_model=SetModelResponse)
def set_active_model(req: SetModelRequest, _user: str | None = Depends(require_user)):
    """Swap the live Ollama model at runtime, without restarting the
    server. This exists purely to support ModelWatch's drift-simulation
    demo (see modelwatch/examples/simulate_drift_demo.py) -- swapping the
    model mid-run is what lets the detect -> alert -> retrain -> recover
    story be demonstrated live instead of requiring a server restart.
    Not a general-purpose admin surface; there's no auth on this route."""
    llm_client.model = req.model
    logger.info("active model changed", extra={"model": req.model})
    return {"model": llm_client.model}


_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.api_host, port=config.api_port)
