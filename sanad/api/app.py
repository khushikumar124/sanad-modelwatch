"""Sanad REST API: upload a contract, summarize it, chat with it.

Thin HTTP layer over the shared ingestion pipeline and the two features.

Ingested documents live in a real database (sanad/db.py), not an
in-memory dict -- a restart doesn't invalidate a doc_id a browser is
still holding, and (unlike the in-memory-dict-plus-JSON-sidecar-file
registry this replaced) more than one process can see the same
documents. Chunks and embeddings already persist in ChromaDB, so
nothing is re-extracted or re-embedded when a document is looked up.
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from sanad import db
from sanad.api.auth import COOKIE_NAME, authenticate, create_session, current_user, require_user
from sanad.api import telemetry
from sanad.api.schemas import (
    ChatRequest,
    ChatResponse,
    ClausesResponse,
    ComparisonResponse,
    CoverageResponse,
    DocumentResponse,
    LoginRequest,
    ObligationsResponse,
    ReviewResponse,
    RiskResponse,
    SessionResponse,
    SetModelRequest,
    SetModelResponse,
    SummaryResponse,
)
from sanad.config import config
from sanad.features.chatbot import ask
from sanad.features.comparison import compare_risk_reports
from sanad.features.contradictions import find_contradictions
from sanad.features.coverage import check_coverage
from sanad.features.obligations import extract_obligations
from sanad.features.review import build_review
from sanad.features.risk_flagger import flag_risks
from sanad.features.trace import build_trace
from sanad.features.summarizer import summarize
from sanad.jobs import jobs
from sanad.ingestion.chunking import chunk_document
from sanad.ingestion.extraction import IMAGE_EXTENSIONS
from sanad.rag.llm_client import LLMConnectionError, OllamaClient
from sanad.rag.pipeline import ingest_document
from sanad.rag.vector_store import VectorStore
from sanad.security import UploadValidationError, validate_upload
from sanad.storage import get_object_store

logging.basicConfig(
    level=getattr(logging, config.log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf"} | IMAGE_EXTENSIONS

vector_store = VectorStore()
llm_client = OllamaClient()
object_store = get_object_store()

Path(config.upload_dir).mkdir(parents=True, exist_ok=True)

if config.auth_enabled and not config.session_secret:
    raise RuntimeError(
        "SANAD_AUTH_ENABLED is on but SANAD_SESSION_SECRET is unset. Sessions would be "
        "signed with an ephemeral key, so every restart would silently log everyone out. "
        "Generate one with: python -c 'import secrets; print(secrets.token_hex(32))'"
    )


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


def _get_record(doc_id: str) -> db.DocumentRecord:
    record = db.get_document(doc_id)
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

    data = await file.read()
    try:
        validate_upload(ext, data)
    except UploadValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    doc_id = str(uuid.uuid4())
    locator = object_store.save(doc_id, ext, data)

    with object_store.open_local(locator) as local_path:
        ingested = ingest_document(local_path, doc_id, vector_store)
    ingested.source_path = locator

    record = db.DocumentRecord.from_ingested(
        ingested, filename=file.filename or f"{doc_id}{ext}", contract_type=contract_type
    )
    db.save_document(record)

    logger.info("document uploaded", extra={"doc_id": doc_id, "doc_filename": record.filename})
    return record.to_response()


@app.get("/api/documents", response_model=list[DocumentResponse])
def list_documents(_user: str | None = Depends(require_user)):
    return [r.to_response() for r in db.list_documents()]


@app.get("/api/documents/{doc_id}", response_model=DocumentResponse)
def get_document(doc_id: str, _user: str | None = Depends(require_user)):
    return _get_record(doc_id).to_response()


@app.delete("/api/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str, _user: str | None = Depends(require_user)):
    record = _get_record(doc_id)
    vector_store.delete_document(doc_id)
    object_store.delete(record.source_path)
    db.delete_document(doc_id)


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


@app.get("/api/documents/{doc_id}/clauses", response_model=ClausesResponse)
def get_clauses(doc_id: str, _user: str | None = Depends(require_user)):
    """The whole document, clause by clause, in order. No LLM call --
    same chunking every other feature scores/cites against, so a chunk
    index from a risk finding, coverage result, or obligation always
    refers to the same clause here. This is what the frontend's
    click-to-source jump and risk heatmap are built on."""
    record = _get_record(doc_id)
    chunks = chunk_document(record.text)
    return {"clauses": [{"index": c.index, "heading": c.heading, "text": c.text} for c in chunks]}


@app.get("/api/documents/{doc_id}/coverage", response_model=CoverageResponse)
def get_coverage(doc_id: str, _user: str | None = Depends(require_user)):
    """Rule-based scan for standard clause categories that don't appear
    anywhere in the document. No LLM call, instant, reproducible -- same
    reasoning as get_risks(). See coverage.py: `not_found` never means
    `missing`, only that this scan's patterns didn't match anything."""
    record = _get_record(doc_id)
    return check_coverage(chunk_document(record.text)).to_dict()


def _compute_obligations(doc_id: str) -> dict:
    record = _get_record(doc_id)
    return extract_obligations(record.text, llm_client).to_dict()


def _compute_review(doc_id: str) -> dict:
    record = _get_record(doc_id)
    chunks = chunk_document(record.text)
    risk_report = flag_risks(chunks)
    coverage_report = check_coverage(chunks)
    obligations_report = extract_obligations(record.text, llm_client)
    contradiction_report = find_contradictions(obligations_report.obligations)
    review = build_review(risk_report, coverage_report, contradiction_report)
    return {**review.to_dict(), "obligations": obligations_report.to_dict()}


@app.get("/api/documents/{doc_id}/obligations", response_model=ObligationsResponse)
def get_obligations(doc_id: str, _user: str | None = Depends(require_user)):
    """Structured obligation/deadline extraction. Unlike risks/coverage
    this is an LLM call over the whole document -- "who owes what to
    whom by when" needs language understanding a rule engine can't do.
    Every obligation is grounding-checked against the document's own
    text (see obligations.py) before being trusted.

    This blocks the request for as long as the model takes (measured:
    anywhere from ~10s to several minutes -- see obligations.py's
    docstring). Prefer POST .../obligations/job for anything driving a
    UI; this synchronous form is kept for scripts/tests/curl."""
    try:
        return _compute_obligations(doc_id)
    except LLMConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/documents/{doc_id}/review", response_model=ReviewResponse)
def get_review(doc_id: str, _user: str | None = Depends(require_user)):
    """"Review Contract" synthesis: top issues, negotiable clauses,
    questions to ask, and areas to clarify -- assembled entirely from
    risk_flagger's findings, coverage.py's scan, and contradictions
    found among extracted obligations. No new judgment is added here;
    see review.py's own docstring. Calls the LLM once (for obligation
    extraction, to find contradictions), so this is not instant.

    Synchronous form, kept for scripts/tests/curl -- prefer
    POST .../review/job for anything driving a UI (see jobs.py)."""
    try:
        return _compute_review(doc_id)
    except LLMConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/api/documents/{doc_id}/obligations/job", status_code=202)
def start_obligations_job(doc_id: str, _user: str | None = Depends(require_user)):
    """Starts obligation extraction in the background and returns
    immediately with a job_id -- poll GET /api/jobs/{job_id} for the
    result. See sanad/jobs.py for why this exists: the synchronous
    version above can block a browser tab for minutes."""
    _get_record(doc_id)  # 404 fast, before handing work to a background thread
    job_id = jobs.submit("obligations", lambda: _compute_obligations(doc_id))
    return {"job_id": job_id}


@app.post("/api/documents/{doc_id}/review/job", status_code=202)
def start_review_job(doc_id: str, _user: str | None = Depends(require_user)):
    """Starts the Review synthesis in the background -- see
    start_obligations_job() and sanad/jobs.py."""
    _get_record(doc_id)
    job_id = jobs.submit("review", lambda: _compute_review(doc_id))
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str, _user: str | None = Depends(require_user)):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job '{job_id}' not found")
    return job.to_dict()


@app.get("/api/documents/{doc_id}/compare/{other_doc_id}", response_model=ComparisonResponse)
def compare_documents(
    doc_id: str, other_doc_id: str, _user: str | None = Depends(require_user)
):
    """Side-by-side risk comparison of two contracts, rule by rule. Reuses
    the same rule-based risk scan get_risks() runs -- comparison is just
    a set operation over which rules fired in each, so it's deterministic
    and instant, same as the single-document scan."""
    record_a = _get_record(doc_id)
    record_b = _get_record(other_doc_id)
    report_a = flag_risks(chunk_document(record_a.text))
    report_b = flag_risks(chunk_document(record_b.text))
    return compare_risk_reports(report_a, report_b).to_dict()


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
    trace = build_trace(
        req.question, result, model_name=config.ollama_model,
        top_k=config.retrieval_top_k, embedder=vector_store.embedder,
    )
    # Operational facts are always recorded; the full trace (question,
    # answer, clause text) rides along only when telemetry_full_trace is
    # on -- see telemetry.py's docstring for the tradeoff.
    telemetry.record_chat(
        grounded=result.grounded,
        citations=len(result.cited_chunks),
        latency_ms=(time.perf_counter() - started) * 1000,
        parse_error=result.parse_error,
        retrieved=len(result.retrieved_chunks),
        doc_id=doc_id,
        model_name=config.ollama_model,
        top_k=config.retrieval_top_k,
        retrieval_scores=[c["distance"] for c in result.retrieved_chunks],
        retrieval_latency_ms=result.retrieval_latency_ms,
        generation_latency_ms=result.generation_latency_ms,
        citations_requested=result.citations_requested,
        full_trace=trace.to_dict() if config.telemetry_full_trace else None,
        question_embedding=vector_store.embedder.embed_one(req.question) if config.telemetry_full_trace else None,
    )
    return {**result.to_dict(), "trace": trace.to_dict()}


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
