"""Sanad REST API: upload a contract, summarize it, chat with it.

Thin HTTP layer over the shared ingestion pipeline and the two features.
Ingested documents live in an in-memory registry keyed by doc_id -- this
is a single-process demo app, not a multi-user service, so that's enough;
restarting the server means re-uploading documents.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from sanad.api.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentResponse,
    SetModelRequest,
    SetModelResponse,
    SummaryResponse,
)
from sanad.config import config
from sanad.features.chatbot import ask
from sanad.features.summarizer import summarize
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


class DocumentRecord:
    def __init__(self, ingested: IngestedDocument, filename: str, contract_type: str | None):
        self.ingested = ingested
        self.filename = filename
        self.contract_type = contract_type
        self.uploaded_at = datetime.now(timezone.utc).isoformat()

    def to_response(self) -> DocumentResponse:
        return DocumentResponse(
            doc_id=self.ingested.doc_id,
            filename=self.filename,
            contract_type=self.contract_type,
            chunk_count=len(self.ingested.chunks),
            used_ocr=self.ingested.used_ocr,
            uploaded_at=self.uploaded_at,
        )


documents: dict[str, DocumentRecord] = {}

app = FastAPI(title="Sanad API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_record(doc_id: str) -> DocumentRecord:
    record = documents.get(doc_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"document '{doc_id}' not found")
    return record


@app.post("/api/documents", response_model=DocumentResponse, status_code=201)
async def upload_document(file: UploadFile = File(...), contract_type: str | None = Form(None)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported file type '{ext}', expected one of {sorted(SUPPORTED_EXTENSIONS)}",
        )

    doc_id = str(uuid.uuid4())
    dest = Path(config.upload_dir) / f"{doc_id}{ext}"
    dest.write_bytes(await file.read())

    ingested = ingest_document(str(dest), doc_id, vector_store)
    record = DocumentRecord(ingested, filename=file.filename or dest.name, contract_type=contract_type)
    documents[doc_id] = record

    logger.info("document uploaded", extra={"doc_id": doc_id, "doc_filename": record.filename})
    return record.to_response()


@app.get("/api/documents", response_model=list[DocumentResponse])
def list_documents():
    return [r.to_response() for r in documents.values()]


@app.get("/api/documents/{doc_id}", response_model=DocumentResponse)
def get_document(doc_id: str):
    return _get_record(doc_id).to_response()


@app.delete("/api/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str):
    record = _get_record(doc_id)
    vector_store.delete_document(doc_id)
    Path(record.ingested.source_path).unlink(missing_ok=True)
    del documents[doc_id]


@app.post("/api/documents/{doc_id}/summarize", response_model=SummaryResponse)
def summarize_document(doc_id: str):
    record = _get_record(doc_id)
    try:
        result = summarize(record.ingested.text, llm_client)
    except LLMConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return result.to_dict()


@app.post("/api/documents/{doc_id}/chat", response_model=ChatResponse)
def chat_with_document(doc_id: str, req: ChatRequest):
    _get_record(doc_id)  # 404 if unknown, even though vector_store would just return no hits
    try:
        result = ask(doc_id, req.question, vector_store, llm_client)
    except LLMConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return result.to_dict()


@app.get("/api/admin/model", response_model=SetModelResponse)
def get_active_model():
    return {"model": llm_client.model}


@app.post("/api/admin/model", response_model=SetModelResponse)
def set_active_model(req: SetModelRequest):
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
