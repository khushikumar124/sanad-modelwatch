"""Document registry, backed by a real database instead of an
in-memory dict plus one JSON sidecar file per document.

The previous registry (process-memory dict, restored from JSON files
glob'd from the upload directory on startup) works for exactly one
process. It has no answer for two app instances behind a load balancer,
or for the small-but-real risk of a JSON file half-written by a crash
mid-`write_text`. A real database is the fix for both, and it does not
have to mean new infrastructure: SQLAlchemy Core (no ORM -- this project
otherwise talks to SQLite in raw SQL, see modelwatch/core/storage.py,
and Core is the right amount of abstraction to also support Postgres
without hand-writing two SQL dialects) defaults to a local SQLite file
so a fresh `./run.sh` still needs nothing installed. Point
SANAD_DATABASE_URL at a real Postgres instance
(e.g. postgresql+psycopg://user:pass@localhost/sanad) the moment this
runs on more than one machine.

Kept intentionally close to the shape sanad/api/app.py already used
(the DocumentRecord dataclass, to_response(), from_ingested()) so this
swap doesn't ripple through the rest of the API layer -- only the
storage functions changed, not what callers pass or receive.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, Integer, MetaData, String, Table, Text, create_engine, delete, insert, select
from sqlalchemy.engine import Engine

from sanad.config import config
from sanad.rag.pipeline import IngestedDocument

logger = logging.getLogger(__name__)

metadata = MetaData()

documents_table = Table(
    "documents",
    metadata,
    Column("doc_id", String, primary_key=True),
    Column("filename", String, nullable=False),
    Column("contract_type", String, nullable=True),
    Column("chunk_count", Integer, nullable=False),
    Column("used_ocr", Boolean, nullable=False),
    Column("uploaded_at", String, nullable=False),
    Column("text", Text, nullable=False),
    Column("source_path", String, nullable=False),
)


@dataclass
class DocumentRecord:
    """What the API needs to serve an already-ingested document. Kept
    flat so a row round-trips through this dataclass without a
    translation layer at every call site."""

    doc_id: str
    filename: str
    contract_type: str | None
    chunk_count: int
    used_ocr: bool
    uploaded_at: str
    text: str
    source_path: str

    def to_response(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "contract_type": self.contract_type,
            "chunk_count": self.chunk_count,
            "used_ocr": self.used_ocr,
            "uploaded_at": self.uploaded_at,
        }

    @classmethod
    def from_ingested(cls, ingested: IngestedDocument, filename: str, contract_type: str | None) -> "DocumentRecord":
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


_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        # check_same_thread=False: FastAPI's default threadpool executor
        # can call in from multiple threads, same reasoning as
        # modelwatch/core/storage.py's Storage class -- SQLite connections
        # aren't safe to share across threads by default otherwise.
        # Irrelevant for Postgres, where connect_args is simply unused.
        connect_args = {"check_same_thread": False} if config.database_url.startswith("sqlite") else {}
        _engine = create_engine(config.database_url, connect_args=connect_args, future=True)
        metadata.create_all(_engine)
    return _engine


def reset_engine() -> None:
    """Test-only: drop the cached engine so a new one (e.g. pointed at a
    fresh tmp_path database) is created on next use."""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None


def _row_to_record(row) -> DocumentRecord:
    return DocumentRecord(
        doc_id=row.doc_id, filename=row.filename, contract_type=row.contract_type,
        chunk_count=row.chunk_count, used_ocr=bool(row.used_ocr), uploaded_at=row.uploaded_at,
        text=row.text, source_path=row.source_path,
    )


def save_document(record: DocumentRecord) -> None:
    with get_engine().begin() as conn:
        conn.execute(delete(documents_table).where(documents_table.c.doc_id == record.doc_id))
        conn.execute(insert(documents_table).values(**asdict(record)))


def get_document(doc_id: str) -> DocumentRecord | None:
    with get_engine().connect() as conn:
        row = conn.execute(select(documents_table).where(documents_table.c.doc_id == doc_id)).first()
        return _row_to_record(row) if row else None


def list_documents() -> list[DocumentRecord]:
    with get_engine().connect() as conn:
        rows = conn.execute(select(documents_table).order_by(documents_table.c.uploaded_at)).all()
        return [_row_to_record(r) for r in rows]


def delete_document(doc_id: str) -> None:
    with get_engine().begin() as conn:
        conn.execute(delete(documents_table).where(documents_table.c.doc_id == doc_id))
