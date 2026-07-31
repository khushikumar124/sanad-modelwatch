"""Central configuration for Sanad, sourced from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Config:
    # Chunking: clause-aware chunker merges/splits to stay within these bounds.
    chunk_max_chars: int = _int_env("SANAD_CHUNK_MAX_CHARS", 1500)
    chunk_min_chars: int = _int_env("SANAD_CHUNK_MIN_CHARS", 200)

    # OCR fallback (Tesseract), used only for pages/images with no text layer.
    ocr_language: str = os.environ.get("SANAD_OCR_LANGUAGE", "eng")
    ocr_dpi: int = _int_env("SANAD_OCR_DPI", 300)

    # Embeddings: all-MiniLM-L6-v2 is a small (~80MB), fast sentence-transformer
    # -- a reasonable default for a laptop with no GPU. Trades some retrieval
    # quality for speed/size versus larger models (e.g. all-mpnet-base-v2).
    embedding_model: str = os.environ.get("SANAD_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Vector store
    chroma_db_path: str = os.environ.get("SANAD_CHROMA_DB_PATH", "sanad_chroma_db")
    chroma_collection: str = os.environ.get("SANAD_CHROMA_COLLECTION", "contracts")

    # Retrieval
    retrieval_top_k: int = _int_env("SANAD_RETRIEVAL_TOP_K", 4)

    # API / server
    api_host: str = os.environ.get("SANAD_API_HOST", "0.0.0.0")
    api_port: int = _int_env("SANAD_API_PORT", 8100)
    upload_dir: str = os.environ.get("SANAD_UPLOAD_DIR", "sanad_uploads")

    # LLM (Ollama), used by rag/llm_client.py from Stage 4 onward
    ollama_base_url: str = os.environ.get("SANAD_OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.environ.get("SANAD_OLLAMA_MODEL", "llama3.2:3b")

    log_level: str = os.environ.get("SANAD_LOG_LEVEL", "INFO")


config = Config()
