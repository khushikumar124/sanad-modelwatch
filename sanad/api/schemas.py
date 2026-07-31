"""Pydantic request/response models for the Sanad REST API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    doc_id: str
    filename: str
    contract_type: str | None
    chunk_count: int
    used_ocr: bool
    uploaded_at: str


class ChatRequest(BaseModel):
    question: str


class SetModelRequest(BaseModel):
    model: str


class SetModelResponse(BaseModel):
    model: str


class SummaryResponse(BaseModel):
    parties: list[str]
    key_obligations: list[str]
    important_dates: list[str]
    notice_period: str | None
    penalty_clauses: list[str]
    termination_conditions: list[str]
    parse_error: bool


class ChatResponse(BaseModel):
    answer: str
    grounded: bool
    cited_chunks: list[dict[str, Any]]
    retrieved_chunks: list[dict[str, Any]]
    parse_error: bool
