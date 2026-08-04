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


class LoginRequest(BaseModel):
    username: str
    password: str


class SessionResponse(BaseModel):
    auth_enabled: bool
    username: str | None


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


class RiskResponse(BaseModel):
    findings: list[dict[str, Any]]
    clauses_scanned: int
    counts: dict[str, int]


class ChatResponse(BaseModel):
    answer: str
    grounded: bool
    cited_chunks: list[dict[str, Any]]
    retrieved_chunks: list[dict[str, Any]]
    parse_error: bool
