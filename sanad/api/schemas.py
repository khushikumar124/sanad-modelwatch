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


class ClauseResponse(BaseModel):
    index: int
    heading: str | None
    text: str


class ClausesResponse(BaseModel):
    clauses: list[ClauseResponse]


class ObligationsResponse(BaseModel):
    obligations: list[dict[str, Any]]
    parse_error: bool
    grounded_count: int
    total_count: int


class CoverageResponse(BaseModel):
    results: list[dict[str, Any]]
    not_found_count: int


class ReviewResponse(BaseModel):
    top_issues: list[dict[str, Any]]
    negotiable_clauses: list[dict[str, Any]]
    questions_to_ask: list[str]
    clarification_areas: list[str]
    #: the obligations extracted along the way (review needs them
    #: internally for contradiction detection) -- included so the
    #: frontend doesn't have to make a second, duplicate extraction
    #: call just to show the obligations table.
    obligations: dict[str, Any]


class ComparisonResponse(BaseModel):
    counts_a: dict[str, int]
    counts_b: dict[str, int]
    only_in_a: list[dict[str, Any]]
    only_in_b: list[dict[str, Any]]
    shared: list[dict[str, Any]]


class ChatResponse(BaseModel):
    answer: str
    grounded: bool
    cited_chunks: list[dict[str, Any]]
    retrieved_chunks: list[dict[str, Any]]
    parse_error: bool
    #: observable pipeline trace -- retrieval ranking/scores, claim-level
    #: evidence verification, grounding/citation scores. See
    #: sanad/features/trace.py. Never includes hidden model reasoning.
    trace: dict[str, Any]
