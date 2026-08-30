"""Pydantic request/response models for the ModelWatch REST API.

Adapter-specific payloads (baseline_data, new_data, new_training_data) are
typed Any deliberately: their shape is defined by each ModelAdapter, not by
the API, so validating them here would re-couple the API to adapter
internals.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class RegisterModelRequest(BaseModel):
    model_id: str
    name: str
    adapter_name: str
    baseline_data: Any
    #: optional registry record -- embedding model, chunk size, top-k,
    #: prompt version, dataset version, whatever describes this model's
    #: configuration at registration time. Opaque to the API/engine.
    config: dict[str, Any] = {}


class CheckRequest(BaseModel):
    new_data: Any


class RetrainRequest(BaseModel):
    new_training_data: Any


class SignalResponse(BaseModel):
    name: str
    value: float
    is_drifted: bool
    detail: dict[str, Any]


class CheckResponse(BaseModel):
    run_id: int
    alert_id: int | None
    health_state: str
    drift_score: float
    quality_score: float | None
    is_drifted: bool
    signals: list[SignalResponse]
    statistics: dict[str, Any] = {}


class HealthResponse(BaseModel):
    model_id: str
    state: str
    consecutive_drifted: int
    consecutive_clean: int
    updated_at: str | None


class RankedSubsystem(BaseModel):
    subsystem: str
    score: float


class DiagnosisResponse(BaseModel):
    likely_subsystem: str | None
    confidence: float
    reasoning: list[str]
    ranked: list[RankedSubsystem]


class RecordExperimentRequest(BaseModel):
    name: str
    kind: str
    config: dict[str, Any] = {}
    results: dict[str, Any] = {}
    status: str = "completed"


class ExperimentResponse(BaseModel):
    id: int
    name: str
    kind: str
    created_at: str
    config: dict[str, Any]
    results: dict[str, Any]
    status: str


class ModelResponse(BaseModel):
    model_id: str
    name: str
    adapter_name: str
    current_version: int
    created_at: str
    config: dict[str, Any] = {}


class RunResponse(BaseModel):
    id: int
    model_id: str
    version: int
    timestamp: str
    drift_score: float
    quality_score: float | None
    is_drifted: bool
    signals: list[dict[str, Any]]
    statistics: dict[str, Any] = {}


class AlertResponse(BaseModel):
    id: int
    model_id: str
    run_id: int
    created_at: str
    message: str
    resolved: bool
    resolved_at: str | None


class VersionResponse(BaseModel):
    id: int
    model_id: str
    version: int
    created_at: str
    reason: str


class RecordTraceRequest(BaseModel):
    trace_id: str
    model_id: str
    data: dict[str, Any]


class TraceResponse(BaseModel):
    id: int
    trace_id: str
    model_id: str
    created_at: str
    data: dict[str, Any]


class TraceDiagnosisResponse(BaseModel):
    category: str
    reasoning: list[str]
    evidence: dict[str, Any]
    operational_note: str | None
