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
    drift_score: float
    quality_score: float | None
    is_drifted: bool
    signals: list[SignalResponse]


class ModelResponse(BaseModel):
    model_id: str
    name: str
    adapter_name: str
    current_version: int
    created_at: str


class RunResponse(BaseModel):
    id: int
    model_id: str
    version: int
    timestamp: str
    drift_score: float
    quality_score: float | None
    is_drifted: bool
    signals: list[dict[str, Any]]


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
