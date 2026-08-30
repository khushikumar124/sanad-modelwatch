"""ModelWatch REST API.

Thin HTTP layer over MonitoringEngine: every endpoint is a direct
translation of an engine method. The dashboard (static HTML/JS) is mounted
at /dashboard and talks to these same endpoints.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from modelwatch.api.adapter_registry import ADAPTER_REGISTRY, build_adapter
from modelwatch.api.schemas import (
    AlertResponse,
    CheckRequest,
    CheckResponse,
    DiagnosisResponse,
    ExperimentResponse,
    HealthResponse,
    ModelResponse,
    RecordExperimentRequest,
    RegisterModelRequest,
    RetrainRequest,
    RunResponse,
    VersionResponse,
)
from modelwatch.config import config
from modelwatch.core.engine import MonitoringEngine, ModelNotRegisteredError, RunNotFoundError
from modelwatch.core.storage import ModelAlreadyExistsError, ModelNotFoundError, Storage

logging.basicConfig(
    level=getattr(logging, config.log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

storage = Storage(config.db_path)
engine = MonitoringEngine(storage)

app = FastAPI(title="ModelWatch API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _reattach_adapters() -> None:
    """Re-create a live adapter instance for every model already in
    storage. Safe because the built-in adapters are stateless -- all their
    behavior is driven by config plus the persisted baseline, not by
    instance state -- so a fresh instance is equivalent to the original.
    """
    for model in engine.list_models():
        if model["adapter_name"] in ADAPTER_REGISTRY:
            engine.attach_adapter(model["model_id"], build_adapter(model["adapter_name"]))
        else:
            logger.warning(
                "model has unknown adapter_name, cannot reattach",
                extra={"model_id": model["model_id"], "adapter_name": model["adapter_name"]},
            )


@app.post("/models", response_model=ModelResponse, status_code=201)
def register_model(req: RegisterModelRequest):
    try:
        adapter = build_adapter(req.adapter_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        return engine.register_model(req.model_id, req.name, adapter, req.baseline_data, config=req.config)
    except ModelAlreadyExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/models", response_model=list[ModelResponse])
def list_models():
    return engine.list_models()


@app.get("/models/{model_id}", response_model=ModelResponse)
def get_model(model_id: str):
    model = engine.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"model '{model_id}' not found")
    return model


@app.post("/models/{model_id}/check", response_model=CheckResponse)
def run_check(model_id: str, req: CheckRequest):
    try:
        return engine.run_check(model_id, req.new_data)
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelNotRegisteredError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/models/{model_id}/history", response_model=list[RunResponse])
def get_history(model_id: str, limit: int | None = None):
    if engine.get_model(model_id) is None:
        raise HTTPException(status_code=404, detail=f"model '{model_id}' not found")
    return engine.get_history(model_id, limit=limit)


@app.get("/alerts", response_model=list[AlertResponse])
def get_alerts(model_id: str | None = None, active_only: bool = False):
    return engine.get_alerts(model_id=model_id, active_only=active_only)


@app.post("/models/{model_id}/retrain", response_model=ModelResponse)
def trigger_retrain(model_id: str, req: RetrainRequest):
    try:
        # retrain_fn is a no-op here: the REST API only resets the baseline to
        # fresh data. Callers with an actual retrainable model (e.g. a script
        # holding the real classifier) should call engine.trigger_retrain()
        # directly with a real retrain_fn instead of going through this endpoint.
        return engine.trigger_retrain(model_id, lambda _data: None, req.new_training_data)
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelNotRegisteredError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/models/{model_id}/health", response_model=HealthResponse)
def get_health(model_id: str):
    if engine.get_model(model_id) is None:
        raise HTTPException(status_code=404, detail=f"model '{model_id}' not found")
    return engine.get_health(model_id)


@app.get("/runs/{run_id}/diagnosis", response_model=DiagnosisResponse)
def get_diagnosis(run_id: int):
    try:
        result = engine.diagnose_run(run_id)
    except RunNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return result.to_dict()


@app.get("/models/{model_id}/versions", response_model=list[VersionResponse])
def get_versions(model_id: str):
    if engine.get_model(model_id) is None:
        raise HTTPException(status_code=404, detail=f"model '{model_id}' not found")
    return engine.get_versions(model_id)


@app.post("/experiments", response_model=ExperimentResponse, status_code=201)
def record_experiment(req: RecordExperimentRequest):
    return engine.record_experiment(req.name, req.kind, req.config, req.results, status=req.status)


@app.get("/experiments", response_model=list[ExperimentResponse])
def list_experiments(kind: str | None = None):
    return engine.list_experiments(kind=kind)


@app.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(experiment_id: int):
    experiment = engine.get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"experiment '{experiment_id}' not found")
    return experiment


@app.get("/", include_in_schema=False)
def _root_to_dashboard():
    """The dashboard lives under /dashboard, so a bare host:port 404s --
    which reads as "the server is broken" rather than "wrong path".
    Send the root there instead."""
    return RedirectResponse(url="/dashboard/")


_DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"
app.mount("/dashboard", StaticFiles(directory=str(_DASHBOARD_DIR), html=True), name="dashboard")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.api_host, port=config.api_port)
