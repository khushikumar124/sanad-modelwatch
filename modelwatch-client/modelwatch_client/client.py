"""A thin HTTP client for a ModelWatch server.

This is deliberately separate from the `modelwatch` package itself
(core engine, adapters, statistical detectors, dashboard): an app that
wants to *report to* ModelWatch shouldn't need to install the whole
monitoring framework, its adapters, or its dashboard just to send an
HTTP request. This package is the seam that makes ModelWatch adoptable
by a project that never clones this repo -- everything it does, it does
by calling the same REST API the dashboard itself uses
(see modelwatch/api/app.py), nothing more.

This mirrors what modelwatch/examples/telemetry_reporter.py already does
for Sanad specifically, generalized into a reusable client any RAG or ML
app can use the same way.
"""
from __future__ import annotations

from typing import Any

import requests


class ModelWatchError(Exception):
    """Raised when a ModelWatch server request fails -- unreachable,
    or returned a non-2xx status. Wraps the underlying requests
    exception so callers depend on this client's own exception type,
    not on `requests` being the transport forever."""


class ModelWatchClient:
    """
    Usage:
        client = ModelWatchClient("http://localhost:8000")
        client.register_model("my-rag-app", "My RAG App", "rag", baseline_events)
        result = client.check("my-rag-app", new_events)
        if result["is_drifted"]:
            ...

    Every method raises ModelWatchError on a network failure or a
    non-2xx response; a 404 (e.g. checking a model that was never
    registered) is not special-cased -- callers that need to
    distinguish "not registered yet" from "server unreachable" should
    catch ModelWatchError and inspect `.status_code`.
    """

    def __init__(self, base_url: str, session: requests.Session | None = None, timeout: float = 30):
        self.base_url = base_url.rstrip("/")
        self._session = session or requests.Session()
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            res = self._session.request(method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs)
            res.raise_for_status()
        except requests.exceptions.RequestException as e:
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            error = ModelWatchError(f"{method} {path} failed: {e}")
            error.status_code = status_code
            raise error from e
        return res.json() if res.content else {}

    # -- models -------------------------------------------------------------

    def register_model(
        self,
        model_id: str,
        name: str,
        adapter_name: str,
        baseline_data: Any,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Registers a new monitored model and builds its baseline.
        adapter_name must be one already known to the server (e.g.
        "rag", "classifier", "llm", "live_telemetry") -- this client
        does not ship its own adapters."""
        return self._request(
            "POST", "/models",
            json={
                "model_id": model_id, "name": name, "adapter_name": adapter_name,
                "baseline_data": baseline_data, "config": config or {},
            },
        )

    def is_registered(self, model_id: str) -> bool:
        res = self._session.get(f"{self.base_url}/models/{model_id}", timeout=self.timeout)
        return res.status_code == 200

    def get_model(self, model_id: str) -> dict[str, Any]:
        return self._request("GET", f"/models/{model_id}")

    def list_models(self) -> list[dict[str, Any]]:
        return self._request("GET", "/models")  # type: ignore[return-value]

    # -- checks ---------------------------------------------------------------

    def check(self, model_id: str, new_data: Any) -> dict[str, Any]:
        """Runs a drift/quality check for `model_id` against its current
        baseline. Returns the same shape as the dashboard's own check
        calls: drift_score, quality_score, is_drifted, per-signal
        breakdown, health_state."""
        return self._request("POST", f"/models/{model_id}/check", json={"new_data": new_data})

    def get_health(self, model_id: str) -> dict[str, Any]:
        return self._request("GET", f"/models/{model_id}/health")

    def get_history(self, model_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        params = {"limit": limit} if limit is not None else {}
        return self._request("GET", f"/models/{model_id}/history", params=params)  # type: ignore[return-value]

    def get_alerts(self, model_id: str | None = None, active_only: bool = False) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"active_only": str(active_only).lower()}
        if model_id is not None:
            params["model_id"] = model_id
        return self._request("GET", "/alerts", params=params)  # type: ignore[return-value]

    def retrain(self, model_id: str, new_training_data: Any) -> dict[str, Any]:
        return self._request("POST", f"/models/{model_id}/retrain", json={"new_training_data": new_training_data})

    # -- RAG X-Ray traces -------------------------------------------------

    def record_trace(self, trace_id: str, model_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Sends one full per-request trace to the RAG X-Ray. See
        sanad/features/trace.py for the shape `data` is expected to
        have if you want it to render in the dashboard's X-Ray panel
        (question, retrieval, claims, scores) -- this client doesn't
        validate that shape, it only forwards it."""
        return self._request("POST", "/traces", json={"trace_id": trace_id, "model_id": model_id, "data": data})

    def get_trace(self, trace_id: str) -> dict[str, Any]:
        return self._request("GET", f"/traces/{trace_id}")

    def list_traces(self, model_id: str | None = None, limit: int = 50, grounded: bool | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if model_id is not None:
            params["model_id"] = model_id
        if grounded is not None:
            params["grounded"] = str(grounded).lower()
        return self._request("GET", "/traces", params=params)  # type: ignore[return-value]

    def diagnose_trace(self, trace_id: str) -> dict[str, Any]:
        return self._request("GET", f"/traces/{trace_id}/diagnosis")

    # -- experiments ------------------------------------------------------

    def record_experiment(
        self, name: str, kind: str, config: dict[str, Any], results: dict[str, Any], status: str = "completed"
    ) -> dict[str, Any]:
        return self._request(
            "POST", "/experiments",
            json={"name": name, "kind": kind, "config": config, "results": results, "status": status},
        )
