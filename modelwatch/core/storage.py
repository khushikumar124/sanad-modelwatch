"""SQLite persistence for ModelWatch.

This module is a thin data-access layer: it knows table schemas and does
CRUD, but holds no drift/quality logic. That logic lives in engine.py and
in the adapters. Baselines and per-signal breakdowns are stored as JSON
blobs since their shape is adapter-specific and opaque to storage.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    model_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    adapter_name TEXT NOT NULL,
    current_version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    config_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS baselines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL REFERENCES models(model_id),
    version INTEGER NOT NULL,
    data_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL REFERENCES models(model_id),
    version INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    drift_score REAL NOT NULL,
    quality_score REAL,
    is_drifted INTEGER NOT NULL,
    signals_json TEXT NOT NULL,
    statistics_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL REFERENCES models(model_id),
    run_id INTEGER NOT NULL REFERENCES runs(id),
    created_at TEXT NOT NULL,
    message TEXT NOT NULL,
    resolved INTEGER NOT NULL DEFAULT 0,
    resolved_at TEXT
);

CREATE TABLE IF NOT EXISTS versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL REFERENCES models(model_id),
    version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    reason TEXT NOT NULL
);

-- One row per model: the alert-hysteresis state machine (see
-- modelwatch/core/health.py). consecutive_drifted/consecutive_clean count
-- consecutive check_drift() runs, reset to 0 on the opposite outcome, so a
-- single unlucky (or lucky) batch can't flip the health state on its own.
CREATE TABLE IF NOT EXISTS model_health (
    model_id TEXT PRIMARY KEY REFERENCES models(model_id),
    state TEXT NOT NULL DEFAULT 'healthy',
    consecutive_drifted INTEGER NOT NULL DEFAULT 0,
    consecutive_clean INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

-- One row per experiment/benchmark run (Drift Lab scenario, benchmark
-- comparison, ablation study, ...). Free-form config/results JSON, like
-- baselines -- storage doesn't interpret them, just persists and lists
-- them so a later reader (or scripts/run_benchmark.py re-running with
-- the same config) has a record of what was actually run and measured.
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    created_at TEXT NOT NULL,
    config_json TEXT NOT NULL DEFAULT '{}',
    results_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'completed'
);

-- Full per-request RAG traces (question, answer, evidence, claim
-- verification -- see sanad/features/trace.py), reported only when the
-- sending app opts in (Sanad: SANAD_TELEMETRY_FULL_TRACE). Deliberately
-- separate from `runs`: a trace is one request's raw pipeline detail for
-- the RAG X-Ray to browse/diagnose, not a statistical drift result, and
-- keeping it out of runs/signals means every existing adapter/query
-- against those tables is unaffected by whether tracing is on.
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT NOT NULL,
    model_id TEXT NOT NULL REFERENCES models(model_id),
    created_at TEXT NOT NULL,
    data_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_model ON alerts(model_id, resolved);
CREATE INDEX IF NOT EXISTS idx_baselines_model ON baselines(model_id, version);
CREATE INDEX IF NOT EXISTS idx_traces_model ON traces(model_id, created_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelAlreadyExistsError(Exception):
    pass


class ModelNotFoundError(Exception):
    pass


class Storage:
    """SQLite-backed storage for models, baselines, runs, alerts, versions.

    One Storage instance owns one sqlite3 connection guarded by a lock --
    sqlite3 connections aren't safe to share across threads, and FastAPI's
    default threadpool executor can call in from multiple threads.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._cursor() as cur:
            cur.executescript(_SCHEMA)
            # Add columns introduced after a database was first created.
            # SQLite has no "ADD COLUMN IF NOT EXISTS", so check first --
            # otherwise an existing modelwatch.db fails to open.
            existing = {row[1] for row in cur.execute("PRAGMA table_info(runs)")}
            if "statistics_json" not in existing:
                cur.execute("ALTER TABLE runs ADD COLUMN statistics_json TEXT NOT NULL DEFAULT '{}'")

            existing_models = {row[1] for row in cur.execute("PRAGMA table_info(models)")}
            if "config_json" not in existing_models:
                cur.execute("ALTER TABLE models ADD COLUMN config_json TEXT NOT NULL DEFAULT '{}'")

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    def close(self) -> None:
        self._conn.close()

    # -- models ---------------------------------------------------------

    def create_model(
        self, model_id: str, name: str, adapter_name: str, config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """config is a free-form registry record of what this model *is*
        at the time of registration -- e.g. for a RAG model: embedding
        model, chunk size, top-k, prompt version, dataset version. It is
        opaque to storage (like a baseline's data_json) but, unlike a
        baseline, it describes the model's configuration rather than a
        statistical snapshot, and is meant to be read by a human comparing
        two registrations, not by an adapter."""
        with self._cursor() as cur:
            cur.execute("SELECT 1 FROM models WHERE model_id = ?", (model_id,))
            if cur.fetchone() is not None:
                raise ModelAlreadyExistsError(f"model '{model_id}' already registered")
            created_at = _now()
            cur.execute(
                "INSERT INTO models (model_id, name, adapter_name, current_version, created_at, config_json)"
                " VALUES (?, ?, ?, 1, ?, ?)",
                (model_id, name, adapter_name, created_at, json.dumps(config or {})),
            )
        logger.info("model registered", extra={"model_id": model_id, "adapter_name": adapter_name})
        return self.get_model(model_id)

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
            row = cur.fetchone()
            if row is None:
                return None
            d = dict(row)
            d["config"] = json.loads(d.pop("config_json", None) or "{}")
            return d

    def list_models(self) -> list[dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM models ORDER BY created_at")
            rows = []
            for row in cur.fetchall():
                d = dict(row)
                d["config"] = json.loads(d.pop("config_json", None) or "{}")
                rows.append(d)
            return rows

    def set_current_version(self, model_id: str, version: int) -> None:
        with self._cursor() as cur:
            cur.execute(
                "UPDATE models SET current_version = ? WHERE model_id = ?", (version, model_id)
            )

    # -- baselines --------------------------------------------------------

    def save_baseline(self, model_id: str, version: int, data: dict[str, Any]) -> int:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO baselines (model_id, version, data_json, created_at) VALUES (?, ?, ?, ?)",
                (model_id, version, json.dumps(data), _now()),
            )
            return cur.lastrowid

    def get_latest_baseline(self, model_id: str) -> dict[str, Any] | None:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM baselines WHERE model_id = ? ORDER BY version DESC, id DESC LIMIT 1",
                (model_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            result = dict(row)
            result["data"] = json.loads(result.pop("data_json"))
            return result

    # -- runs ---------------------------------------------------------------

    def save_run(
        self,
        model_id: str,
        version: int,
        drift_score: float,
        quality_score: float | None,
        is_drifted: bool,
        signals: list[dict[str, Any]],
        statistics: dict[str, Any] | None = None,
    ) -> int:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO runs (model_id, version, timestamp, drift_score, quality_score,"
                " is_drifted, signals_json, statistics_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    model_id,
                    version,
                    _now(),
                    drift_score,
                    quality_score,
                    int(is_drifted),
                    json.dumps(signals),
                    json.dumps(statistics or {}),
                ),
            )
            return cur.lastrowid

    def get_history(self, model_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        with self._cursor() as cur:
            query = "SELECT * FROM runs WHERE model_id = ? ORDER BY timestamp DESC"
            params: tuple[Any, ...] = (model_id,)
            if limit is not None:
                query += " LIMIT ?"
                params = (model_id, limit)
            cur.execute(query, params)
            rows = []
            for row in cur.fetchall():
                d = dict(row)
                d["is_drifted"] = bool(d["is_drifted"])
                d["signals"] = json.loads(d.pop("signals_json"))
                d["statistics"] = json.loads(d.pop("statistics_json", None) or "{}")
                rows.append(d)
            return rows

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
            row = cur.fetchone()
            if row is None:
                return None
            d = dict(row)
            d["is_drifted"] = bool(d["is_drifted"])
            d["signals"] = json.loads(d.pop("signals_json"))
            d["statistics"] = json.loads(d.pop("statistics_json", None) or "{}")
            return d

    # -- alerts ---------------------------------------------------------------

    def create_alert(self, model_id: str, run_id: int, message: str) -> int:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO alerts (model_id, run_id, created_at, message, resolved)"
                " VALUES (?, ?, ?, ?, 0)",
                (model_id, run_id, _now(), message),
            )
            return cur.lastrowid

    def get_alerts(self, model_id: str | None = None, active_only: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM alerts WHERE 1=1"
        params: list[Any] = []
        if model_id is not None:
            query += " AND model_id = ?"
            params.append(model_id)
        if active_only:
            query += " AND resolved = 0"
        query += " ORDER BY created_at DESC"
        with self._cursor() as cur:
            cur.execute(query, params)
            rows = []
            for row in cur.fetchall():
                d = dict(row)
                d["resolved"] = bool(d["resolved"])
                rows.append(d)
            return rows

    def resolve_alerts_for_model(self, model_id: str) -> int:
        with self._cursor() as cur:
            cur.execute(
                "UPDATE alerts SET resolved = 1, resolved_at = ? WHERE model_id = ? AND resolved = 0",
                (_now(), model_id),
            )
            return cur.rowcount

    # -- versions ---------------------------------------------------------------

    def add_version(self, model_id: str, version: int, reason: str) -> int:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO versions (model_id, version, created_at, reason) VALUES (?, ?, ?, ?)",
                (model_id, version, _now(), reason),
            )
            return cur.lastrowid

    def get_versions(self, model_id: str) -> list[dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM versions WHERE model_id = ? ORDER BY version", (model_id,)
            )
            return [dict(row) for row in cur.fetchall()]

    # -- health state (alert hysteresis) -------------------------------------

    def get_health(self, model_id: str) -> dict[str, Any]:
        """Never returns None. A model with no health row yet is
        healthy with a clean streak of zero -- UNLESS it already has an
        unresolved alert (a model registered before the model_health
        table existed, upgraded in place): then it starts as degraded,
        so the health state and the alert it's already carrying agree
        with each other instead of showing "healthy" next to an open
        incident."""
        with self._cursor() as cur:
            cur.execute("SELECT * FROM model_health WHERE model_id = ?", (model_id,))
            row = cur.fetchone()
            if row is not None:
                return dict(row)

        # self._cursor() holds a non-reentrant lock -- get_alerts() below
        # must run after that `with` block has exited, not inside it.
        has_open_alert = self.get_alerts(model_id=model_id, active_only=True) != []
        return {
            "model_id": model_id,
            "state": "degraded" if has_open_alert else "healthy",
            "consecutive_drifted": 1 if has_open_alert else 0,
            "consecutive_clean": 0,
            "updated_at": None,
        }

    # -- experiments ------------------------------------------------------

    def create_experiment(
        self, name: str, kind: str, config: dict[str, Any], results: dict[str, Any], status: str = "completed"
    ) -> int:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO experiments (name, kind, created_at, config_json, results_json, status)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (name, kind, _now(), json.dumps(config), json.dumps(results), status),
            )
            return cur.lastrowid

    def get_experiment(self, experiment_id: int) -> dict[str, Any] | None:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
            row = cur.fetchone()
            if row is None:
                return None
            d = dict(row)
            d["config"] = json.loads(d.pop("config_json"))
            d["results"] = json.loads(d.pop("results_json"))
            return d

    def list_experiments(self, kind: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM experiments"
        params: list[Any] = []
        if kind is not None:
            query += " WHERE kind = ?"
            params.append(kind)
        query += " ORDER BY created_at DESC"
        with self._cursor() as cur:
            cur.execute(query, params)
            rows = []
            for row in cur.fetchall():
                d = dict(row)
                d["config"] = json.loads(d.pop("config_json"))
                d["results"] = json.loads(d.pop("results_json"))
                rows.append(d)
            return rows

    # -- traces (RAG X-Ray) ------------------------------------------------

    def create_trace(self, trace_id: str, model_id: str, data: dict[str, Any]) -> int:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO traces (trace_id, model_id, created_at, data_json) VALUES (?, ?, ?, ?)",
                (trace_id, model_id, _now(), json.dumps(data)),
            )
            return cur.lastrowid

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM traces WHERE trace_id = ?", (trace_id,))
            row = cur.fetchone()
            if row is None:
                return None
            d = dict(row)
            d["data"] = json.loads(d.pop("data_json"))
            return d

    def list_traces(
        self, model_id: str | None = None, limit: int = 50, grounded: bool | None = None
    ) -> list[dict[str, Any]]:
        """Newest first. `grounded` filters on the trace's own recorded
        groundedness (read out of data_json), used by the RAG X-Ray to
        jump straight to refusals when triaging "why did this go wrong"."""
        query = "SELECT * FROM traces"
        params: list[Any] = []
        if model_id is not None:
            query += " WHERE model_id = ?"
            params.append(model_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit if grounded is None else max(limit * 4, limit))  # overfetch when filtering in Python
        with self._cursor() as cur:
            cur.execute(query, params)
            rows = []
            for row in cur.fetchall():
                d = dict(row)
                d["data"] = json.loads(d.pop("data_json"))
                if grounded is not None and d["data"].get("grounded") != grounded:
                    continue
                rows.append(d)
                if len(rows) >= limit:
                    break
            return rows

    def set_health(
        self, model_id: str, state: str, consecutive_drifted: int, consecutive_clean: int
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO model_health (model_id, state, consecutive_drifted, consecutive_clean, updated_at)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(model_id) DO UPDATE SET"
                " state = excluded.state,"
                " consecutive_drifted = excluded.consecutive_drifted,"
                " consecutive_clean = excluded.consecutive_clean,"
                " updated_at = excluded.updated_at",
                (model_id, state, consecutive_drifted, consecutive_clean, _now()),
            )
