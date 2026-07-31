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
    created_at TEXT NOT NULL
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
    signals_json TEXT NOT NULL
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

CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_model ON alerts(model_id, resolved);
CREATE INDEX IF NOT EXISTS idx_baselines_model ON baselines(model_id, version);
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

    def create_model(self, model_id: str, name: str, adapter_name: str) -> dict[str, Any]:
        with self._cursor() as cur:
            cur.execute("SELECT 1 FROM models WHERE model_id = ?", (model_id,))
            if cur.fetchone() is not None:
                raise ModelAlreadyExistsError(f"model '{model_id}' already registered")
            created_at = _now()
            cur.execute(
                "INSERT INTO models (model_id, name, adapter_name, current_version, created_at)"
                " VALUES (?, ?, ?, 1, ?)",
                (model_id, name, adapter_name, created_at),
            )
        logger.info("model registered", extra={"model_id": model_id, "adapter_name": adapter_name})
        return self.get_model(model_id)

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def list_models(self) -> list[dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM models ORDER BY created_at")
            return [dict(row) for row in cur.fetchall()]

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
    ) -> int:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO runs (model_id, version, timestamp, drift_score, quality_score,"
                " is_drifted, signals_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    model_id,
                    version,
                    _now(),
                    drift_score,
                    quality_score,
                    int(is_drifted),
                    json.dumps(signals),
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
