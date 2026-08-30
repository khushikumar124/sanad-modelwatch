"""Optional export of ModelWatch drift-check runs to MLflow.

ModelWatch keeps its own storage and its own dashboard (core/storage.py,
dashboard/index.html) -- this module does not replace either of those,
it's a bridge for a team that already has an MLflow tracking setup
(local file store or a real tracking server) and wants ModelWatch's
drift/quality signals to show up alongside their existing experiment
history there too, instead of only in ModelWatch's own UI.

Deliberately NOT a hard dependency: `mlflow` pulls in a meaningfully
heavier stack (Flask, SQLAlchemy, alembic, ...) than anything else in
modelwatch/requirements.txt, and ModelWatch's own design goal is staying
light and installable without the monitored app's stack (see the
LLMAdapter TF-IDF note in modelwatch/README.md for the same reasoning
applied elsewhere). So: `import mlflow` happens only inside this
module's functions, only when MODELWATCH_MLFLOW_ENABLED is actually on
(see modelwatch/config.py), and any failure here is logged and
swallowed rather than raised -- a drift check must never fail, or fail
differently, because of an optional export succeeding or not. Install
mlflow yourself (`pip install mlflow`) to use this; it is not pulled in
by modelwatch/requirements.txt.

The default tracking URI is a local SQLite-backed store
("sqlite:///mlflow.db"), not mlflow's older plain file store
("file:./mlruns") -- that older backend is in maintenance mode as of
mlflow 3.x and rejects new runs unless MLFLOW_ALLOW_FILE_STORE is set.
Discovered by actually running this against a real local mlflow
instance rather than assuming the file-store example from older mlflow
docs still worked.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def log_run_to_mlflow(
    experiment_name: str,
    tracking_uri: str,
    model_id: str,
    version: int,
    result: dict[str, Any],
    health_state: str | None = None,
) -> None:
    """result: a DriftCheckResult.to_dict()-shaped dict (drift_score,
    quality_score, is_drifted, signals, statistics). Logs one MLflow run
    per call -- one call per modelwatch check, so MLflow's own run
    history becomes a second, independent timeline of the same checks
    ModelWatch's own `runs` table already stores."""
    try:
        import mlflow
    except ImportError:
        logger.warning(
            "MODELWATCH_MLFLOW_ENABLED is on but the mlflow package is not installed -- "
            "run `pip install mlflow` to use this integration. Skipping export for this run."
        )
        return

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"{model_id}-v{version}"):
            mlflow.set_tags({
                "modelwatch.model_id": model_id,
                "modelwatch.version": str(version),
                "modelwatch.is_drifted": str(result.get("is_drifted")),
                **({"modelwatch.health_state": health_state} if health_state else {}),
            })
            mlflow.log_metric("drift_score", float(result.get("drift_score", 0.0)))
            if result.get("quality_score") is not None:
                mlflow.log_metric("quality_score", float(result["quality_score"]))
            for signal in result.get("signals", []):
                # MLflow metric names must be filesystem/URL-safe -- a
                # slash would otherwise be read back as a nested metric
                # path, and a space (e.g. LiveTelemetryAdapter's "refusal
                # rate") makes for an awkward key even where mlflow
                # tolerates it.
                safe_name = str(signal["name"]).replace("/", "_").replace(" ", "_")
                mlflow.log_metric(f"signal.{safe_name}.value", float(signal["value"]))
                mlflow.log_metric(f"signal.{safe_name}.is_drifted", 1.0 if signal["is_drifted"] else 0.0)
    except Exception:
        # Anything from here down is MLflow's own storage/network layer,
        # not ModelWatch's -- a real drift check that already succeeded
        # and was already persisted must not be reported as failed
        # because an optional export to a second system had a problem.
        logger.exception("failed to export modelwatch run to mlflow, continuing without it")
