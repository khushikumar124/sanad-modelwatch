"""Tests for scripts/migrate_documents_to_db.py's migration logic.

Imports the script as a module (sys.path already has the repo root from
conftest/pytest's rootdir insertion) to test its behavior directly
rather than shelling out to it -- faster, and gives real assertions
instead of parsing stdout."""
import json
import subprocess
import sys
from pathlib import Path

from sanad import db


def _write_sidecar(upload_dir: Path, doc_id: str, **fields) -> None:
    defaults = dict(
        filename="lease.pdf", contract_type="rental", chunk_count=5,
        used_ocr=False, uploaded_at="2026-01-01T00:00:00+00:00",
        text="contract text", source_path=str(upload_dir / f"{doc_id}.pdf"),
    )
    defaults.update(fields)
    (upload_dir / f"{doc_id}.json").write_text(json.dumps(defaults))


def _run_migration(upload_dir: Path, database_url: str, dry_run: bool = False) -> subprocess.CompletedProcess:
    repo_root = Path(__file__).resolve().parents[2]
    args = [sys.executable, str(repo_root / "scripts" / "migrate_documents_to_db.py"), "--upload-dir", str(upload_dir)]
    if dry_run:
        args.append("--dry-run")
    env = {"SANAD_DATABASE_URL": database_url, "PATH": "/usr/bin:/bin"}
    return subprocess.run(args, capture_output=True, text=True, env=env, cwd=repo_root)


def test_migrates_sidecar_files_into_the_database(tmp_path):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    _write_sidecar(upload_dir, "doc-1", filename="a.pdf")
    _write_sidecar(upload_dir, "doc-2", filename="b.pdf")
    database_url = f"sqlite:///{tmp_path}/migrated.db"

    result = _run_migration(upload_dir, database_url)
    assert result.returncode == 0, result.stderr
    assert "Migrated 2" in result.stdout

    db.reset_engine()
    from dataclasses import replace
    from sanad.config import config as base_config
    import sanad.db as db_module
    db_module.config = replace(base_config, database_url=database_url)

    docs = {d.doc_id for d in db.list_documents()}
    assert docs == {"doc-1", "doc-2"}
    db.reset_engine()


def test_dry_run_does_not_write_to_the_database(tmp_path):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    _write_sidecar(upload_dir, "doc-1")
    database_url = f"sqlite:///{tmp_path}/migrated.db"

    result = _run_migration(upload_dir, database_url, dry_run=True)
    assert result.returncode == 0, result.stderr
    assert "Would migrate 1" in result.stdout
    assert "migrated  doc-1" not in result.stdout  # only the "would migrate" line, not a real one

    db.reset_engine()
    from dataclasses import replace
    from sanad.config import config as base_config
    import sanad.db as db_module
    db_module.config = replace(base_config, database_url=database_url)

    assert db.list_documents() == []
    db.reset_engine()


def test_already_migrated_document_is_skipped_not_duplicated(tmp_path):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    _write_sidecar(upload_dir, "doc-1")
    database_url = f"sqlite:///{tmp_path}/migrated.db"

    first = _run_migration(upload_dir, database_url)
    assert "Migrated 1" in first.stdout

    second = _run_migration(upload_dir, database_url)
    assert "skipped 1 already-present" in second.stdout
    assert "Migrated 0" in second.stdout


def test_missing_upload_dir_reports_nothing_to_migrate(tmp_path):
    database_url = f"sqlite:///{tmp_path}/migrated.db"
    result = _run_migration(tmp_path / "does-not-exist", database_url)
    assert result.returncode == 0
    assert "nothing to migrate" in result.stdout.lower()


def test_empty_upload_dir_reports_nothing_to_migrate(tmp_path):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    database_url = f"sqlite:///{tmp_path}/migrated.db"
    result = _run_migration(upload_dir, database_url)
    assert result.returncode == 0
    assert "nothing to migrate" in result.stdout.lower()


def test_corrupt_sidecar_is_reported_as_failed_not_a_crash(tmp_path):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    (upload_dir / "bad-doc.json").write_text("not json at all")
    _write_sidecar(upload_dir, "good-doc")
    database_url = f"sqlite:///{tmp_path}/migrated.db"

    result = _run_migration(upload_dir, database_url)
    assert result.returncode == 0, result.stderr
    assert "FAILED" in result.stdout
    assert "Migrated 1" in result.stdout  # the good one still goes through
