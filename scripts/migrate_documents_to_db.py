#!/usr/bin/env python3
"""One-time migration: import documents from the old JSON-sidecar
registry (one `{doc_id}.json` file per document in SANAD_UPLOAD_DIR)
into the new database-backed registry (sanad/db.py).

Needed because upgrading to the database-backed registry otherwise
silently "loses" every document uploaded before the upgrade: the new
code never reads the old sidecar format, so GET /api/documents would
return an empty list even though the files (and their ChromaDB chunks)
are still on disk. Caught by running this repo's own live server after
the migration and finding exactly that -- see docs/architecture.md.

Idempotent: an already-migrated doc_id is skipped, not duplicated, so
running this twice (or against a partially-migrated upload dir) is
safe.

Usage:
    python scripts/migrate_documents_to_db.py
    python scripts/migrate_documents_to_db.py --upload-dir sanad_uploads --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanad import db  # noqa: E402
from sanad.config import config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload-dir", default=config.upload_dir)
    parser.add_argument("--dry-run", action="store_true", help="report what would be migrated without writing")
    args = parser.parse_args()

    upload_dir = Path(args.upload_dir)
    if not upload_dir.is_dir():
        print(f"No upload directory at {upload_dir} -- nothing to migrate.")
        return

    sidecars = sorted(upload_dir.glob("*.json"))
    if not sidecars:
        print(f"No JSON sidecar files found in {upload_dir} -- nothing to migrate.")
        return

    migrated, skipped, failed = 0, 0, 0
    for sidecar in sidecars:
        doc_id = sidecar.stem
        try:
            data = json.loads(sidecar.read_text())
        except (OSError, ValueError) as e:
            print(f"  FAILED to read {sidecar.name}: {e}")
            failed += 1
            continue

        if db.get_document(doc_id) is not None:
            print(f"  skip  {doc_id} ({data.get('filename', '?')}) -- already in the database")
            skipped += 1
            continue

        record = db.DocumentRecord(
            doc_id=doc_id,
            filename=data.get("filename", sidecar.stem),
            contract_type=data.get("contract_type"),
            chunk_count=data.get("chunk_count", 0),
            used_ocr=data.get("used_ocr", False),
            uploaded_at=data.get("uploaded_at", ""),
            text=data.get("text", ""),
            source_path=data.get("source_path", ""),
        )
        if args.dry_run:
            print(f"  would migrate  {doc_id} ({record.filename})")
        else:
            db.save_document(record)
            print(f"  migrated  {doc_id} ({record.filename})")
        migrated += 1

    verb = "Would migrate" if args.dry_run else "Migrated"
    print(f"\n{verb} {migrated}, skipped {skipped} already-present, {failed} failed.")
    if not args.dry_run and migrated:
        print("Sidecar .json files were left in place -- safe to delete once you've confirmed the app sees these documents.")


if __name__ == "__main__":
    main()
