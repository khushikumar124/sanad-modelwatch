"""Object storage for the original uploaded file.

This is deliberately separate from sanad/db.py: db.py stores the
*extracted text* of every document (what retrieval, obligations,
coverage etc. all read), which already survives regardless of this
module. This module only stores the original file bytes -- needed so
a document can be re-ingested (e.g. after a chunking/OCR change) or
downloaded, without asking the user to re-upload.

Two backends behind one interface, selected by config.storage_backend:
- "local" (default): writes straight to config.upload_dir. Needs
  nothing installed, correct for a single machine.
- "s3": an S3-compatible bucket (real AWS, or MinIO/LocalStack/R2 via
  config.s3_endpoint_url). For the moment local disk isn't durable or
  shared across instances. boto3 is only imported when this backend is
  actually selected, so it's not a hard dependency for the local case.

Both backends return an opaque `locator` string from save(), which is
what gets persisted in db.py's source_path column -- callers should not
parse it. open_local() is a context manager because the S3 backend has
to download to a temp file for callers (like ingest_document) that need
a real filesystem path; the local backend just yields the real path
with nothing to clean up.
"""
from __future__ import annotations

import tempfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sanad.config import config


class ObjectStore(ABC):
    @abstractmethod
    def save(self, doc_id: str, ext: str, data: bytes) -> str:
        """Store the file's bytes and return a locator for later access."""

    @abstractmethod
    @contextmanager
    def open_local(self, locator: str) -> Iterator[str]:
        """Yield a real filesystem path to the file's contents."""

    @abstractmethod
    def delete(self, locator: str) -> None:
        """Remove the stored file. Safe to call if it's already gone."""


class LocalDiskStore(ObjectStore):
    def __init__(self, upload_dir: str | None = None):
        self._upload_dir = Path(upload_dir if upload_dir is not None else config.upload_dir)

    def save(self, doc_id: str, ext: str, data: bytes) -> str:
        self._upload_dir.mkdir(parents=True, exist_ok=True)
        dest = self._upload_dir / f"{doc_id}{ext}"
        dest.write_bytes(data)
        return str(dest)

    @contextmanager
    def open_local(self, locator: str) -> Iterator[str]:
        yield locator

    def delete(self, locator: str) -> None:
        Path(locator).unlink(missing_ok=True)


class S3Store(ObjectStore):
    """S3-compatible backend. Talks to real AWS S3 by default; point
    config.s3_endpoint_url at MinIO/LocalStack/R2 (or moto in tests) to
    use something else that speaks the S3 API."""

    _LOCATOR_PREFIX = "s3://"

    def __init__(
        self,
        bucket: str | None = None,
        region: str | None = None,
        endpoint_url: str | None = None,
    ):
        self._bucket = bucket if bucket is not None else config.s3_bucket
        if not self._bucket:
            raise ValueError("SANAD_S3_BUCKET must be set to use the s3 storage backend")
        self._region = region if region is not None else config.s3_region
        self._endpoint_url = endpoint_url if endpoint_url is not None else config.s3_endpoint_url
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import boto3  # deferred: only needed for this backend

            kwargs = {"region_name": self._region}
            if self._endpoint_url:
                kwargs["endpoint_url"] = self._endpoint_url
            self._client = boto3.client("s3", **kwargs)
        return self._client

    def _key_for(self, doc_id: str, ext: str) -> str:
        return f"{doc_id}{ext}"

    def _locator_for(self, key: str) -> str:
        return f"{self._LOCATOR_PREFIX}{self._bucket}/{key}"

    def _key_from_locator(self, locator: str) -> str:
        without_prefix = locator[len(self._LOCATOR_PREFIX):]
        _bucket, _, key = without_prefix.partition("/")
        return key

    def save(self, doc_id: str, ext: str, data: bytes) -> str:
        key = self._key_for(doc_id, ext)
        self.client.put_object(Bucket=self._bucket, Key=key, Body=data)
        return self._locator_for(key)

    @contextmanager
    def open_local(self, locator: str) -> Iterator[str]:
        key = self._key_from_locator(locator)
        suffix = Path(key).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
        try:
            self.client.download_file(self._bucket, key, tmp_path)
            yield tmp_path
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def delete(self, locator: str) -> None:
        key = self._key_from_locator(locator)
        self.client.delete_object(Bucket=self._bucket, Key=key)


def get_object_store() -> ObjectStore:
    if config.storage_backend == "s3":
        return S3Store()
    if config.storage_backend == "local":
        return LocalDiskStore()
    raise ValueError(f"unknown SANAD_STORAGE_BACKEND '{config.storage_backend}', expected 'local' or 's3'")
