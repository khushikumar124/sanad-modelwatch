"""Tests for sanad/storage.py's object-storage abstraction.

The S3 backend is tested against moto's mock S3 -- a real implementation
of the S3 HTTP API, not a hand-rolled stub -- so these tests exercise the
actual boto3 client/request path, just against an in-memory bucket
instead of real AWS."""
import boto3
import pytest
from moto import mock_aws

from sanad.storage import LocalDiskStore, S3Store, get_object_store


def test_local_disk_store_round_trips_bytes(tmp_path):
    store = LocalDiskStore(upload_dir=str(tmp_path))
    locator = store.save("doc-1", ".pdf", b"contract bytes")

    with store.open_local(locator) as path:
        assert open(path, "rb").read() == b"contract bytes"


def test_local_disk_store_delete_removes_the_file(tmp_path):
    store = LocalDiskStore(upload_dir=str(tmp_path))
    locator = store.save("doc-1", ".pdf", b"data")
    assert (tmp_path / "doc-1.pdf").exists()

    store.delete(locator)
    assert not (tmp_path / "doc-1.pdf").exists()


def test_local_disk_store_delete_is_safe_when_already_gone(tmp_path):
    store = LocalDiskStore(upload_dir=str(tmp_path))
    store.delete(str(tmp_path / "never-existed.pdf"))  # should not raise


def test_local_disk_store_creates_upload_dir_if_missing(tmp_path):
    missing_dir = tmp_path / "not-yet-created"
    store = LocalDiskStore(upload_dir=str(missing_dir))
    store.save("doc-1", ".pdf", b"data")
    assert (missing_dir / "doc-1.pdf").exists()


@mock_aws
def test_s3_store_round_trips_bytes():
    boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="sanad-test-bucket")
    store = S3Store(bucket="sanad-test-bucket", region="us-east-1")

    locator = store.save("doc-1", ".pdf", b"contract bytes")
    assert locator == "s3://sanad-test-bucket/doc-1.pdf"

    with store.open_local(locator) as path:
        assert open(path, "rb").read() == b"contract bytes"


@mock_aws
def test_s3_store_delete_removes_the_object():
    boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="sanad-test-bucket")
    store = S3Store(bucket="sanad-test-bucket", region="us-east-1")
    locator = store.save("doc-1", ".pdf", b"data")

    store.delete(locator)

    with pytest.raises(Exception):
        with store.open_local(locator):
            pass


@mock_aws
def test_s3_store_open_local_cleans_up_its_temp_file():
    boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="sanad-test-bucket")
    store = S3Store(bucket="sanad-test-bucket", region="us-east-1")
    locator = store.save("doc-1", ".pdf", b"data")

    with store.open_local(locator) as path:
        tmp_path = path
        assert open(tmp_path, "rb").read() == b"data"

    import os
    assert not os.path.exists(tmp_path)


def test_s3_store_requires_a_bucket():
    with pytest.raises(ValueError, match="SANAD_S3_BUCKET"):
        S3Store(bucket="")


def test_get_object_store_returns_local_by_default(tmp_path, monkeypatch):
    from dataclasses import replace
    import sanad.storage as storage_module
    from sanad.config import config as base_config

    monkeypatch.setattr(storage_module, "config", replace(base_config, storage_backend="local"))
    assert isinstance(get_object_store(), LocalDiskStore)


def test_get_object_store_returns_s3_when_configured(monkeypatch):
    from dataclasses import replace
    import sanad.storage as storage_module
    from sanad.config import config as base_config

    monkeypatch.setattr(
        storage_module, "config", replace(base_config, storage_backend="s3", s3_bucket="some-bucket")
    )
    assert isinstance(get_object_store(), S3Store)


def test_get_object_store_rejects_unknown_backend(monkeypatch):
    from dataclasses import replace
    import sanad.storage as storage_module
    from sanad.config import config as base_config

    monkeypatch.setattr(storage_module, "config", replace(base_config, storage_backend="dropbox"))
    with pytest.raises(ValueError, match="unknown SANAD_STORAGE_BACKEND"):
        get_object_store()
