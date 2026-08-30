"""Tests for sanad/security.py's upload validation.

The ClamAV tests run against a real TCP server implementing clamd's
actual INSTREAM wire protocol (length-prefixed chunks, a single
newline-terminated response line) rather than mocking the `clamd`
client's methods -- same principle as this repo's other tests that use
pytest-httpserver or moto for a real protocol instead of a hand-rolled
stub, just for a protocol neither of those cover."""
import socket
import struct
import threading
from dataclasses import replace

import pytest

from sanad.config import config as base_config
from sanad import security as security_module
from sanad.security import UploadValidationError, validate_upload

_REAL_PDF = b"%PDF-1.4\n%fake but valid header\n...rest of a real pdf..."
_REAL_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20


class _FakeClamd:
    """A minimal real TCP server speaking clamd's INSTREAM protocol:
    reads length-prefixed chunks until a zero-length terminator, then
    replies with one canned newline-terminated response line."""

    def __init__(self, response_line: str):
        self._response_line = response_line
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = self._sock.getsockname()[1]
        self._thread = threading.Thread(target=self._serve_once, daemon=True)
        self._thread.start()

    def _serve_once(self):
        # A single buffered reader for the whole exchange: makefile("rb")
        # can read ahead into its own buffer, so switching to raw
        # conn.recv() partway through (as an earlier version of this did)
        # silently drops already-buffered chunk bytes and the client's
        # final response read then blocks forever waiting for bytes the
        # server thinks it hasn't sent yet.
        conn, _ = self._sock.accept()
        with conn:
            f = conn.makefile("rb")
            f.readline()  # the "nINSTREAM\n" command line
            while True:
                length_bytes = f.read(4)
                if len(length_bytes) < 4:
                    break
                (length,) = struct.unpack("!L", length_bytes)
                if length == 0:
                    break
                f.read(length)
            conn.sendall((self._response_line + "\n").encode("utf-8"))

    def close(self):
        self._sock.close()


@pytest.fixture(autouse=True)
def default_config(monkeypatch):
    monkeypatch.setattr(security_module, "config", replace(base_config, max_upload_mb=1, clamav_host=""))


def test_accepts_a_real_pdf():
    validate_upload(".pdf", _REAL_PDF)  # should not raise


def test_accepts_a_real_png():
    validate_upload(".png", _REAL_PNG)


def test_rejects_a_file_whose_content_does_not_match_its_extension():
    with pytest.raises(UploadValidationError, match="signature check"):
        validate_upload(".pdf", b"this is just plain text, not a pdf")


def test_rejects_an_empty_file():
    with pytest.raises(UploadValidationError, match="empty"):
        validate_upload(".pdf", b"")


def test_rejects_a_file_over_the_size_limit(monkeypatch):
    monkeypatch.setattr(security_module, "config", replace(base_config, max_upload_mb=1, clamav_host=""))
    oversized = _REAL_PDF + b"0" * (2 * 1024 * 1024)
    with pytest.raises(UploadValidationError, match="exceeding"):
        validate_upload(".pdf", oversized)


def test_extension_outside_the_known_table_skips_the_signature_check():
    # .docx isn't in _MAGIC_BYTES -- that's api/app.py's SUPPORTED_EXTENSIONS
    # check's job to reject, not this function's.
    validate_upload(".docx", b"anything at all")


def test_skips_malware_scan_when_clamav_is_not_configured(monkeypatch):
    monkeypatch.setattr(security_module, "config", replace(base_config, max_upload_mb=1, clamav_host=""))
    validate_upload(".pdf", _REAL_PDF)  # no clamd server running anywhere; must not attempt to connect


def test_allows_upload_when_clamav_reports_clean(monkeypatch):
    fake = _FakeClamd("stream: OK")
    try:
        monkeypatch.setattr(
            security_module, "config",
            replace(base_config, max_upload_mb=1, clamav_host="127.0.0.1", clamav_port=fake.port),
        )
        validate_upload(".pdf", _REAL_PDF)
    finally:
        fake.close()


def test_rejects_upload_when_clamav_reports_a_match(monkeypatch):
    fake = _FakeClamd("stream: Eicar-Test-Signature FOUND")
    try:
        monkeypatch.setattr(
            security_module, "config",
            replace(base_config, max_upload_mb=1, clamav_host="127.0.0.1", clamav_port=fake.port),
        )
        with pytest.raises(UploadValidationError, match="malware scan"):
            validate_upload(".pdf", _REAL_PDF)
    finally:
        fake.close()


def test_fails_closed_when_clamav_is_configured_but_unreachable(monkeypatch):
    # Nothing listening on this port -- a configured-but-broken scanner
    # must not silently let uploads through.
    monkeypatch.setattr(
        security_module, "config",
        replace(base_config, max_upload_mb=1, clamav_host="127.0.0.1", clamav_port=1),
    )
    with pytest.raises(UploadValidationError, match="malware scan unavailable"):
        validate_upload(".pdf", _REAL_PDF)
