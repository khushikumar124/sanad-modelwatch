"""Upload validation beyond trusting the filename.

api/app.py's SUPPORTED_EXTENSIONS check only looks at what a client
*claims* the file is -- a request can name any bytes "contract.pdf".
This module checks what's actually true about the bytes:

1. Size, before anything else touches the file (extraction, embedding,
   storage) -- rejecting a 2GB upload should cost nothing.
2. Magic bytes, so a file claiming to be a PDF but not starting with the
   PDF signature is rejected before ingestion ever runs PyMuPDF/Tesseract
   over it.
3. A malware scan via ClamAV's `clamd` daemon, if one is configured
   (SANAD_CLAMAV_HOST). This is the one check that's optional rather
   than always-on: a local dev machine has no reason to run a ClamAV
   daemon just to try this app, so with clamav_host unset, validation
   simply skips this check -- it does not fail closed and it does not
   pretend to have scanned something it didn't. If clamav_host *is* set
   but the daemon can't be reached, that's treated as a real error
   (fail closed), not silently skipped, since a configured-but-broken
   scanner failing open would be worse than not having one.
"""
from __future__ import annotations

import logging

from sanad.config import config

logger = logging.getLogger(__name__)


class UploadValidationError(Exception):
    pass


# (extension, required leading bytes) -- checked as a prefix match.
# TIFF has two valid byte orders, so it gets two entries under the same
# extension; a match against either is accepted.
_MAGIC_BYTES: dict[str, list[bytes]] = {
    ".pdf": [b"%PDF-"],
    ".png": [b"\x89PNG\r\n\x1a\n"],
    ".jpg": [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
    ".tiff": [b"II*\x00", b"MM\x00*"],
    ".tif": [b"II*\x00", b"MM\x00*"],
    ".bmp": [b"BM"],
}


def _check_size(data: bytes) -> None:
    max_bytes = config.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise UploadValidationError(
            f"file is {len(data) / (1024 * 1024):.1f}MB, exceeding the {config.max_upload_mb}MB limit"
        )
    if len(data) == 0:
        raise UploadValidationError("uploaded file is empty")


def _check_magic_bytes(ext: str, data: bytes) -> None:
    signatures = _MAGIC_BYTES.get(ext)
    if signatures is None:
        return  # an extension outside this table isn't this function's job to police
    if not any(data.startswith(sig) for sig in signatures):
        raise UploadValidationError(
            f"file content doesn't match a valid {ext} file (failed signature check)"
        )


def _check_malware(data: bytes) -> None:
    if not config.clamav_host:
        return
    import io

    import clamd  # deferred: only needed when a daemon is actually configured

    try:
        client = clamd.ClamdNetworkSocket(host=config.clamav_host, port=config.clamav_port, timeout=10)
        result = client.instream(io.BytesIO(data))
    except (clamd.ConnectionError, clamd.BufferTooLongError, OSError) as e:
        raise UploadValidationError(f"malware scan unavailable: {e}") from e

    status, signature = result.get("stream", (None, None))
    if status == "FOUND":
        logger.warning("upload rejected by malware scan", extra={"signature": signature})
        raise UploadValidationError(f"file rejected by malware scan (matched signature: {signature})")


def validate_upload(ext: str, data: bytes) -> None:
    """Raises UploadValidationError with a client-safe message if the
    upload should be rejected. Order matters: cheapest and most decisive
    checks first, so an oversized or spoofed file never reaches the
    (comparatively expensive) malware scan."""
    _check_size(data)
    _check_magic_bytes(ext, data)
    _check_malware(data)
