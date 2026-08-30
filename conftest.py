"""Makes modelwatch-client importable during test collection regardless
of whether its editable pip install is actually working.

`pip install -e ./modelwatch-client` normally handles this via a .pth
file in site-packages, but on this project's dev machine (macOS), pip
installs into this venv have repeatedly triggered a filesystem quirk
where that .pth file gets the UF_HIDDEN flag set on it (visible via
`ls -lO`), which makes Python's site module skip it entirely --
`import modelwatch_client` then fails everywhere, including outside
pytest, until someone runs `chflags nohidden` on the file by hand. This
isn't a modelwatch-client bug; inserting the package directly onto
sys.path here means the test suite doesn't depend on that install
mechanism working at all.
"""
import sys
from pathlib import Path

_modelwatch_client_src = Path(__file__).resolve().parent / "modelwatch-client"
if str(_modelwatch_client_src) not in sys.path:
    sys.path.insert(0, str(_modelwatch_client_src))
