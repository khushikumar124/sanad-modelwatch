"""In-process background job execution for slow, blocking work.

Originally lived at `sanad/jobs.py` for Sanad's own LLM-bound requests
(obligation extraction and the Review synthesis it feeds can take
anywhere from ~10s to several minutes on a local model -- running that
synchronously inside an HTTP request means the browser tab sits frozen
the whole time, and a slow request risks hitting proxy/client timeouts
entirely). A route hands the work to a thread pool and returns a job_id
immediately; the caller polls GET /api/jobs/{job_id} until it's done.

Moved here because modelwatch/api/app.py's Drift Lab endpoints reuse the
exact same class -- and it has zero Sanad-specific logic (stdlib only),
so importing it from inside Sanad's package was a real, needless
coupling: ModelWatch's own API could not even start without Sanad's
package being importable, despite this class never touching anything
Sanad-specific. This module has no product-specific logic at all; either
app can depend on it without depending on the other.

Deliberately a ThreadPoolExecutor, not Celery/Redis/RQ: the work here is
blocking I/O (an HTTP call to a local model), not CPU-bound, so a thread
pool is sufficient, and it adds no new infrastructure to either project's
explicit design goal of staying local-first and single-process. This is
the right tool for one process; the moment this needs to run across
multiple worker processes or survive a process restart, swap this module
for a real task queue (Celery+Redis, RQ, or Arq) -- the call sites
(`submit` / `get`) are the seam to swap behind, not something that would
need to change at every call site.

Jobs are kept in memory, capped, and evicted oldest-first -- a
long-running server shouldn't accumulate unbounded state, and a job's
result only needs to survive long enough for the client that started it
to poll for it.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

_MAX_JOBS = 200
_MAX_WORKERS = 4

PENDING = "pending"
RUNNING = "running"
DONE = "done"
ERROR = "error"


@dataclass
class Job:
    job_id: str
    kind: str
    status: str = PENDING
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "elapsed_seconds": round((self.finished_at or time.time()) - self.created_at, 1),
        }


class JobManager:
    def __init__(self, max_workers: int = _MAX_WORKERS, max_jobs: int = _MAX_JOBS):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="job")
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()
        self._lock = threading.Lock()
        self._max_jobs = max_jobs

    def submit(self, kind: str, fn: Callable[[], dict[str, Any]]) -> str:
        job_id = uuid.uuid4().hex
        job = Job(job_id=job_id, kind=kind)
        with self._lock:
            self._jobs[job_id] = job
            while len(self._jobs) > self._max_jobs:
                self._jobs.popitem(last=False)

        def _run() -> None:
            with self._lock:
                job.status = RUNNING
            try:
                result = fn()
                with self._lock:
                    job.result = result
                    job.status = DONE
                    job.finished_at = time.time()
            except Exception as e:  # noqa: BLE001 -- a job's failure must never crash the pool
                logger.exception("job failed", extra={"job_id": job_id, "kind": kind})
                with self._lock:
                    job.error = str(e)
                    job.status = ERROR
                    job.finished_at = time.time()

        self._executor.submit(_run)
        return job_id

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)


jobs = JobManager()
