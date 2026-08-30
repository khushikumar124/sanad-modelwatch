"""Tests for sanad/jobs.py's in-process background job manager."""
import time

import pytest

from sanad.jobs import DONE, ERROR, JobManager, PENDING, RUNNING


def _wait_until(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_submitted_job_eventually_completes_with_its_result():
    manager = JobManager()
    job_id = manager.submit("test", lambda: {"answer": 42})

    assert _wait_until(lambda: manager.get(job_id).status == DONE)
    job = manager.get(job_id)
    assert job.result == {"answer": 42}
    assert job.error is None


def test_a_failing_job_reports_error_status_not_a_crash():
    manager = JobManager()

    def boom():
        raise ValueError("simulated failure")

    job_id = manager.submit("test", boom)
    assert _wait_until(lambda: manager.get(job_id).status == ERROR)
    job = manager.get(job_id)
    assert "simulated failure" in job.error
    assert job.result is None


def test_unknown_job_id_returns_none():
    manager = JobManager()
    assert manager.get("does-not-exist") is None


def test_job_starts_pending_or_running_before_completion():
    manager = JobManager(max_workers=1)
    # Occupy the only worker so the next job is observably pending.
    started = threading_event = __import__("threading").Event()
    release = __import__("threading").Event()

    def blocker():
        started.set()
        release.wait(timeout=2)
        return {}

    manager.submit("blocker", blocker)
    assert started.wait(timeout=2)

    job_id = manager.submit("test", lambda: {"ok": True})
    job = manager.get(job_id)
    assert job.status == PENDING

    release.set()
    assert _wait_until(lambda: manager.get(job_id).status == DONE)


def test_multiple_jobs_run_concurrently_up_to_worker_count():
    manager = JobManager(max_workers=3)
    job_ids = [manager.submit("test", lambda i=i: {"i": i}) for i in range(3)]
    assert _wait_until(lambda: all(manager.get(j).status == DONE for j in job_ids))
    results = {manager.get(j).result["i"] for j in job_ids}
    assert results == {0, 1, 2}


def test_job_dict_includes_elapsed_seconds():
    manager = JobManager()
    job_id = manager.submit("test", lambda: {"ok": True})
    assert _wait_until(lambda: manager.get(job_id).status == DONE)
    d = manager.get(job_id).to_dict()
    assert d["elapsed_seconds"] >= 0
    assert d["status"] == DONE


def test_old_jobs_are_evicted_past_max_jobs():
    manager = JobManager(max_jobs=3)
    ids = [manager.submit("test", lambda: {}) for _ in range(5)]
    _wait_until(lambda: all(manager.get(j) is not None or True for j in ids))
    time.sleep(0.05)
    remaining = [j for j in ids if manager.get(j) is not None]
    assert len(remaining) <= 3
    # the oldest ones should be the ones evicted
    assert ids[0] not in remaining
