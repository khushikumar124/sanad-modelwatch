"""Tests for the golden set and the runner's orchestration logic.

Ollama, Sanad, and ModelWatch aren't all running together in this test
environment, so network calls are exercised against a FakeSession rather
than live services -- these tests verify the runner's request-shaping and
control flow, not real end-to-end model behavior (see the module docstring
on sanad_golden_set_runner.py for how to run that live).
"""
from pathlib import Path

from modelwatch.examples.golden_set import GOLDEN_SET
from modelwatch.examples.sanad_golden_set_runner import (
    MODEL_ID,
    collect_actual_answers,
    ensure_model_registered,
    group_by_source_file,
    run_check,
    to_baseline_data,
    upload_documents,
)


class FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"status {self.status_code}")


class FakeSession:
    def __init__(self, get_handler=None, post_handler=None):
        self.get_handler = get_handler or (lambda url, **kw: FakeResponse({}, 404))
        self.post_handler = post_handler or (lambda url, **kw: FakeResponse({}))
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self.get_handler(url, **kwargs)

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self.post_handler(url, **kwargs)


# -- golden set validity ---------------------------------------------------


def test_golden_set_covers_multiple_contract_types():
    types = {p["contract_type"] for p in GOLDEN_SET}
    assert "rental" in types
    assert "freelance" in types


def test_golden_set_size_is_reasonable():
    assert 10 <= len(GOLDEN_SET) <= 25


def test_golden_set_source_files_exist():
    for pair in GOLDEN_SET:
        assert Path(pair["source_file"]).exists(), f"missing file: {pair['source_file']}"


def test_golden_set_expected_answers_have_no_template_blanks():
    blank_markers = ["[.]", "____", "<<", "[___]", "[__]", "(Amount", "(Date"]
    for pair in GOLDEN_SET:
        for marker in blank_markers:
            assert marker not in pair["expected_answer"], (
                f"expected_answer for {pair['prompt']!r} still contains template blank {marker!r}"
            )


# -- pure shaping functions -------------------------------------------------


def test_group_by_source_file():
    grouped = group_by_source_file(GOLDEN_SET)
    assert len(grouped) == len({p["source_file"] for p in GOLDEN_SET})
    for source_file, pairs in grouped.items():
        assert all(p["source_file"] == source_file for p in pairs)


def test_to_baseline_data_shape():
    baseline = to_baseline_data(GOLDEN_SET[:2])
    assert baseline == [
        {"prompt": GOLDEN_SET[0]["prompt"], "expected_answer": GOLDEN_SET[0]["expected_answer"]},
        {"prompt": GOLDEN_SET[1]["prompt"], "expected_answer": GOLDEN_SET[1]["expected_answer"]},
    ]


# -- network-calling functions, against a fake session ----------------------


def test_upload_documents_uploads_each_unique_file_once():
    small_set = GOLDEN_SET[:3]
    upload_count = {"n": 0}

    def post_handler(url, **kwargs):
        assert url.endswith("/api/documents")
        upload_count["n"] += 1
        return FakeResponse({"doc_id": f"doc-{upload_count['n']}"})

    session = FakeSession(post_handler=post_handler)
    doc_ids = upload_documents(session, "http://sanad", small_set)

    unique_files = {p["source_file"] for p in small_set}
    assert set(doc_ids.keys()) == unique_files
    post_calls = [c for c in session.calls if c[0] == "POST"]
    assert len(post_calls) == len(unique_files)


def test_collect_actual_answers_calls_chat_per_pair():
    golden_subset = GOLDEN_SET[:2]
    doc_ids = {p["source_file"]: "doc-1" for p in golden_subset}

    def post_handler(url, **kwargs):
        question = kwargs["json"]["question"]
        return FakeResponse({"answer": f"answer to: {question}", "grounded": True})

    session = FakeSession(post_handler=post_handler)
    results = collect_actual_answers(session, "http://sanad", golden_subset, doc_ids)

    assert len(results) == 2
    assert results[0]["prompt"] == golden_subset[0]["prompt"]
    assert results[0]["actual_answer"] == f"answer to: {golden_subset[0]['prompt']}"


def test_ensure_model_registered_skips_when_already_registered():
    session = FakeSession(get_handler=lambda url, **kw: FakeResponse({"model_id": MODEL_ID}, 200))
    ensure_model_registered(session, "http://modelwatch", GOLDEN_SET)
    assert not any(c[0] == "POST" for c in session.calls)


def test_ensure_model_registered_registers_when_missing():
    session = FakeSession(
        get_handler=lambda url, **kw: FakeResponse({}, 404),
        post_handler=lambda url, **kw: FakeResponse({"model_id": MODEL_ID}, 201),
    )
    ensure_model_registered(session, "http://modelwatch", GOLDEN_SET)
    post_calls = [c for c in session.calls if c[0] == "POST"]
    assert len(post_calls) == 1
    assert post_calls[0][2]["json"]["adapter_name"] == "llm"


def test_run_check_passes_through_result():
    session = FakeSession(post_handler=lambda url, **kw: FakeResponse({"drift_score": 0.1, "is_drifted": False}))
    result = run_check(session, "http://modelwatch", [{"prompt": "p", "actual_answer": "a"}])
    assert result == {"drift_score": 0.1, "is_drifted": False}
