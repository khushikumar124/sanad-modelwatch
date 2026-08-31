"""Tests for sanad/features/overview.py using a scripted LLM client,
same pattern as test_obligations.py."""
import json

from sanad.features.overview import FOUND, INSUFFICIENT_EVIDENCE, NOT_FOUND, UNCLEAR, build_overview
from sanad.rag.llm_client import LLMClient

DOCUMENT_TEXT = (
    "1. Rent. The Tenant shall pay to the Owner a monthly rent of Rs. 25,000 on or before "
    "the 7th day of each month.\n\n"
    "2. Notice. This agreement may be terminated by either party serving one month prior "
    "notice in writing.\n\n"
    "3. This agreement shall be governed by the laws of India."
)


class FakeLLMClient(LLMClient):
    def __init__(self, response: str):
        self.response = response

    def generate(self, system_prompt: str, user_prompt: str, response_schema: dict | None = None, timeout: float = 180) -> str:
        return self.response


def _base_fields(**overrides) -> dict:
    """All 15 fields default to not_found; overrides supply the ones a
    test cares about, matching the schema's requirement that every
    field key be present."""
    from sanad.features.overview import FIELDS
    fields = {key: {"status": NOT_FOUND, "value": None, "source_quote": None} for key, _ in FIELDS}
    fields.update(overrides)
    return fields


def test_grounded_found_field_is_kept_as_found():
    response = json.dumps({
        "fields": _base_fields(notice_period={
            "status": FOUND, "value": "One month",
            "source_quote": "either party serving one month prior notice in writing",
        }),
        "parties": [], "major_obligations": [],
    })
    overview = build_overview(DOCUMENT_TEXT, FakeLLMClient(response))
    notice = next(f for f in overview.fields if f.key == "notice_period")
    assert notice.status == FOUND
    assert notice.value == "One month"
    assert notice.evidence_chunk_index is not None


def test_found_field_with_unverifiable_quote_is_downgraded_to_insufficient_evidence():
    """The model claiming "found" is not the same as the document
    actually saying it -- a quote that doesn't check out must not be
    shown as fact."""
    response = json.dumps({
        "fields": _base_fields(governing_law={
            "status": FOUND, "value": "Delaware",
            "source_quote": "this contract is governed by the laws of Delaware",  # not in the real document
        }),
        "parties": [], "major_obligations": [],
    })
    overview = build_overview(DOCUMENT_TEXT, FakeLLMClient(response))
    governing_law = next(f for f in overview.fields if f.key == "governing_law")
    assert governing_law.status == INSUFFICIENT_EVIDENCE
    assert governing_law.value is None


def test_found_field_with_no_quote_at_all_is_downgraded():
    response = json.dumps({
        "fields": _base_fields(liability={"status": FOUND, "value": "Capped", "source_quote": None}),
        "parties": [], "major_obligations": [],
    })
    overview = build_overview(DOCUMENT_TEXT, FakeLLMClient(response))
    liability = next(f for f in overview.fields if f.key == "liability")
    assert liability.status == INSUFFICIENT_EVIDENCE


def test_not_found_and_unclear_and_insufficient_evidence_statuses_pass_through():
    response = json.dumps({
        "fields": _base_fields(
            ip_ownership={"status": NOT_FOUND, "value": None, "source_quote": None},
            dispute_resolution={"status": UNCLEAR, "value": None, "source_quote": None},
        ),
        "parties": [], "major_obligations": [],
    })
    overview = build_overview(DOCUMENT_TEXT, FakeLLMClient(response))
    assert next(f for f in overview.fields if f.key == "ip_ownership").status == NOT_FOUND
    # "unclear" with no quote is downgraded to insufficient_evidence (see
    # test_found_field_with_no_quote_at_all_is_downgraded's reasoning --
    # applies to "unclear" too, not just "found").
    assert next(f for f in overview.fields if f.key == "dispute_resolution").status == INSUFFICIENT_EVIDENCE


def test_an_invalid_status_value_falls_back_to_insufficient_evidence():
    response = json.dumps({
        "fields": _base_fields(confidentiality={"status": "definitely_yes", "value": "x", "source_quote": None}),
        "parties": [], "major_obligations": [],
    })
    overview = build_overview(DOCUMENT_TEXT, FakeLLMClient(response))
    assert next(f for f in overview.fields if f.key == "confidentiality").status == INSUFFICIENT_EVIDENCE


def test_all_fifteen_fields_are_always_present_in_the_result():
    from sanad.features.overview import FIELDS
    response = json.dumps({"fields": _base_fields(), "parties": [], "major_obligations": []})
    overview = build_overview(DOCUMENT_TEXT, FakeLLMClient(response))
    assert {f.key for f in overview.fields} == {key for key, _ in FIELDS}


def test_grounded_party_is_marked_grounded_with_evidence():
    response = json.dumps({
        "fields": _base_fields(),
        "parties": [{"text": "Owner", "source_quote": "pay to the Owner a monthly rent"}],
        "major_obligations": [],
    })
    overview = build_overview(DOCUMENT_TEXT, FakeLLMClient(response))
    assert len(overview.parties) == 1
    assert overview.parties[0].grounded is True
    assert overview.parties[0].evidence_chunk_index is not None


def test_ungrounded_obligation_is_kept_but_flagged():
    response = json.dumps({
        "fields": _base_fields(),
        "parties": [],
        "major_obligations": [{"text": "Pay a security deposit", "source_quote": "a completely fabricated quote"}],
    })
    overview = build_overview(DOCUMENT_TEXT, FakeLLMClient(response))
    assert len(overview.major_obligations) == 1
    assert overview.major_obligations[0].grounded is False


def test_malformed_json_output_fails_safe():
    overview = build_overview(DOCUMENT_TEXT, FakeLLMClient("not json at all"))
    assert overview.parse_error is True
    assert overview.fields == []


def test_result_is_json_serializable():
    import json as json_module
    response = json.dumps({
        "fields": _base_fields(notice_period={
            "status": FOUND, "value": "One month",
            "source_quote": "either party serving one month prior notice in writing",
        }),
        "parties": [{"text": "Owner", "source_quote": "pay to the Owner"}],
        "major_obligations": [],
    })
    overview = build_overview(DOCUMENT_TEXT, FakeLLMClient(response))
    json_module.dumps(overview.to_dict())
