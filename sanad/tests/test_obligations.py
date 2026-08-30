"""Tests for sanad/features/obligations.py using a scripted LLM client,
same pattern as test_summarizer.py / test_chatbot.py."""
import json

from sanad.features.obligations import extract_obligations
from sanad.rag.llm_client import LLMClient

DOCUMENT_TEXT = (
    "1. Rent. The Tenant shall pay to the Owner a monthly rent of Rs. 25,000 on or before "
    "the 7th day of each month.\n\n"
    "2. Notice. This agreement may be terminated by either party serving one month prior "
    "notice in writing."
)


class FakeLLMClient(LLMClient):
    def __init__(self, response: str):
        self.response = response

    def generate(self, system_prompt: str, user_prompt: str, response_schema: dict | None = None, timeout: float = 180) -> str:
        return self.response


def test_grounded_obligation_is_marked_grounded():
    response = json.dumps({
        "obligations": [
            {
                "party": "Tenant",
                "obligation": "Pay monthly rent",
                "deadline": "7th day of each month",
                "category": "payment",
                "source_quote": "pay to the Owner a monthly rent of Rs. 25,000 on or before the 7th day of each month",
            }
        ]
    })
    report = extract_obligations(DOCUMENT_TEXT, FakeLLMClient(response))
    assert len(report.obligations) == 1
    assert report.obligations[0].grounded is True
    assert report.obligations[0].category == "payment"
    assert report.grounded_count == 1


def test_near_verbatim_quote_with_one_word_off_is_still_grounded():
    """Regression test from real data: phi3:3.8b quoted a real clause
    with one word paraphrased ('by local' instead of 'by the local'),
    which failed an exact-substring check even though the obligation is
    clearly real. Content-word overlap should still ground it."""
    document_text = (
        "13. That Owner shall pay for all taxes/cesses levied on the premises by the "
        "local or government authorities in the way of property tax."
    )
    response = json.dumps({
        "obligations": [
            {
                "party": "Owner",
                "obligation": "Pay property taxes on the premises",
                "deadline": None,
                "category": "liability",
                "source_quote": "Owner shall pay for all taxes/cesses levied on the premises by local or government authorities",
            }
        ]
    })
    report = extract_obligations(document_text, FakeLLMClient(response))
    assert report.obligations[0].grounded is True


def test_fabricated_obligation_with_no_matching_quote_is_flagged_ungrounded():
    response = json.dumps({
        "obligations": [
            {
                "party": "Tenant",
                "obligation": "Provide a pet deposit",
                "deadline": None,
                "category": "payment",
                "source_quote": "the Tenant shall pay a pet deposit of Rs. 5,000",
            }
        ]
    })
    report = extract_obligations(DOCUMENT_TEXT, FakeLLMClient(response))
    assert len(report.obligations) == 1
    assert report.obligations[0].grounded is False
    assert report.grounded_count == 0


def test_unknown_category_falls_back_to_other():
    response = json.dumps({
        "obligations": [
            {
                "party": "Tenant", "obligation": "Do something", "deadline": None,
                "category": "not-a-real-category",
                "source_quote": "one month prior notice in writing",
            }
        ]
    })
    report = extract_obligations(DOCUMENT_TEXT, FakeLLMClient(response))
    assert report.obligations[0].category == "other"


def test_exact_duplicate_obligations_are_deduplicated():
    """Regression test from real data: phi3:3.8b repeated the same
    obligation (same party, same wording) several times in one response
    on a real rental contract."""
    response = json.dumps({
        "obligations": [
            {"party": "Tenant", "obligation": "Pay rent", "deadline": None, "category": "payment",
             "source_quote": "monthly rent of Rs. 25,000"},
            {"party": "Tenant", "obligation": "Pay rent", "deadline": None, "category": "payment",
             "source_quote": "monthly rent of Rs. 25,000"},
            {"party": "tenant", "obligation": "PAY RENT", "deadline": None, "category": "payment",
             "source_quote": "monthly rent of Rs. 25,000"},  # same after normalization
        ]
    })
    report = extract_obligations(DOCUMENT_TEXT, FakeLLMClient(response))
    assert len(report.obligations) == 1


def test_same_wording_for_different_parties_is_not_deduplicated():
    response = json.dumps({
        "obligations": [
            {"party": "Tenant", "obligation": "Give notice to terminate", "deadline": None,
             "category": "notice", "source_quote": "one month prior notice in writing"},
            {"party": "Owner", "obligation": "Give notice to terminate", "deadline": None,
             "category": "notice", "source_quote": "one month prior notice in writing"},
        ]
    })
    report = extract_obligations(DOCUMENT_TEXT, FakeLLMClient(response))
    assert len(report.obligations) == 2


def test_multiple_obligations_are_all_extracted():
    response = json.dumps({
        "obligations": [
            {"party": "Tenant", "obligation": "Pay rent", "deadline": "monthly", "category": "payment",
             "source_quote": "monthly rent of Rs. 25,000"},
            {"party": "Either party", "obligation": "Give notice to terminate", "deadline": "one month",
             "category": "notice", "source_quote": "serving one month prior notice in writing"},
        ]
    })
    report = extract_obligations(DOCUMENT_TEXT, FakeLLMClient(response))
    assert len(report.obligations) == 2
    assert report.grounded_count == 2


def test_malformed_json_output_fails_safe():
    report = extract_obligations(DOCUMENT_TEXT, FakeLLMClient("not json at all"))
    assert report.parse_error is True
    assert report.obligations == []


def test_result_is_json_serializable():
    response = json.dumps({"obligations": []})
    report = extract_obligations(DOCUMENT_TEXT, FakeLLMClient(response))
    json.dumps(report.to_dict())
