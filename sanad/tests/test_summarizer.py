"""Summarizer tests using a fake LLMClient -- no real Ollama call, so
these run without a local model installed. Covers well-formed JSON,
markdown-fenced JSON, and garbled non-JSON output.
"""
import json

from sanad.features.summarizer import summarize
from sanad.rag.llm_client import LLMClient

WELL_FORMED_RESPONSE = json.dumps(
    {
        "parties": ["Landlord: Mr. Sharma", "Tenant: Ms. Rao"],
        "key_obligations": ["Tenant shall pay rent by the 5th of each month"],
        "important_dates": ["Lease start: 1 April 2024"],
        "notice_period": "30 days",
        "penalty_clauses": ["Late payment attracts 2% monthly interest"],
        "termination_conditions": ["Either party may terminate with 30 days written notice"],
    }
)


class FakeLLMClient(LLMClient):
    def __init__(self, response: str):
        self.response = response
        self.last_system_prompt = None
        self.last_user_prompt = None

    def generate(self, system_prompt: str, user_prompt: str, response_schema: dict | None = None) -> str:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return self.response


def test_summarize_parses_well_formed_json():
    client = FakeLLMClient(WELL_FORMED_RESPONSE)

    result = summarize("some contract text", client)

    assert result.parse_error is False
    assert result.parties == ["Landlord: Mr. Sharma", "Tenant: Ms. Rao"]
    assert result.notice_period == "30 days"
    assert len(result.penalty_clauses) == 1
    assert "some contract text" in client.last_user_prompt


def test_summarize_strips_markdown_code_fences():
    client = FakeLLMClient(f"```json\n{WELL_FORMED_RESPONSE}\n```")

    result = summarize("text", client)

    assert result.parse_error is False
    assert result.notice_period == "30 days"


def test_summarize_handles_garbled_output_without_crashing():
    client = FakeLLMClient("I'm not able to process this request right now, sorry!")

    result = summarize("text", client)

    assert result.parse_error is True
    assert result.parties == []
    assert result.raw_llm_output == "I'm not able to process this request right now, sorry!"


def test_summarize_treats_missing_notice_period_as_none():
    response = json.dumps({"parties": [], "notice_period": None})
    client = FakeLLMClient(response)

    result = summarize("text", client)

    assert result.notice_period is None
    assert result.parse_error is False
