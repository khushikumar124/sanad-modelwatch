"""Structured contract summarization.

Deliberately not a "summarize this document" prompt: the model is asked to
extract specific fields (parties, obligations, dates, notice period,
penalties, termination) into JSON, matching what a person actually needs
to understand a rental/employment/freelance agreement quickly.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from sanad.rag.llm_client import LLMClient

logger = logging.getLogger(__name__)

SUMMARIZATION_SYSTEM_PROMPT = """You are a legal document analyst extracting structured information from Indian contracts (rental agreements, employment offer letters, freelance/service agreements).

Read the ENTIRE contract text provided by the user and extract ONLY information explicitly stated in it. Do not invent, assume, or infer facts that are not present in the text -- if a field is not mentioned in the document, use an empty list (or null for notice_period).

Respond with ONLY a single JSON object, no markdown code fences, no commentary before or after it, matching exactly this schema:
{
  "parties": ["<name/role of each party, e.g. 'Landlord: ...', 'Tenant: ...'>"],
  "key_obligations": ["<one obligation per entry>"],
  "important_dates": ["<one date/deadline per entry, with what it's for>"],
  "notice_period": "<the notice period as stated, or null if not mentioned>",
  "penalty_clauses": ["<one penalty/liquidated-damages clause per entry>"],
  "termination_conditions": ["<one termination condition per entry>"]
}"""

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class ContractSummary:
    parties: list[str] = field(default_factory=list)
    key_obligations: list[str] = field(default_factory=list)
    important_dates: list[str] = field(default_factory=list)
    notice_period: str | None = None
    penalty_clauses: list[str] = field(default_factory=list)
    termination_conditions: list[str] = field(default_factory=list)
    raw_llm_output: str = ""
    parse_error: bool = False

    def to_dict(self) -> dict:
        return {
            "parties": self.parties,
            "key_obligations": self.key_obligations,
            "important_dates": self.important_dates,
            "notice_period": self.notice_period,
            "penalty_clauses": self.penalty_clauses,
            "termination_conditions": self.termination_conditions,
            "parse_error": self.parse_error,
        }


def _parse_summary(raw: str) -> ContractSummary:
    candidate = raw.strip()
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(candidate)
        if not match:
            logger.warning("summarizer output was not JSON, returning raw text", extra={"raw": raw[:200]})
            return ContractSummary(raw_llm_output=raw, parse_error=True)
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("summarizer output had unparseable JSON-like block", extra={"raw": raw[:200]})
            return ContractSummary(raw_llm_output=raw, parse_error=True)

    def as_list(value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value]
        return [str(value)]

    return ContractSummary(
        parties=as_list(data.get("parties")),
        key_obligations=as_list(data.get("key_obligations")),
        important_dates=as_list(data.get("important_dates")),
        notice_period=data.get("notice_period") or None,
        penalty_clauses=as_list(data.get("penalty_clauses")),
        termination_conditions=as_list(data.get("termination_conditions")),
        raw_llm_output=raw,
        parse_error=False,
    )


def summarize(document_text: str, llm_client: LLMClient) -> ContractSummary:
    user_prompt = f"Contract text:\n\n{document_text}"
    raw = llm_client.generate(system_prompt=SUMMARIZATION_SYSTEM_PROMPT, user_prompt=user_prompt)
    return _parse_summary(raw)
