"""Contract Overview: structured extraction of the fields a person
actually needs to orient themselves in a contract before reading it
clause by clause -- parties, dates, payment, termination, governing
law, and so on.

Same extraction pattern as summarizer.py/obligations.py (an LLM call
constrained to a JSON schema, no fixed-field regex), but with two
things summarizer.py's flat string lists don't have:

1. A real status per field -- FOUND / NOT_FOUND / UNCLEAR /
   INSUFFICIENT_EVIDENCE -- instead of an empty list standing in for
   both "the document doesn't have this" and "the model didn't find
   it". The model is asked to report its own status; NOT_FOUND is what
   it says when a field is genuinely absent, UNCLEAR when something
   touches the topic but isn't specific enough to state as fact.
2. Grounding: every FOUND field must quote real contract text, checked
   against the actual document (sanad/features/grounding.py, the same
   check obligations.py uses) before being trusted. A field whose quote
   doesn't check out is downgraded to INSUFFICIENT_EVIDENCE rather than
   shown as fact -- the model claiming something is not the same as the
   document actually saying it.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from sanad.features.grounding import find_evidence_chunk, normalize
from sanad.ingestion.chunking import Chunk, chunk_document
from sanad.rag.llm_client import LLMClient

logger = logging.getLogger(__name__)

FOUND = "found"
NOT_FOUND = "not_found"
UNCLEAR = "unclear"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
_STATUSES = (FOUND, NOT_FOUND, UNCLEAR, INSUFFICIENT_EVIDENCE)

#: (field key, human label) -- the fields section 1 of the spec asks
#: for. "parties" and "major_obligations" are naturally multi-valued
#: (a list of strings each with their own quote); every other field is
#: naturally single-valued for a two-party Indian rental/employment/
#: freelance contract, this project's actual document scope.
FIELDS: tuple[tuple[str, str], ...] = (
    ("effective_date", "Effective date"),
    ("expiry_date", "Expiry date"),
    ("duration", "Duration"),
    ("renewal_terms", "Renewal terms"),
    ("payment_terms", "Payment / compensation"),
    ("notice_period", "Notice period"),
    ("termination_conditions", "Termination conditions"),
    ("governing_law", "Governing law"),
    ("jurisdiction", "Jurisdiction"),
    ("confidentiality", "Confidentiality"),
    ("ip_ownership", "IP ownership"),
    ("non_compete", "Non-compete / non-solicitation"),
    ("liability", "Liability"),
    ("indemnification", "Indemnification"),
    ("dispute_resolution", "Dispute resolution"),
)
_FIELD_KEYS = {key for key, _ in FIELDS}

OVERVIEW_SYSTEM_PROMPT = """You are a legal document analyst producing a structured overview of an Indian contract (rental, employment, or freelance/service agreement).

For EACH of the following fields, report:
- status: exactly one of "found", "not_found", "unclear", "insufficient_evidence"
  - "found": the contract states this clearly and specifically.
  - "not_found": the contract does not address this topic at all.
  - "unclear": the contract touches this topic but the wording is ambiguous or contradictory.
  - "insufficient_evidence": you are not confident enough in either direction to call it found or not found.
- value: a short factual statement of what the contract says, or null if status is not "found".
- source_quote: a short (under 25 words) VERBATIM excerpt from the contract supporting "found" or "unclear", or null otherwise. Copy exact wording, do not paraphrase.

Fields to report on: effective_date, expiry_date, duration, renewal_terms, payment_terms, notice_period, termination_conditions, governing_law, jurisdiction, confidentiality, ip_ownership, non_compete, liability, indemnification, dispute_resolution.

Also extract:
- "parties": one entry per party with their role, e.g. "Landlord: ...", "Tenant: ...". Empty list if the document names no parties.
- "major_obligations": the most important 3-6 concrete obligations, each as {"text": "...", "source_quote": "..."}.

Do not invent, assume, or infer facts not explicitly present in the text. When genuinely unsure, use "unclear" or "insufficient_evidence" rather than guessing "found".

Respond with ONLY a single JSON object, no markdown code fences, no commentary, matching exactly this schema:
{
  "fields": {
    "<field_key>": {"status": "found|not_found|unclear|insufficient_evidence", "value": "<string or null>", "source_quote": "<string or null>"}
  },
  "parties": ["<role: name>", ...],
  "major_obligations": [{"text": "...", "source_quote": "..."}]
}"""

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

_FIELD_ENTRY_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": list(_STATUSES)},
        "value": {"type": ["string", "null"]},
        "source_quote": {"type": ["string", "null"]},
    },
    "required": ["status", "value", "source_quote"],
}

OVERVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "fields": {
            "type": "object",
            "properties": {key: _FIELD_ENTRY_SCHEMA for key, _ in FIELDS},
            "required": [key for key, _ in FIELDS],
        },
        "parties": {"type": "array", "items": {"type": "string"}},
        "major_obligations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "source_quote": {"type": "string"}},
                "required": ["text", "source_quote"],
            },
        },
    },
    "required": ["fields", "parties", "major_obligations"],
}


@dataclass
class OverviewField:
    key: str
    label: str
    status: str
    value: str | None
    source_quote: str | None
    evidence_chunk_index: int | None = None

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "status": self.status,
            "value": self.value,
            "source_quote": self.source_quote,
            "evidence_chunk_index": self.evidence_chunk_index,
        }


@dataclass
class OverviewItem:
    """One entry in parties[] or major_obligations[] -- both are lists
    of grounded, quoted statements, unlike the single-value FIELDS."""
    text: str
    grounded: bool
    source_quote: str = ""
    evidence_chunk_index: int | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "grounded": self.grounded,
            "source_quote": self.source_quote,
            "evidence_chunk_index": self.evidence_chunk_index,
        }


@dataclass
class ContractOverview:
    fields: list[OverviewField] = field(default_factory=list)
    parties: list[OverviewItem] = field(default_factory=list)
    major_obligations: list[OverviewItem] = field(default_factory=list)
    parse_error: bool = False

    def to_dict(self) -> dict:
        return {
            "fields": [f.to_dict() for f in self.fields],
            "parties": [p.to_dict() for p in self.parties],
            "major_obligations": [o.to_dict() for o in self.major_obligations],
            "parse_error": self.parse_error,
        }


def _ground_field(
    key: str, label: str, entry: dict, chunks: list[Chunk], normalized_doc: str
) -> OverviewField:
    status = entry.get("status") if entry.get("status") in _STATUSES else INSUFFICIENT_EVIDENCE
    value = entry.get("value") or None
    source_quote = entry.get("source_quote") or None

    if status in (FOUND, UNCLEAR) and source_quote:
        grounded, evidence_chunk_index = find_evidence_chunk(source_quote, chunks, normalized_doc)
        if not grounded:
            # The model claimed a quote that doesn't actually check out
            # against the document -- the claim is not trustworthy as
            # "found"/"unclear" fact, so it's downgraded rather than
            # shown as if the document really said this.
            logger.warning("overview field claimed grounded but quote not found", extra={"field": key})
            return OverviewField(key=key, label=label, status=INSUFFICIENT_EVIDENCE, value=None, source_quote=None)
        return OverviewField(key=key, label=label, status=status, value=value, source_quote=source_quote,
                              evidence_chunk_index=evidence_chunk_index)

    if status in (FOUND, UNCLEAR) and not source_quote:
        # "found"/"unclear" with no quote at all to check -- same
        # reasoning, not trustworthy as stated.
        return OverviewField(key=key, label=label, status=INSUFFICIENT_EVIDENCE, value=None, source_quote=None)

    return OverviewField(key=key, label=label, status=status, value=None, source_quote=None)


def _ground_item(item: dict, chunks: list[Chunk], normalized_doc: str) -> OverviewItem | None:
    text = str(item.get("text") or item) if isinstance(item, dict) else str(item)
    source_quote = str(item.get("source_quote") or "") if isinstance(item, dict) else ""
    if not text.strip():
        return None
    if source_quote:
        grounded, evidence_chunk_index = find_evidence_chunk(source_quote, chunks, normalized_doc)
    else:
        grounded, evidence_chunk_index = False, None
    return OverviewItem(text=text, grounded=grounded, source_quote=source_quote, evidence_chunk_index=evidence_chunk_index)


def _parse_overview(raw: str, chunks: list[Chunk], document_text: str) -> ContractOverview:
    candidate = raw.strip()
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(candidate)
        data = None
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None
        if data is None:
            logger.warning("overview output was not valid JSON", extra={"raw": raw[:200]})
            return ContractOverview(parse_error=True)

    normalized_doc = normalize(document_text)
    raw_fields = data.get("fields") or {}

    fields = [
        _ground_field(key, label, raw_fields.get(key) or {}, chunks, normalized_doc)
        for key, label in FIELDS
    ]

    parties = [p for p in (_ground_item(item, chunks, normalized_doc) for item in data.get("parties") or []) if p]
    major_obligations = [
        o for o in (_ground_item(item, chunks, normalized_doc) for item in data.get("major_obligations") or []) if o
    ]

    return ContractOverview(fields=fields, parties=parties, major_obligations=major_obligations)


#: Same reasoning as obligations.py's EXTRACTION_TIMEOUT_SECONDS:
#: schema-constrained decoding of a large nested object (15 fields plus
#: two arrays) is measurably slower than a flat schema on a small
#: CPU-bound model.
EXTRACTION_TIMEOUT_SECONDS = 420


def build_overview(document_text: str, llm_client: LLMClient) -> ContractOverview:
    chunks = chunk_document(document_text)
    raw = llm_client.generate(
        system_prompt=OVERVIEW_SYSTEM_PROMPT,
        user_prompt=f"Contract text:\n\n{document_text}",
        response_schema=OVERVIEW_SCHEMA,
        timeout=EXTRACTION_TIMEOUT_SECONDS,
    )
    return _parse_overview(raw, chunks, document_text)
