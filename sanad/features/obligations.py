"""Extracts structured obligations and deadlines from a contract: who
must do what, by when, and under which category (payment, notice,
renewal, termination, deliverable, confidentiality, liability, other).

Same extraction pattern as sanad/features/summarizer.py -- an LLM call
constrained to a JSON schema, no fixed-field regex, because "who owes
what to whom by when" genuinely needs language understanding a rule
engine (see risk_flagger.py) can't do.

Unlike summarizer.py's flat string lists, every obligation here carries
a `source_quote` and gets a real, cheap grounding check: the quote must
actually appear (case/whitespace-normalized) in the contract text.
Small local models occasionally invent plausible-sounding but unquoted
obligations, and there is no reason to trust one that can't point at its
own source. Ungrounded obligations are kept (not silently dropped) but
flagged `grounded: false`, so a reader sees what the model claimed
without being asked to trust it uncritically -- the same "citation over
self-report" principle as sanad/features/chatbot.py's groundedness
check.

Grounding uses an exact substring match first, falling back to a
content-word-overlap check (stopwords excluded) at an 0.85 threshold
when that fails -- a real rental contract test surfaced a genuine,
correct extraction whose quote was one word off from the source text
("by local" vs. "by the local"), which an exact-only check would have
wrongly flagged as unverified.

Known limitation, measured, not hidden: on a real rental contract with
phi3:3.8b, a single /obligations call extracted only 1 obligation, when
the document's risk scan (risk_flagger.py) independently found 3
distinct clauses worth flagging. Schema-constrained array decoding on a
small CPU-bound model appears to stop after satisfying the schema
minimally rather than exhaustively covering the document -- this is a
model-capability limit, not a parsing bug (the one obligation it did
return was extracted, quoted, and grounded correctly). A larger model
would likely recover much of this gap, consistent with this project's
other documented small-model limitations (see DEMO.md).
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

CATEGORIES = (
    "payment", "notice", "renewal", "termination",
    "deliverable", "confidentiality", "liability", "other",
)

OBLIGATIONS_SYSTEM_PROMPT = """You are a legal document analyst extracting concrete obligations and deadlines from an Indian contract (rental, employment, or freelance/service agreement).

Read the ENTIRE contract text and extract every concrete obligation: who must do something, what they must do, and any deadline or trigger condition attached to it. Only extract obligations explicitly stated in the text -- do not invent or infer ones that aren't there.

For each obligation, quote a short (under 25 words) VERBATIM excerpt from the contract that supports it in "source_quote" -- copy the exact wording, do not paraphrase it.

Categorize each obligation as exactly one of: payment, notice, renewal, termination, deliverable, confidentiality, liability, other.

Respond with ONLY a single JSON object, no markdown code fences, no commentary, matching exactly this schema:
{
  "obligations": [
    {
      "party": "<who is obligated, e.g. 'Tenant', 'Employee', 'Company'>",
      "obligation": "<what they must do, one sentence>",
      "deadline": "<the deadline/trigger as stated, or null if none>",
      "category": "<one of: payment, notice, renewal, termination, deliverable, confidentiality, liability, other>",
      "source_quote": "<short verbatim quote from the contract>"
    }
  ]
}"""

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

OBLIGATIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "obligations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "party": {"type": "string"},
                    "obligation": {"type": "string"},
                    "deadline": {"type": ["string", "null"]},
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "source_quote": {"type": "string"},
                },
                "required": ["party", "obligation", "category", "source_quote"],
            },
        },
    },
    "required": ["obligations"],
}


#: normalize() and find_evidence_chunk() moved to sanad/features/grounding.py
#: so sanad/features/overview.py can reuse the exact same, already
#: real-data-tuned grounding check instead of a second copy of it.


@dataclass
class Obligation:
    party: str
    obligation: str
    deadline: str | None
    category: str
    source_quote: str
    grounded: bool
    #: which clause (sanad.ingestion.chunking chunk index) supports this
    #: obligation, when localizable -- lets the UI jump straight to it.
    #: None for an ungrounded obligation, or a grounded one whose quote
    #: happened to straddle a chunk boundary (see _find_evidence_chunk).
    evidence_chunk_index: int | None = None

    def to_dict(self) -> dict:
        return {
            "party": self.party,
            "obligation": self.obligation,
            "deadline": self.deadline,
            "category": self.category,
            "source_quote": self.source_quote,
            "grounded": self.grounded,
            "evidence_chunk_index": self.evidence_chunk_index,
        }


@dataclass
class ObligationsReport:
    obligations: list[Obligation] = field(default_factory=list)
    parse_error: bool = False

    @property
    def grounded_count(self) -> int:
        return sum(1 for o in self.obligations if o.grounded)

    def to_dict(self) -> dict:
        return {
            "obligations": [o.to_dict() for o in self.obligations],
            "parse_error": self.parse_error,
            "grounded_count": self.grounded_count,
            "total_count": len(self.obligations),
        }


def _parse_obligations(raw: str, chunks: list[Chunk], document_text: str) -> ObligationsReport:
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
            logger.warning("obligations output was not valid JSON", extra={"raw": raw[:200]})
            return ObligationsReport(parse_error=True)

    normalized_doc = normalize(document_text)
    obligations = []
    # Real-data finding: on a real rental contract, phi3:3.8b repeated
    # several near-identical obligations (the same clause re-described in
    # slightly different words 2-6 times) -- deduplicate on
    # (party, obligation) so the table doesn't repeat the same finding.
    # A stricter/looser dedup key would either miss real near-duplicates
    # or collapse genuinely distinct obligations; this exact-match key is
    # conservative on purpose.
    seen: set[tuple[str, str]] = set()
    for item in data.get("obligations") or []:
        if not isinstance(item, dict):
            continue
        party = str(item.get("party") or "unspecified")
        obligation_text = str(item.get("obligation") or "")
        dedup_key = (normalize(party), normalize(obligation_text))
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        category = item.get("category") if item.get("category") in CATEGORIES else "other"
        source_quote = str(item.get("source_quote") or "")
        grounded, evidence_chunk_index = find_evidence_chunk(source_quote, chunks, normalized_doc)
        obligations.append(
            Obligation(
                party=party,
                obligation=obligation_text,
                deadline=item.get("deadline") or None,
                category=category,
                source_quote=source_quote,
                grounded=grounded,
                evidence_chunk_index=evidence_chunk_index,
            )
        )
    return ObligationsReport(obligations=obligations)


#: Schema-constrained decoding of an array of objects (each with an enum
#: field) is measurably slower than a flat-object schema on a small
#: CPU-bound model -- measured taking longer than the default 180s
#: timeout on a real rental contract with phi3:3.8b. See llm_client.py's
#: generate() docstring.
EXTRACTION_TIMEOUT_SECONDS = 420


def extract_obligations(document_text: str, llm_client: LLMClient) -> ObligationsReport:
    user_prompt = f"Contract text:\n\n{document_text}"
    raw = llm_client.generate(
        system_prompt=OBLIGATIONS_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_schema=OBLIGATIONS_SCHEMA,
        timeout=EXTRACTION_TIMEOUT_SECONDS,
    )
    chunks = chunk_document(document_text)
    return _parse_obligations(raw, chunks, document_text)
