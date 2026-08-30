"""Contract Scenario Simulator: "what happens if I resign after two
years?" -- answers a hypothetical scenario against a specific uploaded
contract, tracing which real clause(s) govern it.

This is NOT just chatbot.py's Q&A prompt relabeled. Measured directly
before building this: the same retrieval, the same real document, the
same real model, asked "What happens if I want to leave before the
lease ends?" through chatbot.py's own system prompt -- REFUSED, even
though the document has an on-point notice-period clause. Through the
prompt below (explicitly framing the input as a scenario that may not
share the contract's own vocabulary, and asking the model to map it to
governing clauses first), the same retrieval and the same model
correctly grounded the answer in that clause. The difference is real
and was checked, not assumed.

Same grounding discipline as chatbot.py: a citation is what decides
"grounded", never the model's self-report; refuse rather than invent
when truly nothing in the document is relevant.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from sanad.config import config
from sanad.rag.llm_client import LLMClient
from sanad.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

SCENARIO_PROMPT_VERSION = "v1"

SCENARIO_SYSTEM_PROMPT = """You are a legal assistant analyzing a hypothetical SCENARIO against a specific uploaded contract, using ONLY the numbered excerpts provided by the user.

The user describes a real-world situation, not necessarily using the contract's own wording (e.g. "leave early" may correspond to a clause about "termination" or "notice period" -- map the scenario to the relevant concept before deciding).

Rules:
- First identify which excerpt(s), if any, govern or are relevant to this scenario.
- If relevant excerpts exist, explain what the contract says would apply in this scenario, grounded ONLY in those excerpts. Do not invent consequences the excerpts don't state.
- If you do answer, you MUST cite which excerpt number(s) you used in "cited_excerpts". An answer with no citation cannot be treated as grounded.
- Only refuse (set "grounded" to false) if truly no excerpt addresses anything related to the scenario.

Respond with ONLY a single JSON object, no markdown code fences, no commentary before or after it, matching exactly this schema:
{
  "grounded": true or false,
  "answer": "<your analysis, or a refusal explaining the document does not address this scenario>",
  "cited_excerpts": [<excerpt numbers used, e.g. [1, 3]; empty list if not grounded>]
}"""

NO_CONTEXT_ANSWER = "This document has no indexed content to search, so I can't analyze that scenario."
UNGROUNDED_CITATION_ANSWER = (
    "I couldn't find a specific part of the document that governs this scenario."
)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

SCENARIO_SCHEMA = {
    "type": "object",
    "properties": {
        "grounded": {"type": "boolean"},
        "answer": {"type": "string"},
        "cited_excerpts": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["grounded", "answer", "cited_excerpts"],
}


@dataclass
class ScenarioAnswer:
    answer: str
    grounded: bool
    cited_chunks: list[dict] = field(default_factory=list)
    retrieved_chunks: list[dict] = field(default_factory=list)
    raw_llm_output: str = ""
    parse_error: bool = False
    citations_requested: int = 0

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "grounded": self.grounded,
            "cited_chunks": self.cited_chunks,
            "retrieved_chunks": self.retrieved_chunks,
            "parse_error": self.parse_error,
        }


def _build_user_prompt(hits: list[dict], scenario: str) -> str:
    excerpts = "\n\n".join(f"[{i + 1}] {hit['text']}" for i, hit in enumerate(hits))
    return f"Excerpts from the contract:\n\n{excerpts}\n\nScenario: {scenario}"


def _as_int(value) -> int | None:
    """Citation numbers should already be constrained to real integers
    by SCENARIO_SCHEMA, but a backend that can't enforce the schema
    (see rag/llm_client.py's own defensive-parsing precedent) can still
    emit them as numeric strings -- accept "3" the same as 3 rather
    than silently dropping an otherwise-valid citation."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return None


def _parse_answer(raw: str, hits: list[dict]) -> ScenarioAnswer:
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
            logger.warning("scenario simulator output was not valid JSON", extra={"raw": raw[:200]})
            return ScenarioAnswer(
                answer="Sorry, I couldn't produce a valid grounded analysis for this scenario.",
                grounded=False,
                retrieved_chunks=hits,
                raw_llm_output=raw,
                parse_error=True,
            )

    model_says_grounded = bool(data.get("grounded", False))
    answer = str(data.get("answer", ""))
    raw_indices = data.get("cited_excerpts") or []
    citations_requested = len(raw_indices)

    cited_chunks = []
    for raw_i in raw_indices:
        i = _as_int(raw_i)
        if i is not None and 1 <= i <= len(hits):
            cited_chunks.append(hits[i - 1])

    # Same reasoning as chatbot.py: a citation is checkable, the
    # model's own "grounded" claim is not.
    grounded = bool(cited_chunks)

    if model_says_grounded and not cited_chunks:
        logger.warning("model claimed grounded with no valid citation, downgrading to refusal")
        return ScenarioAnswer(
            answer=UNGROUNDED_CITATION_ANSWER,
            grounded=False,
            retrieved_chunks=hits,
            raw_llm_output=raw,
            parse_error=False,
            citations_requested=citations_requested,
        )

    return ScenarioAnswer(
        answer=answer,
        grounded=grounded,
        cited_chunks=cited_chunks,
        retrieved_chunks=hits,
        raw_llm_output=raw,
        parse_error=False,
        citations_requested=citations_requested,
    )


def simulate_scenario(
    doc_id: str,
    scenario: str,
    vector_store: VectorStore,
    llm_client: LLMClient,
    top_k: int | None = None,
) -> ScenarioAnswer:
    hits = vector_store.query(doc_id, scenario, top_k=top_k or config.retrieval_top_k)
    if not hits:
        return ScenarioAnswer(answer=NO_CONTEXT_ANSWER, grounded=False, retrieved_chunks=[])

    user_prompt = _build_user_prompt(hits, scenario)
    raw = llm_client.generate(
        system_prompt=SCENARIO_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_schema=SCENARIO_SCHEMA,
    )
    return _parse_answer(raw, hits)
