"""Grounded Q&A over an uploaded contract.

Retrieves the most relevant chunks for the question, then asks the model
to answer using ONLY those excerpts and to explicitly refuse rather than
guess if they don't support an answer. The model is also required to cite
which excerpt(s) it used; an answer claiming to be grounded but citing
nothing is treated as untrustworthy and downgraded to a refusal, since
"grounded" is meaningless without a traceable citation.
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

CHATBOT_SYSTEM_PROMPT = """You are a legal assistant answering questions about a specific uploaded contract, using ONLY the numbered excerpts provided by the user.

Rules:
- Answer using ONLY the information in the provided excerpts. Do not use outside knowledge of Indian law or typical contract terms to fill gaps.
- If the excerpts do not contain enough information to answer the question, you MUST refuse: set "grounded" to false and write an "answer" explaining that the document does not address this question.
- If you do answer, you MUST cite which excerpt number(s) you used in "cited_excerpts". An answer with no citation cannot be treated as grounded.

Respond with ONLY a single JSON object, no markdown code fences, no commentary before or after it, matching exactly this schema:
{
  "grounded": true or false,
  "answer": "<your answer, or a refusal explaining the document does not address this question>",
  "cited_excerpts": [<excerpt numbers used, e.g. [1, 3]; empty list if not grounded>]
}"""

NO_CONTEXT_ANSWER = "This document has no indexed content to search, so I can't answer that."
UNGROUNDED_CITATION_ANSWER = (
    "I couldn't find a specific part of the document that supports a citeable answer to this question."
)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class ChatAnswer:
    answer: str
    grounded: bool
    cited_chunks: list[dict] = field(default_factory=list)
    retrieved_chunks: list[dict] = field(default_factory=list)
    raw_llm_output: str = ""
    parse_error: bool = False

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "grounded": self.grounded,
            "cited_chunks": self.cited_chunks,
            "retrieved_chunks": self.retrieved_chunks,
            "parse_error": self.parse_error,
        }


def _build_user_prompt(hits: list[dict], question: str) -> str:
    excerpts = "\n\n".join(f"[{i + 1}] {hit['text']}" for i, hit in enumerate(hits))
    return f"Excerpts from the contract:\n\n{excerpts}\n\nQuestion: {question}"


def _parse_answer(raw: str, hits: list[dict]) -> ChatAnswer:
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
            logger.warning("chatbot output was not valid JSON", extra={"raw": raw[:200]})
            return ChatAnswer(
                answer="Sorry, I couldn't produce a valid grounded answer for this question.",
                grounded=False,
                retrieved_chunks=hits,
                raw_llm_output=raw,
                parse_error=True,
            )

    grounded = bool(data.get("grounded", False))
    answer = str(data.get("answer", ""))
    cited_indices = data.get("cited_excerpts") or []

    cited_chunks = [
        hits[i - 1]
        for i in cited_indices
        if isinstance(i, int) and 1 <= i <= len(hits)
    ]

    if grounded and not cited_chunks:
        logger.warning("model claimed grounded with no valid citation, downgrading to refusal")
        return ChatAnswer(
            answer=UNGROUNDED_CITATION_ANSWER,
            grounded=False,
            retrieved_chunks=hits,
            raw_llm_output=raw,
            parse_error=False,
        )

    return ChatAnswer(
        answer=answer,
        grounded=grounded,
        cited_chunks=cited_chunks,
        retrieved_chunks=hits,
        raw_llm_output=raw,
        parse_error=False,
    )


def ask(
    doc_id: str,
    question: str,
    vector_store: VectorStore,
    llm_client: LLMClient,
    top_k: int | None = None,
) -> ChatAnswer:
    hits = vector_store.query(doc_id, question, top_k=top_k or config.retrieval_top_k)
    if not hits:
        return ChatAnswer(answer=NO_CONTEXT_ANSWER, grounded=False, retrieved_chunks=[])

    user_prompt = _build_user_prompt(hits, question)
    raw = llm_client.generate(system_prompt=CHATBOT_SYSTEM_PROMPT, user_prompt=user_prompt)
    return _parse_answer(raw, hits)
