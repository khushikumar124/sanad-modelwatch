"""Grounded Q&A across more than one uploaded document.

Same grounding discipline as features/chatbot.py -- answer using ONLY
the retrieved excerpts, cite which one(s) were used, refuse rather than
guess -- extended to a merged, cross-document excerpt pool so a question
like "does the notice period in the rental agreement match the one in
the employment contract?" can be answered by comparing real text from
both, not by asking the model to remember two separate single-document
answers itself.

Retrieval stays per-document (VectorStore.query is doc_id-scoped, and
mixing documents into one ANN search would let one long document's
chunks drown out a short one's), then the results are merged into one
numbered excerpt list, each one tagged with which document it came
from, before a single LLM call reasons over all of them together.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from sanad.rag.llm_client import LLMClient
from sanad.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

CROSS_DOCUMENT_PROMPT_VERSION = "v1"

CROSS_DOCUMENT_SYSTEM_PROMPT = """You are a legal assistant answering a question that may involve comparing or relating MULTIPLE uploaded contracts, using ONLY the numbered excerpts provided by the user. Each excerpt is labeled with which document it came from.

Rules:
- Answer using ONLY the information in the provided excerpts. Do not use outside knowledge of Indian law or typical contract terms to fill gaps.
- When comparing documents, be explicit about which document each part of your answer refers to.
- If the excerpts do not contain enough information to answer the question, you MUST refuse: set "grounded" to false and write an "answer" explaining that the documents do not address this question.
- If you do answer, you MUST cite which excerpt number(s) you used in "cited_excerpts". An answer with no citation cannot be treated as grounded.

Respond with ONLY a single JSON object, no markdown code fences, no commentary before or after it, matching exactly this schema:
{
  "grounded": true or false,
  "answer": "<your answer, or a refusal explaining the documents do not address this question>",
  "cited_excerpts": [<excerpt numbers used, e.g. [1, 3]; empty list if not grounded>]
}"""

NO_CONTEXT_ANSWER = "None of the selected documents have indexed content to search, so I can't answer that."
UNGROUNDED_CITATION_ANSWER = (
    "I couldn't find a specific part of the selected documents that supports a citeable answer to this question."
)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "grounded": {"type": "boolean"},
        "answer": {"type": "string"},
        "cited_excerpts": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["grounded", "answer", "cited_excerpts"],
}


@dataclass
class CrossDocumentAnswer:
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


def _build_user_prompt(hits: list[dict], question: str) -> str:
    excerpts = "\n\n".join(
        f"[{i + 1}] (from \"{hit['doc_label']}\") {hit['text']}" for i, hit in enumerate(hits)
    )
    return f"Excerpts from multiple documents:\n\n{excerpts}\n\nQuestion: {question}"


def _parse_answer(raw: str, hits: list[dict]) -> CrossDocumentAnswer:
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
            logger.warning("cross-document chatbot output was not valid JSON", extra={"raw": raw[:200]})
            return CrossDocumentAnswer(
                answer="Sorry, I couldn't produce a valid grounded answer for this question.",
                grounded=False,
                retrieved_chunks=hits,
                raw_llm_output=raw,
                parse_error=True,
            )

    model_says_grounded = bool(data.get("grounded", False))
    answer = str(data.get("answer", ""))
    cited_indices = data.get("cited_excerpts") or []
    citations_requested = len(cited_indices)

    cited_chunks = [
        hits[i - 1]
        for i in cited_indices
        if isinstance(i, int) and 1 <= i <= len(hits)
    ]

    # Same reasoning as chatbot.py: a citation is checkable, the model's
    # self-reported "grounded" flag is not.
    grounded = bool(cited_chunks)

    if model_says_grounded and not cited_chunks:
        logger.info("model cited excerpts=false but grounded=true; trusting citations, downgrading to refusal")
        return CrossDocumentAnswer(
            answer=UNGROUNDED_CITATION_ANSWER,
            grounded=False,
            retrieved_chunks=hits,
            raw_llm_output=raw,
            parse_error=False,
            citations_requested=citations_requested,
        )

    return CrossDocumentAnswer(
        answer=answer,
        grounded=grounded,
        cited_chunks=cited_chunks,
        retrieved_chunks=hits,
        raw_llm_output=raw,
        parse_error=False,
        citations_requested=citations_requested,
    )


def ask_across_documents(
    documents: list[tuple[str, str]],
    question: str,
    vector_store: VectorStore,
    llm_client: LLMClient,
    top_k_per_doc: int = 4,
) -> CrossDocumentAnswer:
    """documents: [(doc_id, label), ...] -- label is shown to the model
    and in the response so a citation says which document it came from,
    not just an opaque doc_id. top_k_per_doc is deliberately smaller than
    single-document chat's default top_k=6: with N documents contributing
    top_k_per_doc excerpts each, the combined prompt grows with N, so
    this keeps a 3-document comparison at a similar prompt size to one
    single-document chat request."""
    hits: list[dict] = []
    for doc_id, label in documents:
        doc_hits = vector_store.query(doc_id, question, top_k=top_k_per_doc)
        for hit in doc_hits:
            hits.append({**hit, "doc_id": doc_id, "doc_label": label})

    if not hits:
        return CrossDocumentAnswer(answer=NO_CONTEXT_ANSWER, grounded=False, retrieved_chunks=[])

    user_prompt = _build_user_prompt(hits, question)
    raw = llm_client.generate(
        system_prompt=CROSS_DOCUMENT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_schema=ANSWER_SCHEMA,
    )
    return _parse_answer(raw, hits)
