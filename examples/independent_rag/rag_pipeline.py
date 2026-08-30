"""A tiny, real RAG pipeline with nothing to do with Sanad.

Deliberately self-contained: TF-IDF retrieval (scikit-learn, already a
modelwatch dependency -- no torch/sentence-transformers needed) over a
small hardcoded product-FAQ corpus, and a template-based "generator"
(the top-matching FAQ's answer, or a refusal below a similarity floor).
No LLM call, no API key, no network dependency of its own -- this
exists to prove ModelWatch's RAGAdapter works with ANY retrieval+
generation pipeline that reports the right telemetry shape, not
specifically with Sanad's. A real product RAG app would swap this
retriever/generator for its own; everything below the "pipeline" class
(ModelWatch registration and checking, in run_example.py) stays the
same regardless of what's inside it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FAQ = [
    ("What is your return policy?", "Items can be returned within 30 days of delivery for a full refund."),
    ("How long does shipping take?", "Standard shipping takes 3-5 business days within the country."),
    ("Do you ship internationally?", "Yes, international shipping is available to most countries, typically 7-14 business days."),
    ("How do I track my order?", "A tracking link is emailed once your order ships; it updates within 24 hours."),
    ("What payment methods are accepted?", "We accept major credit cards, PayPal, and bank transfer."),
    ("Can I change my order after placing it?", "Orders can be modified within 1 hour of placement by contacting support."),
    ("Is there a warranty on products?", "Most products carry a 1-year manufacturer warranty against defects."),
    ("How do I cancel a subscription?", "Subscriptions can be cancelled anytime from the account settings page."),
]

REFUSAL_ANSWER = "I don't have information about that in the FAQ."
# Any positive overlap counts as a match with stopwords removed (see
# TfidfVectorizer below) -- a genuinely unrelated query shares zero
# non-stopword terms with every FAQ question and scores exactly 0.0.
GROUNDED_SIMILARITY_FLOOR = 0.05


@dataclass
class RAGAnswer:
    answer: str
    grounded: bool
    top_similarity: float
    retrieval_scores: list[float]
    retrieval_latency_ms: float
    generation_latency_ms: float
    citations: int
    citations_requested: int


class TinyRAGPipeline:
    """Fits a TF-IDF vectorizer over the FAQ questions once, then answers
    by cosine similarity. This is intentionally the simplest possible
    real retrieval -- the point is the ModelWatch integration around it,
    not the retrieval quality."""

    def __init__(self, corpus: list[tuple[str, str]] = FAQ):
        self._questions = [q for q, _ in corpus]
        self._answers = [a for _, a in corpus]
        # stop_words="english": without it, shared function words ("what
        # is the") alone give an unrelated question (e.g. "What is the
        # capital of France?") a deceptively high similarity to an FAQ
        # entry -- a real, measured failure mode of naive lexical
        # retrieval, not a hypothetical one (see this example's README).
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(self._questions)

    def ask(self, query: str, top_k: int = 3) -> RAGAnswer:
        retrieval_started = time.perf_counter()
        query_vec = self._vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self._matrix)[0]
        ranked = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
        # Clamp: floating-point rounding can put cosine similarity a hair
        # outside [0, 1] (observed: 1.0000000000000002 on an exact
        # match) -- real, not hypothetical, caught by this example's own
        # test suite.
        retrieval_scores = [max(0.0, min(1.0, float(similarities[i]))) for i in ranked]
        retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000

        generation_started = time.perf_counter()
        top_similarity = retrieval_scores[0] if retrieval_scores else 0.0
        grounded = top_similarity >= GROUNDED_SIMILARITY_FLOOR
        answer = self._answers[ranked[0]] if grounded else REFUSAL_ANSWER
        generation_latency_ms = (time.perf_counter() - generation_started) * 1000

        return RAGAnswer(
            answer=answer,
            grounded=grounded,
            top_similarity=top_similarity,
            retrieval_scores=retrieval_scores,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            citations=1 if grounded else 0,
            citations_requested=1,
        )
