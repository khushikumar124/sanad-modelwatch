"""ModelAdapter for LLM apps, monitored via a golden question/answer set.

Baseline: a fixed set of {prompt, expected_answer} pairs assumed correct.
Drift/quality: on each check, the caller supplies {prompt, actual_answer}
pairs (the live model's outputs for those same prompts) and we score each
against its expected answer with TF-IDF cosine similarity.

TF-IDF instead of a neural embedding model is a deliberate dependency
choice: it keeps ModelWatch itself free of torch/sentence-transformers, so
the monitoring framework stays lightweight and installable independent of
whatever heavy model stack the monitored application (e.g. Sanad) uses. The
tradeoff is TF-IDF is lexical, not semantic -- a paraphrase with no shared
vocabulary scores as dissimilar even if it means the same thing. See the
README's "known limitations" section.
"""
from __future__ import annotations

import logging
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modelwatch.config import config
from modelwatch.core.adapter_base import DriftCheckResult, ModelAdapter, SignalResult

logger = logging.getLogger(__name__)


class LLMAdapter(ModelAdapter):
    adapter_name = "llm"

    def __init__(self, similarity_threshold: float | None = None):
        self.similarity_threshold = similarity_threshold or config.llm_similarity_threshold

    def build_baseline(self, data: list[dict[str, str]]) -> dict[str, Any]:
        """data = [{"prompt": ..., "expected_answer": ...}, ...] (the golden set)."""
        return {"golden_set": [{"prompt": d["prompt"], "expected_answer": d["expected_answer"]} for d in data]}

    def check_drift(self, baseline: dict[str, Any], new_data: list[dict[str, str]]) -> DriftCheckResult:
        """new_data = [{"prompt": ..., "actual_answer": ...}, ...], matched to the
        golden set by exact prompt match."""
        expected_by_prompt = {g["prompt"]: g["expected_answer"] for g in baseline["golden_set"]}

        matched = []
        for item in new_data:
            expected = expected_by_prompt.get(item["prompt"])
            if expected is None:
                logger.warning("prompt not in golden set, skipping", extra={"prompt": item["prompt"]})
                continue
            matched.append((item["prompt"], expected, item["actual_answer"]))

        if not matched:
            return DriftCheckResult(drift_score=0.0, quality_score=None, is_drifted=False, signals=[])

        expected_texts = [m[1] for m in matched]
        actual_texts = [m[2] for m in matched]
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(expected_texts + actual_texts)
        n = len(matched)
        similarities = cosine_similarity(tfidf[:n], tfidf[n:]).diagonal()

        signals: list[SignalResult] = []
        for (prompt, expected, actual), similarity in zip(matched, similarities):
            similarity = float(similarity)
            signals.append(
                SignalResult(
                    name=prompt[:80],
                    value=similarity,
                    is_drifted=similarity < self.similarity_threshold,
                    detail={"expected_answer": expected, "actual_answer": actual},
                )
            )

        avg_similarity = sum(s.value for s in signals) / len(signals)
        drift_score = 1.0 - avg_similarity
        is_drifted = avg_similarity < self.similarity_threshold

        return DriftCheckResult(
            drift_score=drift_score,
            quality_score=avg_similarity,
            is_drifted=is_drifted,
            signals=signals,
        )
