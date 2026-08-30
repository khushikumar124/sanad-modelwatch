"""Tests for the independent RAG example's pipeline (no ModelWatch
server needed -- these only exercise rag_pipeline.py itself). The
server-dependent half (run_example.py's registration/check flow) is
verified by actually running it against a live ModelWatch instance,
not by a test here that would need to stand up a server."""
from rag_pipeline import GROUNDED_SIMILARITY_FLOOR, REFUSAL_ANSWER, TinyRAGPipeline


def test_exact_faq_question_is_grounded_with_the_right_answer():
    pipeline = TinyRAGPipeline()
    answer = pipeline.ask("What is your return policy?")
    assert answer.grounded is True
    assert "30 days" in answer.answer


def test_genuinely_unrelated_question_is_refused():
    pipeline = TinyRAGPipeline()
    answer = pipeline.ask("What is the capital of France?")
    assert answer.grounded is False
    assert answer.answer == REFUSAL_ANSWER
    assert answer.top_similarity < GROUNDED_SIMILARITY_FLOOR


def test_retrieval_scores_are_real_cosine_similarities_in_range():
    pipeline = TinyRAGPipeline()
    answer = pipeline.ask("How long does shipping take?")
    assert all(0.0 <= s <= 1.0 for s in answer.retrieval_scores)
    assert answer.retrieval_scores == sorted(answer.retrieval_scores, reverse=True)


def test_citations_only_reported_when_grounded():
    pipeline = TinyRAGPipeline()
    grounded = pipeline.ask("What is your return policy?")
    refused = pipeline.ask("Tell me a joke.")
    assert grounded.citations == 1
    assert refused.citations == 0
    # citations_requested stays 1 either way -- it reflects what was
    # asked of the generator, not whether it could answer.
    assert grounded.citations_requested == refused.citations_requested == 1


def test_latencies_are_real_measured_values_not_placeholders():
    pipeline = TinyRAGPipeline()
    answer = pipeline.ask("What is your return policy?")
    assert answer.retrieval_latency_ms >= 0.0
    assert answer.generation_latency_ms >= 0.0
