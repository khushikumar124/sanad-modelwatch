"""LLMAdapter tests against controlled ground truth: expected answers are
paired with actual answers that are either near-identical (clean) or
deliberately unrelated (drifted), so whether drift should fire is known
in advance.
"""
from modelwatch.adapters.llm_adapter import LLMAdapter

GOLDEN_SET = [
    {
        "prompt": "What is the notice period for termination?",
        "expected_answer": "The notice period for termination of this agreement is thirty days written notice.",
    },
    {
        "prompt": "Can the landlord keep the security deposit?",
        "expected_answer": "The landlord may deduct from the security deposit only for unpaid rent or damages beyond normal wear and tear.",
    },
    {
        "prompt": "What is the monthly rent amount?",
        "expected_answer": "The monthly rent amount is twenty five thousand rupees payable in advance.",
    },
]


def test_clean_batch_is_not_flagged():
    adapter = LLMAdapter()
    baseline = adapter.build_baseline(GOLDEN_SET)

    clean_batch = [
        {
            "prompt": g["prompt"],
            # near-identical phrasing to the expected answer
            "actual_answer": g["expected_answer"],
        }
        for g in GOLDEN_SET
    ]

    result = adapter.check_drift(baseline, clean_batch)

    assert result.is_drifted is False
    assert result.quality_score > 0.9
    assert all(not s.is_drifted for s in result.signals)


def test_drifted_batch_is_flagged():
    adapter = LLMAdapter()
    baseline = adapter.build_baseline(GOLDEN_SET)

    drifted_batch = [
        {
            "prompt": g["prompt"],
            # deliberately unrelated text sharing no vocabulary with the expected answer
            "actual_answer": "Bananas are a good source of potassium and grow in tropical climates.",
        }
        for g in GOLDEN_SET
    ]

    result = adapter.check_drift(baseline, drifted_batch)

    assert result.is_drifted is True
    assert result.quality_score < 0.35
    assert all(s.is_drifted for s in result.signals)


def test_quality_score_is_average_similarity():
    adapter = LLMAdapter()
    baseline = adapter.build_baseline(GOLDEN_SET)

    mixed_batch = [
        {"prompt": GOLDEN_SET[0]["prompt"], "actual_answer": GOLDEN_SET[0]["expected_answer"]},
        {"prompt": GOLDEN_SET[1]["prompt"], "actual_answer": GOLDEN_SET[1]["expected_answer"]},
        {
            "prompt": GOLDEN_SET[2]["prompt"],
            "actual_answer": "Bananas are a good source of potassium and grow in tropical climates.",
        },
    ]

    result = adapter.check_drift(baseline, mixed_batch)

    similarities = [s.value for s in result.signals]
    assert result.quality_score == sum(similarities) / len(similarities)
    # two near-identical answers and one wholly unrelated one -> mixed, not clean
    assert 0.3 < result.quality_score < 0.9


def test_unmatched_prompts_are_skipped_not_errored():
    adapter = LLMAdapter()
    baseline = adapter.build_baseline(GOLDEN_SET)

    batch = [{"prompt": "This prompt is not in the golden set.", "actual_answer": "irrelevant"}]

    result = adapter.check_drift(baseline, batch)

    assert result.signals == []
    assert result.quality_score is None
    assert result.is_drifted is False
