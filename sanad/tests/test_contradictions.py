"""Tests for sanad/features/contradictions.py against controlled
obligation lists with known ground truth."""
from sanad.features.contradictions import find_contradictions
from sanad.features.obligations import Obligation


def _ob(category, deadline, grounded=True, obligation=""):
    return Obligation(
        party="Tenant", obligation=obligation or "Do something", deadline=deadline,
        category=category, source_quote="quote", grounded=grounded,
    )


def test_two_different_notice_periods_is_a_contradiction():
    obligations = [
        _ob("notice", "30 days"),
        _ob("notice", "60 days"),
    ]
    report = find_contradictions(obligations)
    assert len(report.contradictions) == 1
    assert report.contradictions[0].category == "notice"
    assert set(report.contradictions[0].values_days) == {30, 60}


def test_same_notice_period_stated_twice_is_not_a_contradiction():
    obligations = [
        _ob("notice", "30 days"),
        _ob("notice", "one month"),  # 30 days under this heuristic's normalization
    ]
    report = find_contradictions(obligations)
    assert report.contradictions == []


def test_word_numbers_are_parsed():
    obligations = [_ob("notice", "one month"), _ob("notice", "two months")]
    report = find_contradictions(obligations)
    assert set(report.contradictions[0].values_days) == {30, 60}


def test_unchecked_category_is_never_flagged_even_with_different_durations():
    """termination is deliberately excluded -- different notice lengths
    for cause vs. no-cause termination are normal, not a contradiction."""
    obligations = [_ob("termination", "10 days"), _ob("termination", "90 days")]
    report = find_contradictions(obligations)
    assert report.contradictions == []


def test_ungrounded_obligations_are_excluded_from_contradiction_checks():
    obligations = [_ob("notice", "30 days"), _ob("notice", "60 days", grounded=False)]
    report = find_contradictions(obligations)
    assert report.contradictions == []


def test_falls_back_to_obligation_text_when_deadline_is_missing():
    obligations = [
        _ob("notice", None, obligation="Give 30 days notice before ending the lease"),
        _ob("notice", None, obligation="Give 90 days notice before ending the lease"),
    ]
    report = find_contradictions(obligations)
    assert len(report.contradictions) == 1


def test_no_contradiction_when_only_one_obligation_in_category():
    obligations = [_ob("notice", "30 days")]
    report = find_contradictions(obligations)
    assert report.contradictions == []


def test_result_is_json_serializable():
    import json
    obligations = [_ob("notice", "30 days"), _ob("notice", "60 days")]
    json.dumps(find_contradictions(obligations).to_dict())
