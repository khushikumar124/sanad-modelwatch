"""Tests for modelwatch/drift/detectors.py -- each detector checked
against a case where drift obviously should and should not fire, plus
the insufficient-sample gate."""
import json
import random

from modelwatch.drift.detectors import (
    embedding_drift,
    ks_test,
    population_stability_index,
    two_proportion_ztest,
    wasserstein_distance,
)


def _same_distribution(n=200, seed=1):
    rnd = random.Random(seed)
    baseline = [rnd.gauss(0.5, 0.1) for _ in range(n)]
    current = [rnd.gauss(0.5, 0.1) for _ in range(n)]
    return baseline, current


def _shifted_distribution(n=200, seed=1, shift=0.4):
    rnd = random.Random(seed)
    baseline = [rnd.gauss(0.5, 0.1) for _ in range(n)]
    current = [rnd.gauss(0.5 + shift, 0.1) for _ in range(n)]
    return baseline, current


def test_ks_test_does_not_flag_identical_distributions():
    baseline, current = _same_distribution()
    result = ks_test(baseline, current)
    assert result.drift_detected is False
    assert result.p_value is not None and result.p_value > 0.05
    assert result.insufficient_sample is False


def test_ks_test_flags_a_clear_shift():
    baseline, current = _shifted_distribution()
    result = ks_test(baseline, current)
    assert result.drift_detected is True
    assert result.p_value < 0.05
    assert result.confidence > 0.9


def test_ks_test_reports_insufficient_sample_below_minimum():
    result = ks_test([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
    assert result.insufficient_sample is True
    assert result.drift_detected is False
    assert result.p_value is None


def test_wasserstein_distance_flags_a_clear_shift_but_not_identical():
    baseline, current = _same_distribution()
    calm = wasserstein_distance(baseline, current)
    assert calm.drift_detected is False

    baseline, current = _shifted_distribution()
    shifted = wasserstein_distance(baseline, current)
    assert shifted.drift_detected is True
    assert shifted.effect_size > calm.effect_size


def test_psi_bands_match_documented_thresholds():
    baseline, current = _same_distribution()
    calm = population_stability_index(baseline, current)
    assert calm.statistic < 0.1
    assert calm.drift_detected is False

    baseline, current = _shifted_distribution()
    shifted = population_stability_index(baseline, current)
    assert shifted.statistic >= 0.1
    assert shifted.drift_detected is True


def test_two_proportion_ztest_flags_a_real_rate_change():
    # 5% baseline refusal rate vs 40% current, both on decent sample sizes
    result = two_proportion_ztest(baseline_successes=5, baseline_n=100, current_successes=40, current_n=100)
    assert result.drift_detected is True
    assert result.effect_size > 0.3


def test_two_proportion_ztest_does_not_flag_noise_on_small_samples():
    # same true rate (10%), small samples -- a couple of extra events by
    # chance should not read as significant
    result = two_proportion_ztest(baseline_successes=1, baseline_n=10, current_successes=2, current_n=10)
    assert result.drift_detected is False


def test_two_proportion_ztest_insufficient_sample_below_minimum():
    result = two_proportion_ztest(baseline_successes=1, baseline_n=3, current_successes=1, current_n=3)
    assert result.insufficient_sample is True


def test_all_detector_results_are_json_serializable():
    """Storage persists statistics via json.dumps -- a stray numpy scalar
    anywhere in a result's to_dict() would fail silently until a real
    drift check ran against the live database."""
    baseline, current = _shifted_distribution()
    embedding_baseline, embedding_current = _embedding_clusters()
    for result in [
        ks_test(baseline, current),
        wasserstein_distance(baseline, current),
        population_stability_index(baseline, current),
        two_proportion_ztest(10, 100, 40, 100),
        embedding_drift(embedding_baseline, embedding_current, n_permutations=20, random_state=1),
    ]:
        json.dumps(result.to_dict())  # raises TypeError if not serializable


def _embedding_clusters(n=30, dims=8, seed=1, shift=0.0):
    """n vectors per sample, drawn from an isotropic Gaussian cluster in
    `dims` dimensions -- current's cluster is centered `shift` away from
    baseline's along one axis, everything else held equal."""
    rnd = random.Random(seed)

    def _sample(center: list[float]) -> list[list[float]]:
        return [[rnd.gauss(c, 0.2) for c in center] for _ in range(n)]

    baseline_center = [0.0] * dims
    current_center = [shift] + [0.0] * (dims - 1)
    return _sample(baseline_center), _sample(current_center)


def test_embedding_drift_does_not_flag_the_same_cluster():
    baseline, current = _embedding_clusters(shift=0.0, seed=1)
    # a fresh draw from the SAME distribution, not literally identical points
    _, current = _embedding_clusters(shift=0.0, seed=2)
    result = embedding_drift(baseline, current, n_permutations=100, random_state=42)
    assert result.drift_detected is False
    assert result.insufficient_sample is False


def test_embedding_drift_flags_a_clearly_separated_cluster():
    baseline, current = _embedding_clusters(shift=3.0, seed=1)
    result = embedding_drift(baseline, current, n_permutations=100, random_state=42)
    assert result.drift_detected is True
    assert result.p_value is not None and result.p_value < 0.05


def test_embedding_drift_effect_size_grows_with_separation():
    baseline, small_shift = _embedding_clusters(shift=0.5, seed=1)
    _, large_shift = _embedding_clusters(shift=3.0, seed=1)
    small_result = embedding_drift(baseline, small_shift, n_permutations=50, random_state=1)
    large_result = embedding_drift(baseline, large_shift, n_permutations=50, random_state=1)
    assert large_result.effect_size > small_result.effect_size


def test_embedding_drift_reports_insufficient_sample_below_minimum():
    baseline, current = _embedding_clusters(n=3)
    result = embedding_drift(baseline, current)
    assert result.insufficient_sample is True
    assert result.drift_detected is False


def test_embedding_drift_rejects_mismatched_dimensionality():
    baseline, _ = _embedding_clusters(dims=8)
    _, current = _embedding_clusters(dims=4)
    try:
        embedding_drift(baseline, current)
        assert False, "expected a ValueError for mismatched dimensionality"
    except ValueError as e:
        assert "dimensionality" in str(e) or "shapes" in str(e)


def test_embedding_drift_is_deterministic_given_a_random_state():
    baseline, current = _embedding_clusters(shift=1.0, seed=1)
    first = embedding_drift(baseline, current, n_permutations=50, random_state=7)
    second = embedding_drift(baseline, current, n_permutations=50, random_state=7)
    assert first.p_value == second.p_value
    assert first.statistic == second.statistic
