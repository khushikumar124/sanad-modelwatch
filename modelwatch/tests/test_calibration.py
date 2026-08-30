import pytest

from modelwatch.core.calibration import calibrate_degraded_after_consecutive, estimate_single_check_fpr


def test_estimate_single_check_fpr_counts_true_fraction():
    assert estimate_single_check_fpr([True, False, False, False]) == 0.25
    assert estimate_single_check_fpr([False, False]) == 0.0
    assert estimate_single_check_fpr([True, True]) == 1.0


def test_estimate_single_check_fpr_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        estimate_single_check_fpr([])


def test_calibration_recommends_higher_threshold_for_noisier_detectors():
    noisy = calibrate_degraded_after_consecutive(single_check_fpr=0.2, target_incident_fpr=0.01)
    quiet = calibrate_degraded_after_consecutive(single_check_fpr=0.02, target_incident_fpr=0.01)
    assert noisy.recommended_consecutive > quiet.recommended_consecutive


def test_calibration_achieves_the_target_when_not_capped():
    result = calibrate_degraded_after_consecutive(single_check_fpr=0.13, target_incident_fpr=0.01)
    assert result.capped is False
    assert result.achieved_incident_fpr <= result.target_incident_fpr


def test_calibration_matches_hand_computed_example():
    # p=0.1: k=1 -> 0.1 (>0.01), k=2 -> 0.01 (<=0.01) -- exactly at target
    result = calibrate_degraded_after_consecutive(single_check_fpr=0.1, target_incident_fpr=0.01)
    assert result.recommended_consecutive == 2
    assert result.achieved_incident_fpr == pytest.approx(0.01)


def test_calibration_recommends_one_when_already_below_target():
    result = calibrate_degraded_after_consecutive(single_check_fpr=0.001, target_incident_fpr=0.01)
    assert result.recommended_consecutive == 1


def test_calibration_zero_observed_fpr_recommends_minimum_without_overclaiming():
    result = calibrate_degraded_after_consecutive(single_check_fpr=0.0, target_incident_fpr=0.01)
    assert result.recommended_consecutive == 1
    assert result.achieved_incident_fpr == 0.0
    assert result.capped is False


def test_calibration_caps_at_max_consecutive_for_very_noisy_detectors():
    # p=0.9: even at k=10, 0.9**10 ≈ 0.35, nowhere near a 0.01 target
    result = calibrate_degraded_after_consecutive(single_check_fpr=0.9, target_incident_fpr=0.01, max_consecutive=10)
    assert result.recommended_consecutive == 10
    assert result.capped is True
    assert result.achieved_incident_fpr > result.target_incident_fpr


def test_calibration_rejects_out_of_range_inputs():
    with pytest.raises(ValueError, match="single_check_fpr"):
        calibrate_degraded_after_consecutive(single_check_fpr=1.5)
    with pytest.raises(ValueError, match="target_incident_fpr"):
        calibrate_degraded_after_consecutive(single_check_fpr=0.1, target_incident_fpr=0.0)
    with pytest.raises(ValueError, match="max_consecutive"):
        calibrate_degraded_after_consecutive(single_check_fpr=0.1, max_consecutive=0)


def test_calibration_result_to_dict_is_json_serializable():
    import json

    result = calibrate_degraded_after_consecutive(single_check_fpr=0.13, target_incident_fpr=0.01)
    json.dumps(result.to_dict())
