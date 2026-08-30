"""Unit tests for the alert-hysteresis state machine in
modelwatch/core/health.py. Pure function, no storage/engine involved."""
from modelwatch.core.health import DEGRADED, HEALTHY, RECOVERING, WARNING, next_health_state


def test_default_thresholds_alert_on_first_drifted_run():
    """degraded_after_consecutive=1 by default -- reproduces the original
    'alert on the very first drifted check' behavior."""
    t = next_health_state(HEALTHY, is_drifted=True, consecutive_drifted=0, consecutive_clean=0)
    assert t.state == DEGRADED
    assert t.should_create_alert is True


def test_clean_run_from_healthy_stays_healthy_no_alert():
    t = next_health_state(HEALTHY, is_drifted=False, consecutive_drifted=0, consecutive_clean=0)
    assert t.state == HEALTHY
    assert t.should_create_alert is False
    assert t.should_resolve_alerts is False


def test_hysteresis_requires_consecutive_drifted_runs_before_degraded():
    # degraded_after_consecutive=2: one drifted run alone is only a warning
    t1 = next_health_state(
        HEALTHY, is_drifted=True, consecutive_drifted=0, consecutive_clean=0,
        warning_after_consecutive=1, degraded_after_consecutive=2,
    )
    assert t1.state == WARNING
    assert t1.should_create_alert is False

    t2 = next_health_state(
        WARNING, is_drifted=True, consecutive_drifted=t1.consecutive_drifted, consecutive_clean=0,
        warning_after_consecutive=1, degraded_after_consecutive=2,
    )
    assert t2.state == DEGRADED
    assert t2.should_create_alert is True


def test_a_single_clean_run_clears_a_warning_without_reaching_degraded():
    t1 = next_health_state(
        HEALTHY, is_drifted=True, consecutive_drifted=0, consecutive_clean=0,
        degraded_after_consecutive=3,
    )
    assert t1.state == WARNING

    t2 = next_health_state(
        WARNING, is_drifted=False, consecutive_drifted=t1.consecutive_drifted, consecutive_clean=0,
        degraded_after_consecutive=3,
    )
    assert t2.state == HEALTHY
    assert t2.should_resolve_alerts is False  # never alerted, nothing to resolve


def test_recovery_requires_consecutive_clean_runs_then_one_more_to_reach_healthy():
    # degraded, recovery_after_consecutive=2
    t1 = next_health_state(DEGRADED, is_drifted=False, consecutive_drifted=0, consecutive_clean=0, recovery_after_consecutive=2)
    assert t1.state == DEGRADED  # only 1 clean run so far, needs 2

    t2 = next_health_state(DEGRADED, is_drifted=False, consecutive_drifted=0, consecutive_clean=t1.consecutive_clean, recovery_after_consecutive=2)
    assert t2.state == RECOVERING
    assert t2.should_resolve_alerts is False  # not fully healthy yet

    t3 = next_health_state(RECOVERING, is_drifted=False, consecutive_drifted=0, consecutive_clean=t2.consecutive_clean, recovery_after_consecutive=2)
    assert t3.state == HEALTHY
    assert t3.should_resolve_alerts is True


def test_relapse_during_recovery_returns_to_degraded_without_a_new_alert():
    t = next_health_state(RECOVERING, is_drifted=True, consecutive_drifted=0, consecutive_clean=3, degraded_after_consecutive=1)
    assert t.state == DEGRADED
    assert t.should_create_alert is False  # incident already open, reuse it


def test_staying_degraded_across_multiple_drifted_runs_only_alerts_once():
    t1 = next_health_state(HEALTHY, is_drifted=True, consecutive_drifted=0, consecutive_clean=0, degraded_after_consecutive=1)
    assert t1.should_create_alert is True

    t2 = next_health_state(DEGRADED, is_drifted=True, consecutive_drifted=t1.consecutive_drifted, consecutive_clean=0, degraded_after_consecutive=1)
    assert t2.state == DEGRADED
    assert t2.should_create_alert is False
