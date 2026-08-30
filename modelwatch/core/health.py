"""Alert-hysteresis state machine: HEALTHY -> WARNING -> DEGRADED ->
RECOVERING -> HEALTHY.

Pure function, no I/O -- MonitoringEngine reads/writes the persisted
state (modelwatch/core/storage.py's model_health table) around calling
this, so the transition logic itself is trivially unit-testable.

Why a state machine instead of "alert whenever check_drift() says
is_drifted": a single anomalous batch (one unlucky refusal, one slow
request) can trip almost any threshold-based detector. Requiring
`degraded_after_consecutive` consecutive drifted runs before creating an
incident, and `recovery_after_consecutive` consecutive clean runs before
closing it, means a real regression still gets caught (usually within a
couple of poll cycles) while single-batch noise self-corrects on the very
next check without ever paging anyone.

Defaults: degraded_after_consecutive=1 reproduces the older "alert on the
very first drifted check" behavior exactly -- this is deliberate, so
existing callers/tests that rely on that see no behavior change unless
they opt into hysteresis by raising the threshold (see
MODELWATCH_DEGRADED_AFTER_CONSECUTIVE). warning_after_consecutive=1 means
WARNING always shows on the first drifted run regardless -- it never
creates an alert or blocks anything, so there's no compatibility reason
to delay it.
"""
from __future__ import annotations

from dataclasses import dataclass

HEALTHY = "healthy"
WARNING = "warning"
DEGRADED = "degraded"
RECOVERING = "recovering"


@dataclass
class HealthTransition:
    state: str
    consecutive_drifted: int
    consecutive_clean: int
    should_create_alert: bool
    should_resolve_alerts: bool


def next_health_state(
    current_state: str,
    is_drifted: bool,
    consecutive_drifted: int,
    consecutive_clean: int,
    warning_after_consecutive: int = 1,
    degraded_after_consecutive: int = 1,
    recovery_after_consecutive: int = 1,
) -> HealthTransition:
    if is_drifted:
        consecutive_drifted += 1
        consecutive_clean = 0

        was_degraded = current_state in (DEGRADED, RECOVERING)
        if consecutive_drifted >= degraded_after_consecutive:
            new_state = DEGRADED
        elif consecutive_drifted >= warning_after_consecutive:
            new_state = WARNING
        else:
            new_state = current_state if was_degraded else HEALTHY

        # Fire exactly once per incident: only on the transition into
        # DEGRADED from a state that wasn't already DEGRADED. A relapse
        # from RECOVERING back to DEGRADED reuses the still-open alert
        # rather than creating a second one for the same incident.
        should_create_alert = new_state == DEGRADED and not was_degraded
        return HealthTransition(new_state, consecutive_drifted, consecutive_clean, should_create_alert, False)

    # clean run
    consecutive_clean += 1
    consecutive_drifted = 0

    if current_state == DEGRADED:
        new_state = RECOVERING if consecutive_clean >= recovery_after_consecutive else DEGRADED
        return HealthTransition(new_state, consecutive_drifted, consecutive_clean, False, False)

    if current_state == RECOVERING:
        return HealthTransition(HEALTHY, consecutive_drifted, consecutive_clean, False, True)

    # WARNING or HEALTHY: a clean run clears a warning immediately -- it
    # never created an alert, so there's nothing to resolve either.
    return HealthTransition(HEALTHY, consecutive_drifted, consecutive_clean, False, False)
