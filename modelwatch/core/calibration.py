"""Auto-calibration for alert-hysteresis thresholds (core/health.py).

`degraded_after_consecutive` -- how many consecutive drifted checks
before an incident actually opens -- has been a hand-set config value
(MODELWATCH_DEGRADED_AFTER_CONSECUTIVE, default 1, see health.py's own
docstring for why). Guessing that number either creates too many false
incidents (set too low) or misses a real regression for too long (set
too high). What it should actually be depends on a measurable quantity:
how often a *single* check falsely reports drift on genuinely clean
traffic. This module turns that measured rate into a threshold
recommendation instead of a guess.

The math: requiring k consecutive drifted checks before alerting turns a
single-check false-positive rate p into an approximate incident-level
false-positive rate of p^k -- IF checks are statistically independent.
That assumption is a real simplification, not a proven guarantee:
consecutive checks over overlapping or adjacent traffic windows can be
correlated (a single slow-burning issue can make several checks in a
row more likely to false-alarm together, not independently), and this
module does not attempt to measure or correct for that. Treat the
recommendation as a reasoned starting point derived from a real number,
not an exact statistical guarantee -- the same honesty this project
applies to every other measured claim (see docs/research.md).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CalibrationResult:
    single_check_fpr: float
    target_incident_fpr: float
    recommended_consecutive: int
    achieved_incident_fpr: float
    #: True if max_consecutive was hit without reaching target_incident_fpr
    #: -- the recommendation is the best available within that cap, not a
    #: guarantee the target was actually met.
    capped: bool

    def to_dict(self) -> dict:
        return {
            "single_check_fpr": self.single_check_fpr,
            "target_incident_fpr": self.target_incident_fpr,
            "recommended_consecutive": self.recommended_consecutive,
            "achieved_incident_fpr": self.achieved_incident_fpr,
            "capped": self.capped,
        }


def estimate_single_check_fpr(check_results: list[bool]) -> float:
    """check_results: one bool per historical check run against traffic
    KNOWN to be clean (no real drift present) -- True means that check
    incorrectly reported drift. Returns the fraction that did.

    This is exactly what modelwatch/experiments/benchmark.py's clean
    trials (drift_injected=False) already measure as a method's
    false_positive_rate -- feed that number in rather than guessing one
    by hand. See scripts/calibrate_hysteresis.py for the intended use.
    """
    if not check_results:
        raise ValueError("need at least one historical check result to estimate a false-positive rate")
    return sum(1 for r in check_results if r) / len(check_results)


def calibrate_degraded_after_consecutive(
    single_check_fpr: float,
    target_incident_fpr: float = 0.01,
    max_consecutive: int = 10,
) -> CalibrationResult:
    """Smallest k (up to max_consecutive) such that single_check_fpr**k
    is at or below target_incident_fpr -- see this module's docstring
    for the independence assumption behind that formula."""
    if not 0.0 <= single_check_fpr <= 1.0:
        raise ValueError(f"single_check_fpr must be in [0, 1], got {single_check_fpr}")
    if not 0.0 < target_incident_fpr < 1.0:
        raise ValueError(f"target_incident_fpr must be in (0, 1), got {target_incident_fpr}")
    if max_consecutive < 1:
        raise ValueError(f"max_consecutive must be at least 1, got {max_consecutive}")

    if single_check_fpr <= 0.0:
        # Never observed a false alarm in the calibration data -- there's
        # no measured rate to extrapolate a smaller one from, so this
        # recommends the smallest threshold rather than implying a
        # zero-risk guarantee that wasn't actually measured.
        return CalibrationResult(single_check_fpr, target_incident_fpr, 1, 0.0, False)

    # A tiny epsilon absorbs float rounding at exact-power boundaries
    # (0.1**2 == 0.010000000000000002 in IEEE 754, not exactly 0.01) so a
    # target chosen to land exactly on p**k doesn't get pushed to k+1 by
    # a rounding artifact rather than a real gap.
    epsilon = 1e-12
    k = 1
    while single_check_fpr**k > target_incident_fpr + epsilon and k < max_consecutive:
        k += 1

    achieved = single_check_fpr**k
    capped = achieved > target_incident_fpr
    return CalibrationResult(single_check_fpr, target_incident_fpr, k, achieved, capped)
