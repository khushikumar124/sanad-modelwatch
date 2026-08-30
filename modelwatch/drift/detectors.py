"""Reusable statistical drift detectors, decoupled from any adapter.

Each function compares a baseline sample against a current sample and
returns a DetectorResult with the raw statistic AND the p-value/effect
size it was computed from -- never just a verdict. That's what lets a
reader (or the diagnosis engine in modelwatch/diagnosis/) distinguish:

* statistical significance -- "is this difference likely not noise?"
  (p_value)
* practical / effect-size significance -- "is this difference big enough
  to matter?" (effect_size)

A large sample can make a trivial difference statistically significant;
a small sample can hide a large true difference. `drift_detected` here
always requires the *statistical* test to clear its threshold -- callers
that also want an effect-size floor (e.g. "and Wasserstein distance above
X") should check effect_size themselves, since what counts as "large
enough" is domain-specific and this module can't know it.

Every detector also reports a `confidence` in [0, 1], defined the same
way across detectors so they're comparable: for p-value-based tests,
confidence = 1 - p_value (how unlikely the observed difference is under
the null of "no real change"). PSI has no p-value by construction (it's
a fixed-bin divergence, not a hypothesis test), so its confidence instead
follows the standard PSI banding (see population_stability_index) --
documented there, not invented ad hoc.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from scipy import stats
from scipy.spatial.distance import jensenshannon

#: Below this many observations in either sample, a test is not run --
#: distributional tests on a handful of points produce p-values that look
#: precise but aren't. Sliding-window / hysteresis logic (modelwatch/core)
#: is expected to check `insufficient_sample` before treating a result as
#: real, rather than silently declaring "no drift" for the wrong reason.
MIN_SAMPLE_SIZE = 8


@dataclass
class DetectorResult:
    detector: str
    statistic: float
    p_value: float | None
    effect_size: float
    drift_detected: bool
    confidence: float
    insufficient_sample: bool = False
    n_baseline: int = 0
    n_current: int = 0
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "drift_detected": self.drift_detected,
            "confidence": self.confidence,
            "insufficient_sample": self.insufficient_sample,
            "n_baseline": self.n_baseline,
            "n_current": self.n_current,
            "detail": self.detail,
        }


def _insufficient(detector: str, n_baseline: int, n_current: int) -> DetectorResult:
    return DetectorResult(
        detector=detector,
        statistic=0.0,
        p_value=None,
        effect_size=0.0,
        drift_detected=False,
        confidence=0.0,
        insufficient_sample=True,
        n_baseline=n_baseline,
        n_current=n_current,
        detail={"reason": f"fewer than {MIN_SAMPLE_SIZE} observations in baseline or current sample"},
    )


def ks_test(
    baseline: Sequence[float], current: Sequence[float], alpha: float = 0.05
) -> DetectorResult:
    """Two-sample Kolmogorov-Smirnov test: are baseline and current drawn
    from the same distribution? Distribution-shape-agnostic -- catches
    shifts, spread changes, and multimodal changes a mean/variance
    comparison would miss. effect_size is the KS statistic itself (max
    CDF distance, in [0, 1])."""
    if len(baseline) < MIN_SAMPLE_SIZE or len(current) < MIN_SAMPLE_SIZE:
        return _insufficient("ks", len(baseline), len(current))
    statistic, p_value = stats.ks_2samp(baseline, current)
    return DetectorResult(
        detector="ks",
        statistic=float(statistic),
        p_value=float(p_value),
        effect_size=float(statistic),
        drift_detected=bool(p_value < alpha),
        confidence=max(0.0, min(1.0, 1.0 - p_value)),
        n_baseline=len(baseline),
        n_current=len(current),
        detail={"alpha": alpha},
    )


def wasserstein_distance(
    baseline: Sequence[float], current: Sequence[float], relative_threshold: float = 0.2
) -> DetectorResult:
    """Earth-mover's distance between the two samples, in the original
    units of the signal (e.g. cosine distance, milliseconds). Unlike KS,
    this has no p-value -- it directly measures how far the distributions
    are apart, which KS's max-CDF-gap statistic can under-report for a
    broad, gradual shift.

    `drift_detected` fires when the distance exceeds `relative_threshold`
    times the baseline's own spread (its interquartile range), so the
    same relative_threshold means "the shift is big relative to how noisy
    this signal normally is" whether the signal ranges over 0-1 or over
    hundreds of milliseconds. confidence scales linearly with how far past
    that threshold the distance falls, capped at 1.0 -- there is no p-value
    to report one from, so it says "how far past the line", not
    "how unlikely under a null".
    """
    if len(baseline) < MIN_SAMPLE_SIZE or len(current) < MIN_SAMPLE_SIZE:
        return _insufficient("wasserstein", len(baseline), len(current))
    distance = float(stats.wasserstein_distance(baseline, current))
    q1, q3 = stats.scoreatpercentile(baseline, [25, 75])
    iqr = max(float(q3) - float(q1), 1e-9)  # avoid dividing by zero on a degenerate (constant) baseline
    ratio = distance / iqr
    drift_detected = bool(ratio > relative_threshold)
    confidence = max(0.0, min(1.0, ratio / (relative_threshold * 2)))
    return DetectorResult(
        detector="wasserstein",
        statistic=distance,
        p_value=None,
        effect_size=distance,
        drift_detected=drift_detected,
        confidence=confidence,
        n_baseline=len(baseline),
        n_current=len(current),
        detail={"baseline_iqr": iqr, "ratio_to_iqr": ratio, "relative_threshold": relative_threshold},
    )


def population_stability_index(
    baseline: Sequence[float], current: Sequence[float], bins: int = 10
) -> DetectorResult:
    """PSI over `bins` quantile buckets of the baseline. Standard banding
    (widely used in credit-risk model monitoring, not invented here):
    PSI < 0.1 no significant shift, 0.1-0.25 moderate shift worth a look,
    > 0.25 significant shift. confidence is that banding renormalized to
    [0, 1] (0 at PSI=0, 1 at PSI>=0.25), not a p-value -- PSI is a fixed
    divergence measure, not a hypothesis test.
    """
    if len(baseline) < MIN_SAMPLE_SIZE or len(current) < MIN_SAMPLE_SIZE:
        return _insufficient("psi", len(baseline), len(current))

    edges = sorted(set(stats.scoreatpercentile(baseline, list(_linspace(0, 100, bins + 1)))))
    if len(edges) < 3:
        # baseline has too little spread to form distinct bins (e.g. every
        # value identical) -- PSI is undefined in any meaningful sense.
        return DetectorResult(
            detector="psi",
            statistic=0.0,
            p_value=None,
            effect_size=0.0,
            drift_detected=False,
            confidence=0.0,
            n_baseline=len(baseline),
            n_current=len(current),
            detail={"reason": "baseline has insufficient spread to bin"},
        )

    edges[0] = -float("inf")
    edges[-1] = float("inf")

    def _bucket_fractions(values: Sequence[float]) -> list[float]:
        counts = [0] * (len(edges) - 1)
        for v in values:
            for i in range(len(edges) - 1):
                if edges[i] < v <= edges[i + 1]:
                    counts[i] += 1
                    break
        n = len(values)
        # additive smoothing avoids a log(0) when a bucket is empty in one sample
        return [(c + 0.5) / (n + 0.5 * len(counts)) for c in counts]

    base_fracs = _bucket_fractions(baseline)
    curr_fracs = _bucket_fractions(current)
    psi = sum((c - b) * _safe_log(c / b) for b, c in zip(base_fracs, curr_fracs))

    confidence = max(0.0, min(1.0, psi / 0.25))
    return DetectorResult(
        detector="psi",
        statistic=psi,
        p_value=None,
        effect_size=psi,
        drift_detected=psi >= 0.1,
        confidence=confidence,
        n_baseline=len(baseline),
        n_current=len(current),
        detail={"bins": len(edges) - 1, "band": _psi_band(psi)},
    )


def chi_square_test(
    baseline_counts: dict[str, int], current_counts: dict[str, int], alpha: float = 0.05
) -> DetectorResult:
    """Pearson's chi-square test of independence: has the distribution
    across CATEGORIES (not continuous values -- use ks_test/
    wasserstein_distance/population_stability_index for those) changed
    between baseline and current? Tests baseline and current as two rows
    of a contingency table over the union of categories seen in either.

    effect_size is Cramer's V (statistic normalized by sample size and
    degrees of freedom, in [0, 1]) rather than the raw chi-square
    statistic, which scales with sample size and so isn't comparable
    across differently-sized batches the way the other detectors'
    effect sizes are.
    """
    categories = sorted(set(baseline_counts) | set(current_counts))
    n_baseline = sum(baseline_counts.values())
    n_current = sum(current_counts.values())
    if n_baseline < MIN_SAMPLE_SIZE or n_current < MIN_SAMPLE_SIZE or len(categories) < 2:
        return _insufficient("chi_square", n_baseline, n_current)

    baseline_row = [baseline_counts.get(c, 0) for c in categories]
    current_row = [current_counts.get(c, 0) for c in categories]
    table = [baseline_row, current_row]

    statistic, p_value, dof, _expected = stats.chi2_contingency(table)
    n_total = n_baseline + n_current
    min_dim = min(len(table) - 1, len(categories) - 1)
    cramers_v = float((statistic / (n_total * max(min_dim, 1))) ** 0.5) if min_dim > 0 else 0.0

    return DetectorResult(
        detector="chi_square",
        statistic=float(statistic),
        p_value=float(p_value),
        effect_size=cramers_v,
        drift_detected=bool(p_value < alpha),
        confidence=max(0.0, min(1.0, 1.0 - p_value)),
        n_baseline=n_baseline,
        n_current=n_current,
        detail={"alpha": alpha, "categories": categories, "degrees_of_freedom": int(dof)},
    )


def jensen_shannon_divergence(
    baseline_counts: dict[str, int], current_counts: dict[str, int], threshold: float = 0.1
) -> DetectorResult:
    """Jensen-Shannon divergence between two categorical distributions --
    a symmetric, bounded (0 to ln(2) in nats, or 0 to 1 using scipy's
    base-2 convention) alternative to chi-square. Like PSI, this is a
    direct divergence measure, not a hypothesis test: no p-value, and
    `drift_detected` fires on crossing `threshold` directly rather than
    a significance level. Reaches for this over chi-square when the
    question is "how different are these two distributions" rather than
    "is this difference likely due to chance" -- chi-square's p-value
    conflates sample size with distributional difference; JS divergence
    doesn't scale with sample size at all.
    """
    categories = sorted(set(baseline_counts) | set(current_counts))
    n_baseline = sum(baseline_counts.values())
    n_current = sum(current_counts.values())
    if n_baseline < MIN_SAMPLE_SIZE or n_current < MIN_SAMPLE_SIZE or len(categories) < 2:
        return _insufficient("jensen_shannon", n_baseline, n_current)

    baseline_probs = [baseline_counts.get(c, 0) / n_baseline for c in categories]
    current_probs = [current_counts.get(c, 0) / n_current for c in categories]

    # scipy's jensenshannon returns a *distance* (sqrt of the divergence),
    # base 2 by default -- squaring it back gives the divergence itself,
    # bounded in [0, 1] in that base, which is what's reported here as
    # both statistic and effect_size (no separate scale between them,
    # same as population_stability_index).
    js_distance = float(jensenshannon(baseline_probs, current_probs, base=2))
    js_divergence = js_distance ** 2

    return DetectorResult(
        detector="jensen_shannon",
        statistic=js_divergence,
        p_value=None,
        effect_size=js_divergence,
        drift_detected=js_divergence >= threshold,
        confidence=max(0.0, min(1.0, js_divergence / threshold)) if threshold > 0 else 0.0,
        n_baseline=n_baseline,
        n_current=n_current,
        detail={"threshold": threshold, "categories": categories},
    )


def two_proportion_ztest(
    baseline_successes: int,
    baseline_n: int,
    current_successes: int,
    current_n: int,
    alpha: float = 0.05,
) -> DetectorResult:
    """Two-proportion z-test: has a rate (refusal rate, citation-validity
    rate, ...) changed between baseline and current? This replaces a bare
    "delta > tolerance" check with a test that accounts for sample size --
    the same 10-point rate jump is much less surprising on 10 events than
    on 1000, and a fixed tolerance can't tell the difference.
    """
    if baseline_n < MIN_SAMPLE_SIZE or current_n < MIN_SAMPLE_SIZE:
        return _insufficient("two_proportion_ztest", baseline_n, current_n)

    p_base = baseline_successes / baseline_n
    p_curr = current_successes / current_n
    p_pool = (baseline_successes + current_successes) / (baseline_n + current_n)
    se = (p_pool * (1 - p_pool) * (1 / baseline_n + 1 / current_n)) ** 0.5

    if se == 0:
        # both samples are 100% (or 0%) successes -- no variance to test
        return DetectorResult(
            detector="two_proportion_ztest",
            statistic=0.0,
            p_value=1.0,
            effect_size=abs(p_curr - p_base),
            drift_detected=False,
            confidence=0.0,
            n_baseline=baseline_n,
            n_current=current_n,
            detail={"baseline_rate": p_base, "current_rate": p_curr},
        )

    z = (p_curr - p_base) / se
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
    return DetectorResult(
        detector="two_proportion_ztest",
        statistic=float(z),
        p_value=p_value,
        effect_size=abs(p_curr - p_base),
        drift_detected=bool(p_value < alpha),
        confidence=max(0.0, min(1.0, 1.0 - p_value)),
        n_baseline=baseline_n,
        n_current=current_n,
        detail={"baseline_rate": p_base, "current_rate": p_curr, "alpha": alpha},
    )


def embedding_drift(
    baseline: Sequence[Sequence[float]],
    current: Sequence[Sequence[float]],
    n_permutations: int = 200,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> DetectorResult:
    """Two-sample test for whether a distribution of embedding vectors
    has shifted, using Maximum Mean Discrepancy (MMD^2) with an RBF
    kernel (Gretton et al., 2012) -- the standard way to compare two
    samples of high-dimensional vectors directly, rather than reducing
    each vector to a single scalar (e.g. a similarity score) first and
    running ks_test/wasserstein_distance on that. Collapsing to a scalar
    can hide a real shift: a new topic cluster of questions can leave the
    *mean* similarity to retrieved chunks unchanged while the
    distribution's actual shape moves completely.

    Every other detector in this module has a closed-form p-value; MMD
    doesn't, so one is estimated by permutation: pool both samples,
    reshuffle the baseline/current labels `n_permutations` times, and
    measure how often a random split produces an MMD^2 at least as large
    as the one actually observed. That fraction is the p-value.
    `effect_size` is the MMD^2 statistic itself; `drift_detected` follows
    the permutation p-value against `alpha`, the same convention as
    ks_test and two_proportion_ztest.

    ModelWatch has no embedding model of its own (see this module's and
    the README's notes on staying independent of whatever stack the
    monitored app uses) -- embeddings must be pre-computed by the caller
    and passed in as plain float sequences. Anything that produces a
    fixed-length vector per event works; this function only ever sees
    numbers.
    """
    import numpy as np

    if len(baseline) < MIN_SAMPLE_SIZE or len(current) < MIN_SAMPLE_SIZE:
        return _insufficient("embedding_drift", len(baseline), len(current))

    X = np.asarray(baseline, dtype=float)
    Y = np.asarray(current, dtype=float)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"baseline and current embeddings must be equal-width 2D arrays, got shapes {X.shape} and {Y.shape}"
        )

    rng = np.random.default_rng(random_state)

    def _sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)

    def _mmd2(a: np.ndarray, b: np.ndarray, gamma: float) -> float:
        def _rbf_mean_off_diagonal(sq: np.ndarray) -> float:
            k = np.exp(-gamma * sq)
            n = k.shape[0]
            return (k.sum() - np.trace(k)) / (n * (n - 1))

        kxx = _rbf_mean_off_diagonal(_sq_dists(a, a))
        kyy = _rbf_mean_off_diagonal(_sq_dists(b, b))
        kxy = np.exp(-gamma * _sq_dists(a, b)).mean()
        return float(kxx + kyy - 2 * kxy)

    combined = np.vstack([X, Y])
    pairwise_sq = _sq_dists(combined, combined)
    nonzero = pairwise_sq[pairwise_sq > 0]
    # Median heuristic (Gretton et al.): a data-driven RBF bandwidth
    # instead of a hand-tuned one, so this works across embedding models
    # with very different native scales without per-model configuration.
    median_sq_dist = float(np.median(nonzero)) if nonzero.size else 1.0
    gamma = 1.0 / (2 * median_sq_dist) if median_sq_dist > 0 else 1.0

    observed = _mmd2(X, Y, gamma)

    m = len(X)
    n_total = len(combined)
    exceed_count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n_total)
        if _mmd2(combined[perm[:m]], combined[perm[m:]], gamma) >= observed:
            exceed_count += 1
    # Add-one smoothing: the standard correction for a permutation
    # p-value, since a p-value of exactly 0.0 from a finite number of
    # permutations overstates the evidence -- with n_permutations trials
    # the smallest true p-value this procedure can support is 1/(n+1).
    p_value = (exceed_count + 1) / (n_permutations + 1)

    return DetectorResult(
        detector="embedding_drift",
        statistic=observed,
        p_value=p_value,
        effect_size=observed,
        drift_detected=bool(p_value < alpha),
        confidence=max(0.0, min(1.0, 1.0 - p_value)),
        n_baseline=len(baseline),
        n_current=len(current),
        detail={"alpha": alpha, "n_permutations": n_permutations, "kernel": "rbf", "gamma": gamma},
    )


def _linspace(start: float, stop: float, n: int) -> list[float]:
    if n == 1:
        return [start]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]


def _safe_log(x: float) -> float:
    import math

    return math.log(x) if x > 0 else 0.0


def _psi_band(psi: float) -> str:
    if psi < 0.1:
        return "no significant shift"
    if psi < 0.25:
        return "moderate shift"
    return "significant shift"
