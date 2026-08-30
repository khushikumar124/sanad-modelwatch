# Drift detection

## Statistical detectors (`modelwatch/drift/detectors.py`)

Four reusable, adapter-independent functions, each comparing a baseline
sample against a current sample and returning a `DetectorResult` with
the raw statistic **and** the p-value/effect size it came from -- never
just a boolean:

| detector | question it answers | used for |
|---|---|---|
| `ks_test` | are these two samples from the same distribution? (shape-agnostic) | retrieval-score distribution, generation latency |
| `wasserstein_distance` | how far apart are they, in the signal's own units? | retrieval-score distribution (catches a broad gradual shift KS's max-CDF-gap can under-report) |
| `population_stability_index` | standard credit-risk-style PSI banding | available for any bounded numeric signal |
| `two_proportion_ztest` | has a rate changed, accounting for sample size? | refusal rate, citation-validity ratio |

Every result reports `insufficient_sample` and refuses to call a verdict
below `MIN_SAMPLE_SIZE` (8) observations in either sample -- a
distributional test on a handful of points produces a p-value that looks
precise but isn't. `confidence` is `1 - p_value` for the two
p-value-based tests; PSI and Wasserstein (which have no p-value by
construction) instead use a documented, fixed effect-size banding -- see
each function's docstring for the exact rule. This is the "statistical
vs. practical significance" distinction Phase 7 of the project brief
asked for: a large sample can make a trivial difference statistically
significant, and this module never conflates the two.

## RAGAdapter (`modelwatch/adapters/rag_adapter.py`)

Supersedes `LiveTelemetryAdapter`'s aggregate-rate-only approach (kept
unmodified under `adapter_name="live_telemetry"` for anything already
registered against it). RAGAdapter keeps every raw per-event value from
the telemetry schema ([telemetry.md](telemetry.md)) and runs real tests
against them:

- **retrieval** -- KS + Wasserstein on pooled retrieval-score distributions
- **generation_latency** -- KS on per-request generation latency
- **refusal** -- two-proportion z-test on grounded/refused counts
- **citation_validity** -- two-proportion z-test on valid vs. requested citations

This is the "RAG quality vector" the project brief asked for: four
independent signals, never collapsed into one opaque number.
`drift_score` (fraction of signals drifted) is reported alongside the
full per-signal breakdown, not instead of it.

## Alert hysteresis (`modelwatch/core/health.py`)

A single anomalous batch can trip almost any threshold-based detector.
The health state machine is `HEALTHY -> WARNING -> DEGRADED -> RECOVERING
-> HEALTHY`, and only creates an alert on entering `DEGRADED`, only
resolves one on confirmed return to `HEALTHY` (which always passes
through one `RECOVERING` check first -- a single clean batch isn't
proof the incident is over).

Defaults (`MODELWATCH_DEGRADED_AFTER_CONSECUTIVE=1`, etc.) reproduce the
original "alert on the very first drifted check" behavior exactly, so
existing callers see no behavior change. Raising the threshold via env
var is the recommended production setting once a model has enough
traffic for consecutive batches to mean something -- see
`modelwatch/config.py`.

## Root-cause diagnosis (`modelwatch/diagnosis/engine.py`)

Given a drifted run's four RAGAdapter signals, ranks three subsystem
hypotheses -- `retrieval`, `generation`, `operational` -- using a fixed,
documented rule: retrieval drift plus downstream citation/refusal drift
is attributed to retrieval (worse context -> more refusals/bad
citations), not double-counted as three separate problems; citation or
refusal drift *without* retrieval drift points at generation instead.
`confidence` is the sum of the confidence values the contributing
signals already reported (not invented), clamped to 1.0. The full
ranking (not just the top pick) is always returned, exposed in the
dashboard's "Why this alert?" panel next to the raw statistics behind
it.

## Known limitations

- The diagnosis rules encode one specific causal story (retrieval
  problems cascade into refusal/citation problems). A genuinely novel
  failure mode -- e.g. a prompt injection that changes citation behavior
  *and* somehow also degrades retrieval scores independently -- would be
  misattributed. The ranking exposes the runner-up score specifically so
  a reader isn't stuck trusting the top pick blindly.
- `two_proportion_ztest`'s null hypothesis (independent binomial samples)
  is an approximation for `citations_valid / citations_requested`, since
  citations within one answer aren't independent trials. Treated as
  adequate for alerting, not as a rigorous inferential claim.
