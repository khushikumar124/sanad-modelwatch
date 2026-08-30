# Research positioning

## The question

> Can multidimensional monitoring -- combining statistical, retrieval,
> generation, and operational signals -- detect silent degradation in
> RAG systems earlier and more reliably than conventional single-metric
> threshold monitoring?

This is a real, open question this codebase is built to let someone
*investigate*, not a claim that has been proven. No wording in this repo
should be read as claiming "first", "novel", "state-of-the-art", or
"publication-ready" -- none of that is asserted here, and none of it
should be inferred from the presence of a research-shaped structure.

## Hypotheses and what's actually been measured

**H1: multidimensional monitoring detects more injected degradation
types than single-metric monitoring.**
Supported by one benchmark run (`docs/experiments.md`, n=300 synthetic
trials): each single-signal baseline is blind (near-zero recall) to
drift types outside its own signal, while the four-signal adapter
catches all four modeled types. This is a real, reproducible result on
*synthetic* trials with known ground truth, not on live traffic --
synthetic because measuring recall against a known label requires
knowing the true label, which live traffic doesn't come with.

**H2: multidimensional monitoring reduces false positives.**
Not directly tested. The one benchmark run shows `rag_adapter_full`'s
false-positive rate (0.08) is *higher* than the single-metric baselines'
(0.00, 0.06) at matched trial composition -- running four independent
tests instead of one increases the chance that at least one fires by
chance on a clean batch. Alert hysteresis (`docs/drift_detection.md`)
exists partly to counteract this in production (a single anomalous
signal on one batch doesn't create an alert), but that mitigation itself
hasn't been benchmarked. **H2 as stated is not supported by what's been
measured; if anything the naive comparison points the other way**, and
resolving that tension (hysteresis-adjusted FPR vs. single-shot FPR)
would be a real next step, not something to paper over.

**H3: retrieval-level signals enable earlier root-cause identification.**
The diagnosis engine (`docs/drift_detection.md`) demonstrably attributes
a retrieval-caused incident to the retrieval subsystem rather than
treating downstream refusal/citation drift as independent problems (see
the Drift Lab result in `docs/experiments.md`). "Earlier" specifically
has not been measured -- there is no detection-delay experiment in this
codebase yet (see Limitations).

**H4: persistent sequential detection (hysteresis) improves reliability
over single-window thresholds.**
The state machine is implemented and unit/integration tested
(`modelwatch/tests/test_health.py`, `test_engine_integration.py`), and
its logic is sound by construction (a relapse after N consecutive
drifted batches is harder to trigger by noise than one bad batch). It
has not been benchmarked against a non-hysteresis baseline on real or
synthetic sequential traffic.

## What this codebase does NOT claim

- That any number in `docs/evaluation.md` or `docs/experiments.md`
  generalizes beyond the one run it was measured on. LLM sampling
  variance alone means a repeat run will differ.
- That the risk-flagging rules (`sanad/features/risk_flagger.py`) or
  anything else in this repo constitutes legal advice.
- That the retrieval ground truth in `datasets/sanad_eval/` is
  human-verified -- it's a documented lexical-overlap heuristic (see
  `docs/evaluation.md`).
- That the diagnosis engine's subsystem attribution generalizes to
  failure modes outside the specific causal pattern it was built to
  recognize (see `docs/drift_detection.md`'s limitations).

## Concrete next steps for someone continuing this

1. **Detection-delay experiment**: simulate a sequential stream of
   batches with drift introduced at a known point, measure how many
   batches (or how much wall-clock time under realistic traffic) each
   method/threshold setting takes to raise an alert.
2. **Hysteresis-adjusted FPR**: rerun the benchmark's false-positive
   comparison with `MODELWATCH_DEGRADED_AFTER_CONSECUTIVE=2` or higher,
   to see whether hysteresis actually closes the FPR gap noted under H2.
3. **Human-annotated retrieval ground truth**: replace or supplement
   `datasets/sanad_eval/`'s auto-derived `relevant_chunks` with a
   human-reviewed pass, and re-measure `retrieval_hit_rate` /
   `citation_correctness` against it.
4. ~~**Ablation study**~~ -- done, see [experiments.md](experiments.md#ablation-study):
   removing any one of RAGAdapter's four signals drops recall to
   0.02-0.06 on exactly the drift type that signal exists to catch,
   while leaving the other three untouched, confirming the four signals
   are non-redundant. This doesn't establish four is the *right* number
   of signals -- only that each of the current four is pulling its own
   weight.
