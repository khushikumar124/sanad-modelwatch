# Drift Lab, the experiment registry, and the benchmark

## Drift Lab (`modelwatch/experiments/drift_lab.py`)

Runs controlled interventions on Sanad's **real** pipeline (real
`VectorStore`, real Ollama calls) and reports measured effects -- no
fabricated "expected detection". Two scenarios:

- `retrieval_narrowing` -- shrinks `top_k` so the correct chunk is often
  not retrieved at all, simulating a retrieval regression.
- `chunk_fragmentation` -- re-chunks source documents into much smaller
  pieces, fragmenting clauses across boundaries.

```bash
python -m modelwatch.experiments.run_drift_lab retrieval_narrowing --limit 10
```

### A real measured run

Captured 2026-08-30, `retrieval_narrowing` on 10 real dataset cases
against `phi3:3.8b`, `top_k` reduced from 6 to 1:

```
is_drifted:      True
drift_score:     0.500
  retrieval            value=0.717 <- DRIFTED
  generation_latency   value=0.700 <- DRIFTED
  refusal              value=0.300
  citation_validity    value=0.875
diagnosis:       retrieval (confidence=1.00)
```

RAGAdapter correctly flagged the retrieval-score distribution shift
(confidence 1.00) from a real intervention on real traffic. It also
flagged generation latency -- a genuine, expected side effect: a
narrower `top_k` means a shorter prompt, so generation is measurably
faster, not slower. The diagnosis engine correctly attributed the
incident to `retrieval` rather than treating latency as an independent
problem, since citation/refusal stayed stable. This is one run on one
model, not a claim the effect always looks exactly this way.

## Experiment registry (`modelwatch/core/storage.py`'s `experiments` table)

Free-form `name` / `kind` / `config` / `results` / `status`, same
persistence pattern as baselines. `MonitoringEngine.record_experiment` /
`list_experiments` / `get_experiment`, exposed at `POST/GET /experiments`.
Used by `scripts/run_benchmark.py --register` to leave a record of what
was actually run.

## Benchmark (`modelwatch/experiments/benchmark.py`)

**DEMO / SIMULATED DATA.** Compares three detection methods on synthetic
trials with known injected-drift ground truth (the only way to measure
precision/recall against a *known* label):

- `single_threshold_refusal` -- the original `LiveTelemetryAdapter` rule
- `ks_retrieval_only` -- one real statistical test, one signal (an
  ablation of RAGAdapter down to a single dimension)
- `rag_adapter_full` -- all four RAGAdapter signals

```bash
python scripts/run_benchmark.py --n-trials 300 --seed 0
```

### A real measured run (n=300 trials, seed=0)

```
method                     precision    recall        f1       fpr
------------------------------------------------------------------
single_threshold_refusal        1.00      0.23      0.38      0.00
ks_retrieval_only                0.94      0.26      0.41      0.06
rag_adapter_full                 0.98      1.00      0.99      0.08

recall by injected drift type:
  single_threshold_refusal  citation=0.00, latency=0.00, refusal=1.00, retrieval=0.00
  ks_retrieval_only         citation=0.02, latency=0.03, refusal=0.00, retrieval=1.00
  rag_adapter_full          citation=1.00, latency=1.00, refusal=1.00, retrieval=1.00
```

This is a clean, direct demonstration of **H1** (see
[research.md](research.md)): each single-signal method has ~100% recall
on the one drift type that touches its own signal and near-zero recall
on every other type, dragging its overall recall down to ~0.23-0.26
across a mixed trial set. The multidimensional adapter catches all four
because it's the only method actually looking at all four signals --
this is expected by construction, not a surprising finding, but it is a
real, reproducible number rather than an assumed one.

**What this does NOT show**: anything about detection on live traffic,
detection delay under sequential monitoring, or performance on drift
types not modeled here (e.g. gradual concept drift, adversarial inputs).
See [research.md](research.md)'s limitations section.

## Ablation study

Unlike the benchmark above (independently built single-signal
detectors), the ablation compares RAGAdapter against **itself** with one
signal's verdict excluded from the overall drift call -- isolating each
signal's marginal contribution rather than conflating it with a
difference in detector implementation.

```bash
python scripts/run_benchmark.py --ablation --n-trials 300 --seed 0
```

### A real measured run (n=300 trials, seed=0)

```
method                     precision    recall        f1       fpr
------------------------------------------------------------------
full                            0.98      1.00      0.99      0.08
without_retrieval               0.99      0.76      0.86      0.02
without_generation_latency      0.98      0.76      0.86      0.06
without_refusal                 0.97      0.77      0.86      0.08
without_citation_validity       0.97      0.75      0.85      0.08

recall by injected drift type:
  full                       citation=1.00, latency=1.00, refusal=1.00, retrieval=1.00
  without_retrieval          citation=1.00, latency=1.00, refusal=1.00, retrieval=0.05
  without_generation_latency  citation=1.00, latency=0.05, refusal=1.00, retrieval=1.00
  without_refusal             citation=1.00, latency=1.00, refusal=0.02, retrieval=1.00
  without_citation_validity   citation=0.06, latency=1.00, refusal=1.00, retrieval=1.00
```

Each signal's removal drops recall on *only* the drift type that signal
exists to catch (down to 0.02-0.06 -- consistent with the residual false
positives you'd expect from four independent statistical tests, not a
sign the ablation is broken), while the other three drift types stay at
~1.00. This confirms the four signals are non-redundant: no signal is
silently doing another signal's job. It does not by itself establish
that four is the right number, or that these four are the best possible
four -- only that each of these four currently pulls its own weight.
