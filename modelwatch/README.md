# ModelWatch

A small drift/quality monitoring framework with a model-agnostic core
(see the boundary below). The monitoring engine never contains
model-type-specific logic — it only talks to a
`ModelAdapter` interface (`build_baseline`, `check_drift`). New model types
are added by writing a new adapter, not by editing the engine.

Two adapters ship with it:

- **ClassifierAdapter** — tabular ML models. Drift = per-feature
  Kolmogorov-Smirnov two-sample test against a stored baseline. Quality =
  accuracy against labels, when supplied.
- **LLMAdapter** — LLM apps, monitored via a golden question/answer set.
  Drift/quality = TF-IDF cosine similarity between actual and expected
  answers on a new batch.
- **RAGAdapter** — a fourth adapter for RAG apps specifically, running
  real statistical tests (KS, Wasserstein, PSI, two-proportion z-test —
  `drift/detectors.py`) on four independent signals: retrieval-score
  distribution, generation latency, refusal rate, citation validity.
  Never collapsed into one number, and every result carries a p-value/
  effect-size/confidence — "insufficient data" rather than a fabricated
  score when a batch is too small to say anything.

Built on the same signals, an **RAG reliability layer** for debugging,
not just alerting:

- **RAG X-Ray** (`core/storage.py`'s `traces` table, dashboard trace
  browser) — full per-request pipeline traces (retrieved chunks with
  similarity scores, claim-level evidence verification, citation
  checks), so a bad answer is debuggable, not just flagged. Requires the
  monitored app to opt into full-trace telemetry (Sanad:
  `SANAD_TELEMETRY_FULL_TRACE`) — a documented, reversible tradeoff since
  it means ModelWatch sees real question/answer/clause text.
- **Diagnosis engine** (`diagnosis/engine.py` for a drifted batch,
  `diagnosis/trace_diagnosis.py` for one request) — given a drift alert
  or a single bad trace, ranks which subsystem (retrieval, generation,
  operational) is the likely cause, with a documented scoring rule, not
  a black box. Shows up in the dashboard as "Why this alert?".
- **Alert hysteresis** (`core/health.py`) — a
  healthy→warning→degraded→recovering→healthy state machine, so a
  relapse needs several consecutive drifted batches, not one noisy one,
  before it pages anyone.
- **Drift Lab** (`experiments/drift_lab.py`) — controlled failure
  injection against Sanad's real pipeline (retrieval narrowing, chunk
  fragmentation), for testing whether the detectors actually catch a
  known problem instead of only working on paper.
- **Benchmark/ablation framework** (`experiments/benchmark.py`) —
  compares detection methods, and drops each RAGAdapter signal in turn,
  on synthetic trials with known ground truth. Real measured numbers
  (not aspirational ones) are in
  [`docs/research.md`](../docs/research.md) and
  [`docs/experiments.md`](../docs/experiments.md).
- **Alert delivery** (`alerts/notifier.py`) — an alert transition can
  push to a webhook (Slack incoming-webhook JSON, or a generic JSON
  payload), configured via `MODELWATCH_ALERT_WEBHOOK_URL`.

For adopting any of this outside this repo, `modelwatch-client/` is a
standalone, pip-installable client (`pip install -e ./modelwatch-client`)
with a LangChain callback handler
(`modelwatch_client.integrations.langchain.ModelWatchCallbackHandler`)
for recording traces from an existing LangChain RAG pipeline without
hand-instrumenting it — see
[`modelwatch-client/README.md`](../modelwatch-client/README.md).

For a team that already has an MLflow tracking setup, `integrations/
mlflow_export.py` exports each drift-check run (drift/quality scores,
per-signal values, health state) there too, alongside ModelWatch's own
storage and dashboard — not a replacement for either. Opt-in via
`MODELWATCH_MLFLOW_ENABLED=true` (`pip install mlflow` first; it's
deliberately not a hard dependency of this project, since it pulls in a
meaningfully heavier stack than anything else here). Defaults to a
local SQLite-backed tracking store (`MODELWATCH_MLFLOW_TRACKING_URI`,
default `sqlite:///mlflow.db`) — point it at a real MLflow tracking
server URL instead to use one.

## What is and isn't general

"Model-agnostic" is true of the core and not of everything, so it is worth
being precise about where the line falls.

**Fully general — works with any model, and contains no model-type logic:**

| Component | Why it stays general |
|---|---|
| `core/adapter_base.py` | The interface itself: `build_baseline` / `check_drift` |
| `core/engine.py` | Register, check, alert, retrain, version. Imports only `logging`, `typing`, the adapter interface and storage — no scipy, no sklearn, no branching on model type |
| `core/storage.py` | Persists baselines, signals and statistics as opaque JSON; never inspects their shape |
| `api/app.py` | REST layer; routes by `adapter_name` through the registry |

Adding a model type costs **one adapter file and one line** in
`api/adapter_registry.py`. Nothing above changes.

**Adapters are model-type-specific by definition** — that is what they are
for — but they differ in how broadly they apply:

- `ClassifierAdapter` — any tabular model with numeric features. Broadly
  general across that whole class of model.
- `LLMAdapter` — any LLM application that can answer a fixed golden set.
  General across LLM apps, not tied to Sanad.
- `LiveTelemetryAdapter` — **specific to a RAG chatbot.** "Refusal rate"
  and "citation rate" are not general ML concepts; they only mean
  something for an app that can decline to answer and cite sources.
  Despite the generic name, this one does not transfer to an arbitrary
  model. A genuinely general equivalent would watch predictions against
  labels (accuracy, MAE) rather than these signals.

**Not general — a known limitation:**

- `dashboard/index.html` holds an `ADAPTERS` descriptor map keyed by
  adapter name, supplying the detector label, metric wording and
  explainer text. A fourth adapter therefore renders with generic
  fallback text until an entry is added. The engine does not need editing
  to support a new model type; **the dashboard does.** The principled fix
  is a `describe()` on the adapter interface so an adapter supplies its
  own presentation metadata, which would restore the property that the
  UI never needs to know what adapters exist.

Sanad's chatbot ([`../sanad`](../sanad)) is the first real integration:
`examples/sanad_golden_set_runner.py` periodically runs a golden set
through Sanad's live chatbot and reports the results here, so Sanad's own
drift/quality shows up on this dashboard.

## Architecture

```
                    ┌─────────────────────────────┐
                    │   dashboard/index.html      │
                    │   (vanilla JS + Chart.js,    │
                    │    polls every ~8s)          │
                    └──────────────┬───────────────┘
                                   │ mounted at /dashboard
                    ┌──────────────▼───────────────┐
                    │   api/app.py (FastAPI)        │
                    │   register / check / history / │
                    │   alerts / retrain / versions  │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │  core/engine.py                │
                    │  MonitoringEngine               │
                    │  (register_model, run_check,     │
                    │   trigger_retrain — zero model-  │
                    │   type-specific logic)            │
                    └──────┬───────────────────┬───────┘
                           │                    │
              ┌────────────▼─────────┐  ┌───────▼─────────────┐
              │ core/adapter_base.py  │  │ core/storage.py       │
              │ ModelAdapter interface │  │ SQLite: models,        │
              │ (build_baseline,        │  │ baselines, runs,        │
              │  check_drift)            │  │ alerts, versions          │
              └────────────┬─────────┘  └─────────────────────┘
                           │
           ┌───────────────┴────────────────┐
┌──────────▼──────────┐          ┌───────────▼─────────────┐
│ adapters/            │          │ adapters/                  │
│ classifier_adapter.py │          │ llm_adapter.py               │
│ (KS test, accuracy)    │          │ (TF-IDF similarity)           │
└───────────────────────┘          └────────────────────────────┘

examples/sanad_golden_set_runner.py  ──HTTP──▶  Sanad API (../sanad)
examples/simulate_drift_demo.py      ──HTTP──▶  Sanad API + ModelWatch API
```

## Quickstart

```bash
cd modelwatch
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run the test suite
pytest

# start the API + dashboard
uvicorn modelwatch.api.app:app --port 8000
```

> The commands above assume you run them from the repository root (so
> `modelwatch.api.app` resolves as a package); adjust `cd`/`PYTHONPATH` if
> you copy this project out on its own.

Open `http://localhost:8000/dashboard/` — it's empty until a model is
registered. Register one and run a check:

```bash
curl -X POST http://localhost:8000/models -H 'Content-Type: application/json' -d '{
  "model_id": "demo-classifier",
  "name": "Demo Classifier",
  "adapter_name": "classifier",
  "baseline_data": {"features": {"age": [23, 45, 31, 29], "income": [50000, 62000, 48000, 71000]}}
}'

curl -X POST http://localhost:8000/models/demo-classifier/check -H 'Content-Type: application/json' -d '{
  "new_data": {"features": {"age": [24, 44, 30], "income": [51000, 60000, 49000]}}
}'
```

Refresh the dashboard and pick "Demo Classifier" from the model selector.

## API

| Method | Path                          | Purpose                                   |
|--------|-------------------------------|--------------------------------------------|
| POST   | `/models`                     | Register a model + build its baseline      |
| GET    | `/models`                     | List registered models                     |
| GET    | `/models/{id}`                | Get one model                              |
| POST   | `/models/{id}/check`          | Run a drift/quality check                  |
| GET    | `/models/{id}/history`        | Run history                                |
| GET    | `/alerts`                     | List alerts (optional `model_id`, `active_only`) |
| POST   | `/models/{id}/retrain`        | Reset baseline to fresh data, bump version |
| GET    | `/models/{id}/versions`       | Version history                            |

## Configuration

All via environment variables (see `config.py`):

| Variable                                       | Default           | Meaning |
|-------------------------------------------------|--------------------|---------|
| `MODELWATCH_DB_PATH`                             | `modelwatch.db`    | SQLite file path |
| `MODELWATCH_API_HOST` / `MODELWATCH_API_PORT`     | `0.0.0.0` / `8000` | API bind address |
| `MODELWATCH_KS_PVALUE_THRESHOLD`                  | `0.05`             | KS test significance level per feature |
| `MODELWATCH_CLASSIFIER_DRIFT_FEATURE_FRACTION`     | `0.3`              | Fraction of features that must be individually flagged to call the batch drifted |
| `MODELWATCH_CLASSIFIER_BONFERRONI_CORRECTION`      | `true`             | Divide the per-feature threshold by feature count, holding the family-wise false-alarm rate near the threshold |
| `MODELWATCH_LLM_SIMILARITY_THRESHOLD`              | `0.35`             | Minimum average TF-IDF similarity before an LLM batch is flagged as drifted |
| `MODELWATCH_LOG_LEVEL`                             | `INFO`             | Log level for library/service code |

## Testing

```bash
pytest modelwatch/tests/ -v
```

Tests use controlled ground truth throughout: data generated with a known
distribution (so whether drift *should* fire is known ahead of time, not
inferred from the test's own output), a construction with a known exact
accuracy, and a full engine-integration test proving that registering a
model plus running a drifted check produces exactly one alert.

## Known limitations

- **Adapters must be live in-process.** `check_drift`/`trigger_retrain`
  need an actual `ModelAdapter` instance, which can't be generically
  serialized to SQLite — only the baseline data and metadata persist. The
  API re-attaches fresh adapter instances for every stored model on
  startup (safe because the two built-in adapters are stateless), but a
  custom adapter with real internal state would need to handle this itself
  via `attach_adapter()`.
- **The LLM detector can only separate models whose quality gap exceeds
  its own run-to-run noise.** This was measured, and it's the sharpest
  limitation of the LLM path. Golden-set runs against Sanad with the
  *same* model (`phi3:3.8b`) scored 0.539, 0.491 and 0.497 — a spread of
  roughly 0.05. Note where that spread does and does not come from:
  back-to-back runs with no model reload are deterministic (verified —
  two consecutive runs produced character-identical answers and the same
  0.5385 score, because `temperature=0.1` plus schema-constrained
  decoding is effectively greedy). The variance appears across model
  *reloads* and across different subsets of pairs. Swapping the chatbot to `llama3.2:3b` moved
  quality by only ~0.02, i.e. **less than the noise floor**. No threshold
  setting separates those two models honestly: at 0.35 neither run flags,
  and at 0.50 the healthy baseline flags too.

  With a model from a genuinely different class the signal is
  unambiguous. Swapping Sanad's chatbot from `phi3:3.8b` to
  `qwen2.5:0.5b` and back, at the **default** 0.35 threshold:

  | step | quality | is_drifted | alerts |
  |---|---|---|---|
  | baseline (phi3) | 0.558 | no | 0 |
  | swapped to qwen2.5:0.5b | **0.261** | **yes** | 1 |
  | after retrain, back on phi3 | 0.517 | no | 0 |

  That is a 0.30 drop — six times the ~0.05 noise floor — so it clears
  the threshold with room to spare, and `trigger_retrain` bumped the
  model to v2 and resolved the alert. Reproduce with:
  `python -m modelwatch.examples.simulate_drift_demo --drift-model qwen2.5:0.5b --limit 5`

  Every check now reports its own spread alongside the headline score —
  standard deviation, standard error and a 95% interval — so this is
  visible rather than something you have to know about. Measured on
  Sanad with 4 golden pairs: quality 0.496, 95% CI [0.348, 0.644]
  against a 0.35 threshold. The interval spans the threshold, meaning at
  that sample size you cannot conclude the system is healthy *or*
  degraded. The alert rule stays a plain mean-vs-threshold comparison —
  requiring the whole interval to clear the threshold would suppress real
  regressions on small golden sets — but the interval tells the reader
  how much to trust it.

  Practical consequences:
  - Calibrate `MODELWATCH_LLM_SIMILARITY_THRESHOLD` against the *same*
    set of pairs you will check with — a `--limit`ed subset has a
    different baseline than the full golden set.
  - Only trust the LLM detector for regressions substantially larger than
    ~0.05. For the drift demo, use a model in a clearly different class
    (e.g. `qwen2.5:0.5b`), not a same-size sibling.
  - Averaging several runs per check, rather than one, would lower the
    noise floor and is the obvious improvement here.

  A more robust design would also store the baseline's own achieved
  quality at registration and flag *relative* drops, removing the manual
  calibration step; the adapter can't do that today because
  `build_baseline` only receives the golden set, never any model output
  to score against it.

- **LLMAdapter uses TF-IDF, not a neural embedding.** This is a deliberate
  dependency choice — it keeps ModelWatch itself free of torch/
  sentence-transformers regardless of what the monitored application uses
  — but it's lexical, not semantic. A paraphrase with no shared vocabulary
  scores as dissimilar even if it means the same thing, so quality/drift
  scores are a rougher proxy than a neural-embedding approach would give.
- **ClassifierAdapter still false-alarms a few percent of the time.**
  Testing each feature separately is a multiple-comparisons problem, so a
  Bonferroni correction is applied by default
  (`MODELWATCH_CLASSIFIER_BONFERRONI_CORRECTION`). Measured over 500 clean
  batches drawn from the identical baseline distribution on a 2-feature
  model: **13.2% false alarms uncorrected, 6.4% corrected.** That residual
  is inherent to threshold-based testing at α=0.05 — it isn't a bug, but
  on a low-feature model you should expect the occasional spurious alert.
  The aggregate rule (fraction of flagged features) is also still a
  heuristic, not a principled combination of per-feature evidence.
- **No authentication** on the API — fine for local/demo use, not for
  exposing this beyond localhost.
- **SQLite** is a single file with a single connection guarded by one
  lock — adequate for a demo/course project, not built for high-concurrency
  production write load.
- **Alerts only resolve via `trigger_retrain`.** There's no manual
  "acknowledge and dismiss" path for an alert that turns out to be a false
  positive short of resetting the baseline.
