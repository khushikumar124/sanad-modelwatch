# ModelWatch

A small, model-agnostic drift/quality monitoring framework. The monitoring
engine never contains model-type-specific logic — it only talks to a
`ModelAdapter` interface (`build_baseline`, `check_drift`). New model types
are added by writing a new adapter, not by editing the engine.

Two adapters ship with it:

- **ClassifierAdapter** — tabular ML models. Drift = per-feature
  Kolmogorov-Smirnov two-sample test against a stored baseline. Quality =
  accuracy against labels, when supplied.
- **LLMAdapter** — LLM apps, monitored via a golden question/answer set.
  Drift/quality = TF-IDF cosine similarity between actual and expected
  answers on a new batch.

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
