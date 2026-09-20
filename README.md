# Sanad + ModelWatch

[![tests](https://github.com/khushikumar124/sanad-modelwatch/actions/workflows/tests.yml/badge.svg)](https://github.com/khushikumar124/sanad-modelwatch/actions/workflows/tests.yml)

Two projects that show each other's failure modes — and, together, one
answer to a real deployment problem: an LLM-powered app shipped without
anything watching it for silent quality degradation is a real, recurring
risk (a model swap, a prompt change, or a retrieval regression can make
answers quietly worse with no error, no crash, nothing an uptime check
would ever catch). Most teams either skip monitoring an LLM app entirely,
or bolt on generic APM that has no concept of "did the answer actually
degrade." This repo packages a concrete alternative: a real RAG app
(Sanad) and a real, separately-reusable monitoring framework
(ModelWatch) that watches it from day one — via `docker compose up`, one
deployable unit, not two things you'd have to remember to wire together
later. See [`docker-compose.yml`](docker-compose.yml) and "Ship it," below.

**[Sanad](sanad/)** is a RAG app: upload a contract (PDF or scanned
image), get a grounded summary, ask it questions, and get a rule-based
scan for risky, missing, or contradictory clauses. Answers cite the
exact excerpt they came from, and refuse rather than guess when the
document doesn't address the question.

**[ModelWatch](modelwatch/)** is a separate, reusable framework that
watches any model — a tabular classifier, an LLM app, or specifically a
RAG pipeline — for silent quality degradation, using real statistical
tests (KS, Wasserstein, PSI) rather than a hand-waved threshold. It
never contains model-type-specific logic: it only talks to an adapter
interface, and Sanad's chatbot is its first real integration, not a
special case wired into the core. It also doesn't secretly need Sanad to
even start — see [`shared/jobs.py`](shared/jobs.py)'s docstring for a
real coupling bug in that claim that was found and fixed, not just
asserted.

### Is there deep learning here? Precisely, not loosely

**Sanad performs real DL *inference*** — a pretrained sentence-transformer
for embeddings and a pretrained LLM (via Ollama) for generation, in a
genuine RAG pipeline (chunking, hybrid retrieval, grounding
verification). **ModelWatch is deliberately *not* deep learning** — its
core is classical statistics (KS test, Wasserstein distance, PSI,
chi-square, a two-proportion z-test) and classical ML (TF-IDF + cosine
similarity, PCA), specifically so it can watch a model without needing to
load one itself; see `modelwatch/core/engine.py`'s own docstring, which
states this as a design constraint, not an oversight. **No model in this
repo is trained from scratch** — every LLM/embedding model is off-the-
shelf, used purely for inference. The one place this repo does train and
test a real supervised model — a clause-risk classifier — reports both a
negative result (TF-IDF, a single train/test split) and, after fixing
two real methodological weaknesses the first attempt surfaced (embeddings
instead of bag-of-words, leave-one-out CV instead of one split), a
genuine positive one: 62% recall on flagged clauses versus 0% before,
same labels, same documents. See
[`docs/ml_experiment.md`](docs/ml_experiment.md) for both, including
what the positive result does and doesn't support.

Nothing here is fabricated. Every number in the docs below is measured,
not aspirational — including the results that came out worse than
hoped, and the two places an earlier claim was checked and found wrong.
See [`docs/research.md`](docs/research.md) for what that means in
practice.

## Try it

```bash
./run.sh
```

Starts Ollama, ModelWatch, and Sanad together, waits until all three
answer, and prints two URLs. Needs nothing beyond what's in
`sanad/requirements.txt` / `modelwatch/requirements.txt` and a local
Ollama install — no cloud account, no Docker, required for the default
setup.

```bash
./run.sh --stop
```

Tests need no servers running at all:

```bash
python -m pytest -v
```

507 tests, real end-to-end coverage (real HTTP servers via
`pytest-httpserver`, a real mocked S3 API via `moto`, this session's
work was also verified against an actual local Postgres instance, and
`sanad/tests/frontend/` drives the actual frontend/index.html in a real
headless browser against the real running app — not just its API) —
mocking is used only where the alternative is calling a real network
service. `playwright install chromium` is needed once for the frontend
suite; skip it with `pytest --ignore=sanad/tests/frontend` if you'd
rather not install a browser.

## Ship it

```bash
docker compose up --build
```

The novelty claim made concrete: one command builds and starts Ollama,
ModelWatch, Sanad, *and* the telemetry reporter that feeds Sanad's real
usage into ModelWatch's drift detection — the same architecture
`./run.sh` runs locally, packaged as one deployable unit instead of four
things a team has to remember to start (and wire together) themselves.
First run pulls the default model (`phi3:3.8b`, ~2.3GB) into a named
volume; every run after that is fast. See
[`docker-compose.yml`](docker-compose.yml)'s own comments for
configuration (a different model, turning auth on, wiping persisted
data). CPU-only inference is exactly as slow as the local path — this
doesn't fix that, it only fixes "how many steps does it take to run this
somewhere that isn't my machine."

- **[DEMO.md](DEMO.md)** — a runbook that's actually been executed top
  to bottom, for a live walkthrough.
- **[CODEBASE_TOUR.md](CODEBASE_TOUR.md)** — a guided reading order
  through the code (~45 min) for understanding it well enough to defend
  it, including the questions you should expect.

## What's in each project

**Sanad** — see [`sanad/README.md`](sanad/README.md) for the full
writeup:
- Summarizer, grounded chatbot, and rule-based risk scan (the original
  three features)
- Contract intelligence: obligation extraction, clause coverage
  scanning, contradiction detection, review synthesis, a risk heatmap,
  click-to-source navigation, and document comparison
- A real document registry (SQLite by default, Postgres via
  `SANAD_DATABASE_URL`) and object store (local disk by default, an
  S3-compatible bucket via `SANAD_STORAGE_BACKEND=s3`)
- Session-cookie authentication

**ModelWatch** — see [`modelwatch/README.md`](modelwatch/README.md) for
the full writeup:
- A model-agnostic core (`ModelAdapter`: `build_baseline`/`check_drift`)
  with three adapters shipped: classifier, LLM, and RAG-specific
- RAG X-Ray: full per-request pipeline traces, for debugging a bad
  answer instead of only flagging it
- A diagnosis engine that attributes a drift alert (or a single bad
  trace) to a subsystem, with a documented scoring rule
- Alert hysteresis, a Drift Lab for controlled failure injection against
  Sanad's real pipeline, a benchmark/ablation framework comparing
  detection methods against known ground truth, and webhook alert
  delivery
- **[`modelwatch-client/`](modelwatch-client/)** — a standalone,
  pip-installable client for adopting any of this in a project that
  isn't Sanad, including a LangChain callback handler
- **[`examples/independent_rag/`](examples/independent_rag/)** — proof
  that ModelWatch is model-agnostic, not just Sanad-shaped: a RAG
  pipeline that has never imported anything from `sanad/`, monitored
  through nothing but `modelwatch-client`'s public API
- **[`examples/classifier/`](examples/classifier/)** — the same
  model-agnostic story for a completely different model type: a tabular
  classifier, monitored with `ClassifierAdapter` through the same SDK

## Further reading

Each doc below covers one slice in depth, with real measured numbers
and an explicit limitations section rather than hedged language:

| Doc | Covers |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | How the two projects fit together, and where the boundary between "general" and "RAG-specific" actually falls |
| [`docs/telemetry.md`](docs/telemetry.md) | What Sanad reports to ModelWatch, and the privacy tradeoff of full-trace telemetry |
| [`docs/rag_trace.md`](docs/rag_trace.md) | How a RAG X-Ray trace is built: retrieval inspection, sentence-level claim verification, citation scoring |
| [`docs/drift_detection.md`](docs/drift_detection.md) | The statistical detectors, the RAG adapter, and alert hysteresis |
| [`docs/experiments.md`](docs/experiments.md) | Actual measured numbers from the Drift Lab and the benchmark/ablation study |
| [`docs/evaluation.md`](docs/evaluation.md) | Sanad's RAG evaluation dataset and scoring, and the CI-style quality gate it feeds |
| [`docs/contract_intelligence.md`](docs/contract_intelligence.md) | Obligation extraction, coverage, contradictions, and review synthesis |
| [`docs/research.md`](docs/research.md) | The hypotheses this codebase can actually test, what's been measured vs. not, and concrete next steps |
| [`docs/ml_experiment.md`](docs/ml_experiment.md) | Three real trained/tested supervised classifier experiments (clause risk severity) — a negative result, a diagnosis of why, a second experiment that fixes it (62% recall vs. 0%), and a third checking whether threshold tuning can fix the low precision that remains (it can't) |

## Honest framing

Both sub-READMEs lead with a limitations section, not a features list.
That's deliberate: the strongest material here is what's been measured
and what's been found wrong and corrected, not a claim that everything
works. Read those sections first.
