# Architecture

For the guided code tour (read-this-file-order for learning the codebase),
see [`CODEBASE_TOUR.md`](../CODEBASE_TOUR.md). This document is the
reference map: what exists, why it's split this way, and where the line
between "generic framework" and "Sanad-specific" actually sits.

## The two systems

**Sanad** ([`sanad/`](../sanad/)) is a RAG contract-analysis application:
upload a PDF, ask grounded questions about it, get a rule-based risk scan,
compare two contracts' risk profiles. See [`sanad/README.md`](../sanad/README.md).

**ModelWatch** ([`modelwatch/`](../modelwatch/)) is a model-agnostic
monitoring framework: given an adapter that knows how to build a baseline
and check new data against it, it persists runs, raises/resolves alerts
with hysteresis, and can diagnose likely root cause. See
[`modelwatch/README.md`](../modelwatch/README.md).

Sanad has zero import-time dependency on ModelWatch. It publishes an
operational telemetry buffer (`sanad/api/telemetry.py`) and answers
`GET /api/telemetry`; it has no idea anything reads that endpoint. A
separate reporter process (`modelwatch/examples/telemetry_reporter.py`)
polls it and forwards to ModelWatch's API. Deleting ModelWatch entirely
would not break Sanad.

ModelWatch's core (`modelwatch/core/`, `modelwatch/adapters/`,
`modelwatch/drift/`, `modelwatch/diagnosis/`) has zero import-time
dependency on Sanad -- the engine only ever calls the `ModelAdapter`
interface. The one deliberate exception is
`modelwatch/experiments/drift_lab.py`, which exists specifically to run
controlled experiments against Sanad's real pipeline; it's documented as
an exception in its own module docstring, in the same place
`modelwatch/examples/*.py` already was.

## Module map

```
sanad/
  api/            FastAPI app, auth, telemetry buffer, request/response schemas
  ingestion/      PDF/OCR extraction, clause-aware chunking
  rag/            embeddings, ChromaDB vector store, LLM client abstraction
  features/       chatbot (grounded Q&A), summarizer, risk_flagger, comparison
  evaluation/     dataset loader, deterministic + embedding-based metrics,
                  in-process runner, CLI (see docs/evaluation.md)
  frontend/       vanilla HTML/CSS/JS contract workspace

modelwatch/
  core/           adapter_base (the interface), engine, storage (SQLite),
                  health (alert hysteresis state machine)
  adapters/       classifier, llm (golden-set), live_telemetry, rag
  drift/          detectors.py -- KS, Wasserstein, PSI, two-proportion z-test
                  (see docs/drift_detection.md)
  diagnosis/      root-cause ranking over a drifted run's signals
  experiments/    drift_lab.py, benchmark.py (see docs/experiments.md)
  api/            FastAPI app, adapter registry, request/response schemas
  dashboard/      vanilla HTML/CSS/JS observability console
  examples/       operator scripts (telemetry_reporter, golden-set runner,
                  drift-simulation demo)

datasets/sanad_eval/   the RAG evaluation dataset (docs/evaluation.md)
experiments/results/   benchmark/quality-gate output, gitignored-scale JSON
scripts/               run_benchmark.py, quality_gate.py
```

## Why an adapter interface

`modelwatch/core/adapter_base.py`'s `ModelAdapter` is the one seam the
engine depends on: `build_baseline()` and `check_drift()`. The engine
imports no scipy, no sklearn, and never branches on model type. Four
adapters exist today (classifier, llm, live_telemetry, rag) and each
implements the same two methods completely differently -- KS tests over
tabular features, TF-IDF similarity over a golden Q&A set, tolerance
checks over aggregate rates, and (the newest) four real statistical tests
over raw per-request RAG telemetry. Adding a fifth model type is a new
adapter file plus one registry line
(`modelwatch/api/adapter_registry.py`), never an engine change.

## Documents

- [contract_intelligence.md](contract_intelligence.md) -- obligation/deadline
  extraction, missing-clause detection, contradiction detection, and the
  Review synthesis
- [rag_trace.md](rag_trace.md) -- making the AI visible: retrieval inspector,
  claim verification, the RAG X-Ray, and per-request diagnosis
- [telemetry.md](telemetry.md) -- the structured per-request event schema
- [evaluation.md](evaluation.md) -- the Sanad RAG evaluation dataset + engine
- [drift_detection.md](drift_detection.md) -- statistical detectors, RAGAdapter,
  alert hysteresis, root-cause diagnosis
- [experiments.md](experiments.md) -- Drift Lab, the experiment registry, the benchmark
- [research.md](research.md) -- the research question this is actually built to
  investigate, what's been measured so far, and what hasn't
