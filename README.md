# Sanad + ModelWatch

Two projects that show each other's failure modes.

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
special case wired into the core.

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

301 tests, real end-to-end coverage (real HTTP servers via
`pytest-httpserver`, a real mocked S3 API via `moto`, and this session's
work was also verified against an actual local Postgres instance) —
mocking is used only where the alternative is calling a real network
service.

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

## Honest framing

Both sub-READMEs lead with a limitations section, not a features list.
That's deliberate: the strongest material here is what's been measured
and what's been found wrong and corrected, not a claim that everything
works. Read those sections first.
