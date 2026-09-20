# Codebase tour

For learning the project well enough to defend it. Read in this order —
roughly 45 minutes.

## Running it without Claude Code

Open VS Code, `File > Open Folder` → `SWE_PROJ`. Then `Terminal > New
Terminal` and:

```bash
./run.sh
```

That's it. The script `cd`s to its own directory and activates the venv
itself, so it works from any terminal, in or out of VS Code. It prints two
URLs — open both in a browser.

```bash
./run.sh --stop
```

If you'd rather see live logs (better for a demo — you can point at
requests arriving), run the two servers in the foreground in separate
terminals instead:

```bash
source .venv/bin/activate && uvicorn sanad.api.app:app --port 8100
```

```bash
source .venv/bin/activate && uvicorn modelwatch.api.app:app --port 8000
```

Ollama must be running either way (`ollama serve`, or the desktop app).

Tests need no servers at all:

```bash
source .venv/bin/activate && python -m pytest sanad/tests/ modelwatch/tests/ -v
```

## The one-sentence version

> Sanad is a RAG app that answers questions about uploaded contracts using
> only the contract's own text; ModelWatch is a separate, reusable
> framework that watches any model for silent quality degradation, and
> Sanad's chatbot is its first real integration.

## Read in this order

### 1. The load-bearing idea (10 min)

**`modelwatch/core/adapter_base.py`** — start here. It's short. Two
methods, `build_baseline` and `check_drift`, plus two result types.

**`modelwatch/core/engine.py`** — then this. The thing to notice: it
imports no scipy, no sklearn, and never branches on model type. It only
calls the two methods above. That is the whole architectural claim —
*the engine cannot tell a tabular classifier from an LLM*.

Now skim both adapters and see how differently they implement the same
interface:

- **`adapters/classifier_adapter.py`** — KS test per feature, accuracy
- **`adapters/llm_adapter.py`** — golden Q&A set, TF-IDF similarity

> **Likely question:** *"Why an interface instead of just two functions?"*
> Because adding a third model type must not require editing the engine,
> the storage layer, the API, or the dashboard. Adding one adapter file
> plus one line in `api/adapter_registry.py` is the whole change.

### 2. How Sanad answers a question (15 min)

Follow one request end to end:

1. **`sanad/api/app.py`** → `chat_with_document` — the HTTP entry point
2. **`sanad/features/chatbot.py`** → `ask()` — the real logic
3. **`sanad/rag/vector_store.py`** → `query()` — retrieves top-k chunks,
   scoped to one `doc_id`
4. Back in `chatbot.py` → `_build_user_prompt`, then `_parse_answer`

The part worth understanding deeply is `_parse_answer`. Groundedness is
decided by **whether the model cited a real excerpt**, not by the
`grounded` boolean the model reports about itself. A citation is
checkable; a self-assessment isn't. This was a real bug: phi3 returned
`"grounded": false` alongside a correct, correctly-cited answer, so good
answers rendered as refusals.

### 3. The ingestion pipeline (10 min)

- **`sanad/ingestion/extraction.py`** — PyMuPDF per page; if a page has no
  text layer, Tesseract OCR that page. Mixed documents work.
- **`sanad/ingestion/chunking.py`** — splits on *clause boundaries*
  (numbered clauses, `(a)` sub-clauses, ALL-CAPS headers), not fixed
  character windows. Legal documents have real structure; using it keeps
  one clause per chunk instead of cutting obligations in half.
- **`sanad/rag/pipeline.py`** — the shared extract → chunk → index path.
  Both features call this; neither re-implements it.

### 4. The integration (10 min)

- **`modelwatch/examples/golden_set.py`** — 22 question/answer pairs across
  all three contract types, each grounded in real sample-document text
- **`modelwatch/examples/sanad_golden_set_runner.py`** — uploads the docs,
  asks Sanad each question, scores the answers, reports to ModelWatch
- **`modelwatch/examples/simulate_drift_demo.py`** — baseline → swap model
  → alert → retrain → recover

### 5. The research extension (10 min, if you have it)

Built on top of everything above, without changing any of it:

- **`sanad/api/telemetry.py`** — the event schema grew richer (retrieval
  distances, split retrieval/generation latency, citation-validity
  counts), still with zero question/answer/chunk text. See
  `docs/telemetry.md`.
- **`modelwatch/drift/detectors.py`** — real KS/Wasserstein/PSI/
  two-proportion-z-test functions, decoupled from any adapter.
- **`modelwatch/adapters/rag_adapter.py`** — a fourth adapter
  (`adapter_name="rag"`) that runs those tests against the richer
  telemetry: five independent signals (retrieval, generation latency,
  refusal, citation validity, and a question-embedding distribution
  shift via Maximum Mean Discrepancy — `embedding_drift()` in
  `detectors.py`), never collapsed into one number. `live_telemetry`
  still works unchanged for anything already registered against it.
- **`modelwatch/core/health.py`** — alert hysteresis
  (healthy→warning→degraded→recovering→healthy), defaulted to reproduce
  the original one-shot alerting exactly. `modelwatch/core/calibration.py`
  and `scripts/calibrate_hysteresis.py` turn the hysteresis threshold
  into something computed from a measured false-positive rate rather
  than hand-picked.
- **`modelwatch/diagnosis/engine.py`** — given a drifted run's signals,
  ranks which subsystem (retrieval/generation/operational) is the likely
  cause, with a documented (not black-box) scoring rule. Shows up in the
  dashboard as "Why this alert?".
- **`modelwatch/experiments/drift_lab.py`** and **`benchmark.py`** — real
  controlled interventions on Sanad's live pipeline, and a real
  comparison of detection methods on synthetic trials with known ground
  truth. See `docs/experiments.md` for actual measured numbers from both.
- **`sanad/evaluation/`** — a 22-case RAG evaluation dataset and a
  deterministic + embedding-based scoring engine, feeding
  `scripts/quality_gate.py` (a real CI-style regression gate).

Full writeup, including what's been measured and what hasn't (detection
delay is still open; hysteresis-adjusted false-positive rate has a real
calibrated number now, but the independence assumption behind it hasn't
been verified against real sequential traffic), in **`docs/research.md`**.

## Questions you should be ready for

**"Why is the chatbot wrong so often?"**
It over-refuses rather than inventing — a deliberate trade. Measured 0.536
average similarity on the golden set. Two distinct causes were separated
by checking retrieval directly: one retrieval bug we fixed (`top_k=4` hid
a clause ranked 6th, → 0.458 to 0.536), and the 3.8B model failing to cite
what it was given. A larger model would recover much of the rest.

**"Isn't the classifier data fake?"**
Yes, and say so first. `demo-classifier` is synthetic — the project has no
real tabular model. It exists to prove the adapter interface works for a
second model type. `sanad-chatbot` is the real integration.

**"How do you know the drift detector works?"**
`modelwatch/tests/test_classifier_adapter.py` — data generated from a
*known* distribution, so the correct answer is known before the test runs.
Plus a measured false-alarm rate: 13.2% on a 2-feature model, cut to 6.4%
with a Bonferroni correction.

**"Why TF-IDF and not embeddings for the LLM adapter?"**
To keep ModelWatch installable without torch, so it stays independent of
whatever stack the monitored app uses. The cost is that it's lexical, not
semantic — a correct paraphrase can score worse than a wrong answer that
reuses the same words. That's in the README.

**"What would you do next?"**
Average several runs per check to lower the detector's noise floor; store
the baseline's achieved quality so drift can be measured *relatively*
instead of against a hand-set threshold; a bigger local model.

**"Is there any actual deep learning in this project, or did you just
train nothing?"**
Say both halves. Sanad does real DL *inference* (a pretrained sentence-
transformer for embeddings, a pretrained LLM for generation) — no model
is trained from scratch anywhere in this repo. ModelWatch is deliberately
*not* deep learning (`modelwatch/core/engine.py` imports no scipy/sklearn
even; only its adapters do) — classical statistics, on purpose, so it can
watch a model without loading one itself. The one place a real supervised
model *is* trained and tested with a held-out procedure is `ml/`, and
it's genuinely four experiments, documented in `docs/ml_experiment.md`.
The first (`train_risk_classifier.py`: TF-IDF, one 75/25 split) is a
negative result — with only a handful of positive training examples
this repo's sample contracts provide, the model is statistically
identical to a majority-class baseline. Rather than stopping there, the
second (`train_embeddings_loocv.py`: sentence embeddings, leave-one-out
CV) fixes the two things actually holding the first one back —
representation and evaluation efficiency, not a retrained-until-lucky
number — and gets a real positive result: roughly 60% recall on flagged
clauses vs. 0% before, same labels, same documents (this exact number
has moved between 58-62% as the dataset grew across three sizes — say
that unprompted too, don't present a single snapshot as fixed).
Precision is still low (0.20), so it's a recall-oriented pre-filter, not
a deployable detector — say that unprompted, don't wait to be asked. The
obvious next question, "can you just raise the threshold to fix
precision," was actually tried, not just claimed impossible:
`threshold_tuning.py` sweeps the cutoff, and on the original small
dataset found 0.5 was already close to the best F1 available. The
fourth experiment then pulled the two remaining honest levers —
LLM-paraphrase augmentation of the existing flagged clauses, evaluated
with leave-one-**group**-out CV so a paraphrase can't leak into training
while its own source clause is held out, plus more real documents added
in two separate rounds (10 → 18 → 32 documents, 13 → 16 → 19 real
positives) specifically to grow the positive class. Augmentation vs. no
augmentation on the same dataset reliably roughly doubles precision
(0.20 → 0.50 currently); re-running the threshold sweep on the larger
dataset also reversed the original negative finding, with the best
threshold moving up as data grew (0.5 → 0.6 → 0.7 across three sizes).
But — say this part unprompted too, it's the most defensible thing about
this project's ML story — adding more real documents was *not* a clean
monotonic win: precision was actually higher at 18 documents (0.56) than
at 32 (0.50), because the second round's new categories (power of
attorney, corporate resolutions) are more different from the original
rental/employment/freelance clauses the rule engine was tuned against.
Reporting that regression honestly, rather than only keeping the
better-looking earlier number, is the actual point of running four
experiments instead of stopping at the first flattering one.

**"You said ModelWatch is Sanad-independent — is it, really?"**
It wasn't, technically, until this was caught: `modelwatch/api/app.py`
had a hard top-level `from sanad.jobs import JobManager`, so ModelWatch's
own API couldn't start without Sanad's package (and its full dependency
list) being importable, despite that class being pure stdlib with zero
Sanad-specific logic. Fixed by moving it to `shared/jobs.py`, a module
neither app depends on the other to reach. Worth naming as an example of
auditing your own claim and finding it wrong, not assuming it holds.

## Honest framing

The strongest material is the limitations sections in both READMEs. They
contain measured numbers, not hedges — including two places where an
earlier claim was wrong and got corrected once it was actually checked.
Lead with that. It's the difference between "it works" and "here is what
it does, here is what it doesn't, and here is how I know."
