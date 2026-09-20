# Demo runbook

Everything below has been run end to end. Follow it top to bottom.

## 0. Start (do this ~5 min before)

One command starts Ollama, ModelWatch and Sanad together, waits until all
three answer, and prints the URLs:

```bash
./run.sh
```

To stop them again:

```bash
./run.sh --stop
```

`run.sh` frees the ports before starting, so re-running it is always safe.
That also clears the one failure mode worth knowing about: **if Sanad
returns 500 on upload**, its data directory was removed while it was
running — just re-run `./run.sh`.

Two tabs open: `http://localhost:8100/` (Sanad) and
`http://localhost:8000/dashboard/` (ModelWatch).

## 1. The tests (2 min)

Lead with this — it's the strongest part.

```bash
source .venv/bin/activate && python -m pytest -v
```

Point at `modelwatch/tests/test_classifier_adapter.py`: the drift tests use
data generated from a *known* distribution, so whether drift should fire is
known before the test runs. Not "it didn't crash".

## 2. Sanad: upload → orient → flag risk → ask (8 min)

At `http://localhost:8100/`, sign in with `demo` / `demopass123` if prompted.

1. Upload `sanad/sample_docs/rental/rental_agreement_sample_1.pdf`, type
   "Rental". Note the badges: **18 chunks indexed**, **native text**.
   Mention the OCR fallback exists for scanned pages and is tested.
2. **Risk scan tab (opens by default) — lead with this, not Summary.**
   Point at the donut + stat tiles first: real per-clause severity counts,
   not a fabricated "safety score" (the ring is built from the exact same
   data the heatmap below it uses — one shared computation, two views).
   Click a heatmap cell or a risk card's "View in document →" to show
   click-to-source jumping straight to the flagged clause.
3. **Overview tab** — 15 key fields (parties, dates, payment, termination,
   etc.), each `found`/`not_found`/`unclear`/`insufficient_evidence`, plus
   the Knowledge Map (hub-and-spoke SVG built from the same grounded data,
   nothing new inferred). This is a real LLM job — **pre-warm it before
   presenting** (see "Before you present," below) since it can take over
   a minute on a local CPU-only model.
4. **Summary tab** — Generate Summary, ~30s–2min depending on load.
   Structured fields, not a paragraph blob: parties, obligations, dates,
   notice period, penalties, termination.
5. **Ask tab** — ask a question from the verified list below. Point out
   the "Asking about *filename*" context chip, then ask something the
   document does *not* cover, e.g. *"What is the visitor parking policy?"*
   → it refuses instead of inventing. Open the citation disclosure and the
   "AI / RAG trace" on a grounded answer to show the retrieved clause and
   per-sentence claim verification.
6. **Review tab** — obligations, coverage gaps, and contradictions
   synthesized into one report with suggested negotiation questions. Also
   a real LLM job — pre-warm it too.
7. If you uploaded a second document earlier, show **Compare** (semantic-
   impact diff of the two risk profiles) and **Versions** (upload a
   revised copy of the same contract, jump straight into Compare against
   a specific prior version).

The refusal is the point worth dwelling on. An answer with no valid citation
is downgraded to a refusal in `features/chatbot.py` — the system fails
toward "I don't know" rather than toward confident invention.

**Before you present:** Overview, Summary, and Review each trigger a real
local LLM call and cache their result *in the browser tab* only (no
server-side cache, and it resets on page reload). Click through all three
tabs for your demo document(s) once, right before you go on, so they're
already rendered when you get there live — and never trigger two of these
jobs at once (a local model serves one request at a time; a second job
queued behind the first can hit Ollama's 180s timeout).

## 3. ModelWatch dashboard (3 min)

At `http://localhost:8000/dashboard/`. If empty, seed a classifier:

```bash
curl -s -X POST http://localhost:8000/models -H 'Content-Type: application/json' -d '{"model_id":"demo-classifier","name":"Loan Approval Classifier","adapter_name":"classifier","baseline_data":{"features":{"age":[23,45,31,29,38,42,27,35],"income":[50000,62000,48000,71000,55000,66000,51000,59000]}}}'
```

```bash
curl -s -X POST http://localhost:8000/models/demo-classifier/check -H 'Content-Type: application/json' -d '{"new_data":{"features":{"age":[72,78,69,75],"income":[120000,131000,118000,127000]}}}'
```

Refresh: drift spikes, an alert appears. Click the drifted row for the
per-feature KS statistics and p-values.

(If asked "how would I connect my own model?": point to the dashboard's
own "Connect Your Model" page, sidebar under Systems — it shows the same
registration snippet plus the request schema for each real adapter.)

The architectural claim: `core/engine.py` never imports scipy or sklearn and
has no model-type branching. It only calls `build_baseline` / `check_drift`
on the `ModelAdapter` interface. Same engine, same dashboard, for a tabular
classifier and for an LLM chatbot.

## 4. The drift story — the real answer to "does ModelWatch actually work?"

**Not optional — lead with this if the question "how do you know ModelWatch
detects real degradation" comes up.** This is a controlled experiment, not a
staged number: it makes Sanad genuinely worse (swaps its live model), proves
ModelWatch catches it, then proves it recovers.

```bash
source .venv/bin/activate && python -m modelwatch.examples.simulate_drift_demo --drift-model qwen2.5:0.5b --limit 5
```

Have `http://localhost:8000/dashboard/` open in a tab, model picker set to
**"Sanad RAG Chatbot · llm"** (not "Sanad Chatbot (live traffic)" — that's a
different model, fed by real usage, not this controlled run). Run the command,
then refresh the dashboard:

| step | quality | drift score | status | open alerts |
|---|---|---|---|---|
| 1. baseline (phi3:3.8b) | ~0.50 | ~0.49 | Healthy | 0 |
| 2. swapped to qwen2.5:0.5b | **~0.14** | **~0.87** | **Degraded — drift detected** | **1** |
| 4. retrain (swap back to phi3, reset baseline) | — | — | version bumps v1→v2 | alert resolved |
| 5. follow-up confirms recovery | ~0.50 | ~0.50 | Healthy | 0 |

(Exact numbers vary run to run — local model sampling isn't deterministic —
but the *shape*, a sharp quality drop crossing the 0.35 alert threshold and a
clean recovery, has been reproduced multiple times.)

What to actually show on stage, in order:
1. **Overview page**, model = sanad-chatbot, after step 2: the red "Degraded —
   drift detected" banner, and the Quality & Drift chart's dashed alert-
   threshold line being crossed — this is the single clearest visual in the
   whole project.
2. **Active Incidents**: the alert, then empty again after the retrain step —
   proves detection *and* resolution, not just detection.
3. **Models & Versions**: `sanad-chatbot`'s version history shows `v1 → v2`
   with reason `retrain` and a real timestamp — a persisted record, not a
   toast that already disappeared.

If short on time, show the table above instead of running it live — but
running it live is the stronger claim, and it now reliably works (see
gotchas below).

### Gotchas that will bite you if you skip this section

- **Auth must be off** for the script to reach Sanad (`session.get`/`post`
  calls in `sanad_golden_set_runner.py` don't log in). If your `.env` has
  `SANAD_AUTH_ENABLED=true`, temporarily `mv .env .env.bak`, `./run.sh
  --stop && ./run.sh`, run the demo, then `mv .env.bak .env` and restart
  again afterward.
- **Close other memory-heavy apps first** (other Claude Code sessions,
  Chrome with many tabs, etc.). This script loads sentence-transformers,
  ChromaDB, and calls a local LLM all at once — under severe memory
  pressure, Sanad's process can be killed outright mid-request. Check
  `memory_pressure` if anything seems to hang; a few hundred MB free is
  enough.
- **If Sanad crashes with no error message at all** (the process just
  disappears), check `~/Library/Logs/DiagnosticReports/Python-*.ips` for a
  `SIGSEGV` inside `chromadb_rust_bindings.abi3.so`. This happened once
  during real testing — root cause was a `sanad_chroma_db` left in a bad
  state by an earlier crashed run, not this script. Fix: stop the app,
  move `sanad_chroma_db` and `sanad_documents.db` aside (they're
  regenerable local data, not source), restart. Should not recur on a
  vector store that's never been through a mid-write crash.

## Verified demo questions

Last probe on `phi3:3.8b` scored 5/7. **Use these — they answered:**

| Document | Question |
|---|---|
| `rental/rental_agreement_sample_1.pdf` | Who pays for major structural repairs? |
| `rental/rental_agreement_sample_1.pdf` | When is the monthly rent due? |
| `rental/rental_agreement_sample2.pdf` | What is the term of this lease? |
| `freelance/freelance_agreement_sample2.pdf` | What law governs this agreement? |
| `freelance/service_agreement_sample1.pdf` | How are disputes resolved under this agreement? |

**Avoid these two — they currently refuse** even though the answer is in the
document, and retrieval was verified to be surfacing the right clause:

- *"What is my notice period?"*
- *"Does this agreement create an employer-employee relationship?"*

For the refusal demo use *"What is the visitor parking policy?"* — genuinely
absent from the document, so refusing is the correct behaviour.

A local 3.8B model is not deterministic, so any single question can flip
between runs. **If one refuses on stage:** say plainly that it is a measured,
documented limitation and move on. That reads far better than looking
surprised — and you can point at the exact number (0.536 average similarity,
losses dominated by false refusals) to show it was quantified, not hand-waved.

## Honest limitations (have these ready)

The prof will probe. These are measured, not hedges:

- **Chatbot over-refuses.** Golden-set answer similarity averages 0.536 on
  phi3:3.8b. The losses are mostly false refusals, not wrong answers.
- **One cause was ours, one is the model's.** A retrieval bug (`top_k=4`
  hid a clause that ranked 6th) was found and fixed → 0.458 to 0.536. The
  rest is the small local model failing to cite what it was given.
- **The LLM drift detector has a ~0.05 noise floor.** Three runs of the
  *same* model scored 0.539 / 0.491 / 0.497. So it cannot distinguish
  phi3 from llama3.2:3b (a ~0.02 gap) — which is why the drift demo uses a
  0.5B model, rather than a threshold tuned until the demo passed.
- **TF-IDF is lexical, not semantic.** A correct paraphrase can score worse
  than a wrong answer reusing the same words.
- **No multi-turn chat.** Each question is answered independently, with
  no conversation history.

## 5. If there's real time left: the research extension

Everything above still applies unchanged. This is additional depth
built on top of it — see `docs/` for the full writeup.

```bash
# real statistical drift detection (KS, Wasserstein, two-proportion
# z-test) over Sanad's structured per-request telemetry, plus root-cause
# diagnosis, replacing step 3's threshold-only classifier demo:
curl -s http://localhost:8000/models/sanad-live/health   # HEALTHY/WARNING/DEGRADED/RECOVERING

# controlled intervention on Sanad's real pipeline (needs Ollama):
python -m modelwatch.experiments.run_drift_lab retrieval_narrowing --limit 10

# does multidimensional monitoring actually catch more than a single
# threshold? (synthetic trials, real measured numbers, see docs/experiments.md)
python scripts/run_benchmark.py --n-trials 300
python scripts/run_benchmark.py --ablation --n-trials 300

# CI-style regression gate against a real evaluation dataset:
python scripts/quality_gate.py --save-baseline   # once
python scripts/quality_gate.py                   # every run after
```

Say plainly what this is and isn't: the benchmark/ablation numbers are
from **synthetic trials with known ground truth**, not live traffic —
see `docs/research.md`'s limitations section, which also lists what
hasn't been measured yet (detection delay, hysteresis-adjusted false
positive rate). That honesty is itself part of the answer if asked "is
this rigorous" — the repo says clearly what has and hasn't been shown.

## If something breaks

- Sanad 500 on upload → restart the Sanad process.
- "Ollama … not found" → `ollama serve`, and `ollama pull phi3:3.8b`.
- Dashboard empty → no models registered yet; run the curl in step 3.
- Everything else → fall back to `pytest`, which needs no servers at all.
