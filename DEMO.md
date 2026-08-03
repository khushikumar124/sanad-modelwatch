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
source .venv/bin/activate && python -m pytest sanad/tests/ modelwatch/tests/ -v
```

Point at `modelwatch/tests/test_classifier_adapter.py`: the drift tests use
data generated from a *known* distribution, so whether drift should fire is
known before the test runs. Not "it didn't crash".

## 2. Sanad: upload → summarise → ask (5 min)

At `http://localhost:8100/`:

1. Upload `sanad/sample_docs/rental/rental_agreement_sample_1.pdf`, type
   "Rental". Note the badges: **18 chunks indexed**, **native text**.
   Mention the OCR fallback exists for scanned pages and is tested.
2. **Generate Summary** — ~30 s. Structured fields, not a paragraph blob:
   parties, obligations, dates, notice period, penalties, termination.
3. Ask a question from the verified list below.
4. Ask something the document does *not* cover, e.g.
   *"What is the visitor parking policy?"* → it refuses instead of inventing.
   Then open the sources disclosure on a grounded answer to show the clause
   it used.

The refusal is the point worth dwelling on. An answer with no valid citation
is downgraded to a refusal in `features/chatbot.py` — the system fails
toward "I don't know" rather than toward confident invention.

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

The architectural claim: `core/engine.py` never imports scipy or sklearn and
has no model-type branching. It only calls `build_baseline` / `check_drift`
on the `ModelAdapter` interface. Same engine, same dashboard, for a tabular
classifier and for an LLM chatbot.

## 4. The drift story (optional — 8 min, only if time)

```bash
source .venv/bin/activate && python -m modelwatch.examples.simulate_drift_demo --drift-model qwen2.5:0.5b --limit 5
```

Measured result at the default threshold:

| step | quality | drifted | alerts |
|---|---|---|---|
| baseline (phi3:3.8b) | 0.558 | no | 0 |
| swapped to qwen2.5:0.5b | **0.261** | **yes** | 1 |
| after retrain, back on phi3 | 0.517 | no | 0 |

If short on time, show this table instead of running it live.

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
- **No multi-turn chat**, in-memory document registry, no auth.

## If something breaks

- Sanad 500 on upload → restart the Sanad process.
- "Ollama … not found" → `ollama serve`, and `ollama pull phi3:3.8b`.
- Dashboard empty → no models registered yet; run the curl in step 3.
- Everything else → fall back to `pytest`, which needs no servers at all.
