# Sanad RAG evaluation

## The dataset

`datasets/sanad_eval/sanad_eval_v1.jsonl` -- 22 real questions over
Sanad's real sample contracts (rental, employment, freelance/service),
built from the existing hand-verified golden set
(`modelwatch/examples/golden_set.py`, whose `expected_answer`s were each
checked against the actual contract text, not guessed).

**DEMO / SYNTHETIC ground truth, clearly labelled as such**: each case's
`relevant_chunks` (which chunk of the real chunked document supports the
answer) is auto-derived by `datasets/sanad_eval/build_dataset.py` via
lexical-overlap matching between `expected_answer` and the document's
real chunks -- not a human reading the contract and marking the right
chunk. This is a heuristic, and can be wrong for an answer that
paraphrases heavily instead of echoing contract vocabulary. It's useful
for exercising the evaluation engine and for regression-testing "did
retrieval/citation quality change" -- it is not a substitute for
human-annotated ground truth if these numbers are ever cited as evidence
about a production model. See `EvalCase` in
`sanad/evaluation/dataset.py` for the exact field-by-field caveat.

Categories present: `compensation`, `security_deposit`, `notice_period`,
`obligations`, `termination`, `duration`, `intellectual_property`,
`employment_status`, `working_restrictions`, `governing_law`,
`dispute_resolution`, `probation`.

## The evaluation engine

`sanad/evaluation/` runs the dataset through Sanad's *actual* production
code path (`sanad.features.chatbot.ask()` against a real `VectorStore`,
in-process, no HTTP) and scores each case. Metrics are labelled by how
they're computed (`sanad/evaluation/metrics.py`'s docstring):

- **deterministic** -- exact set/boolean comparisons, no model in the
  loop: `retrieval_hit` (was a ground-truth chunk actually retrieved?),
  `retrieval_rank`, `citation_correct` (did the answer cite a chunk that
  actually supports it?), `refused`, `parse_error`, latency splits.
- **embedding-based** -- cosine similarity of sentence-transformer
  embeddings between `expected_answer` and the actual answer. Not exact,
  but reproducible.

There is deliberately **no LLM-judge metric**. Every RAG evaluation
framework eventually reaches for "ask another LLM whether this answer is
good", and it was left out on purpose: it would introduce a second
model's own quality/cost/non-determinism into what is otherwise a fully
reproducible evaluation. If one is added later, it must be behind an
explicit flag, never a silent default.

`sanad/evaluation/aggregate.py` rolls per-case results into
`EvalSummary`: overall rates plus a per-category breakdown, so a
regression concentrated in one clause type doesn't get diluted into a
moving-but-not-alarming overall number.

## Running it

```bash
python -m sanad.evaluation.run_eval                      # full dataset against Ollama
python -m sanad.evaluation.run_eval --out results.json    # save full per-case results
```

## A real measured run

Captured 2026-08-30 against `phi3:3.8b` (Sanad's default model), full
22-case dataset, via `scripts/quality_gate.py --save-baseline`:

| metric | value |
|---|---|
| retrieval hit rate | 0.955 |
| citation correctness | 0.824 |
| mean semantic similarity | 0.668 |
| refusal rate | 0.227 |

These are one run's numbers, not an average over repeated runs -- LLM
sampling means a second run will differ somewhat (this is exactly why
`scripts/quality_gate.py` uses tolerances rather than exact-match
comparison; see [drift_detection.md](drift_detection.md) for the same
principle applied to live monitoring). Treat them as a snapshot, not a
claim about long-run model behavior.

## Testing the engine without Ollama

`sanad/tests/test_evaluation.py` exercises the same engine with a
scripted `FakeLLMClient`, so the evaluation *mechanics* (does
`score_case` correctly flag a retrieval hit/miss, does aggregation
exclude refused answers from the citation-correctness denominator) are
covered by fast, deterministic CI tests that need no Ollama.
