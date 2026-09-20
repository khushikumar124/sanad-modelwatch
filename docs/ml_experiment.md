# Risk-severity classifier: three experiments, one fix that actually mattered

See [`ml/README.md`](../ml/README.md) for what this experiment is and,
just as importantly, what it isn't claiming. In short: does a lightweight
learned model reproduce `sanad/features/risk_flagger.py`'s severity
labels from clause text alone, on clauses it never saw during training?

**Three experiments, run in this order, all real:**

1. **TF-IDF + a single 75/25 split** ([`ml/train_risk_classifier.py`](../ml/train_risk_classifier.py))
   — **negative result**. The learned model was statistically
   indistinguishable from a majority-class baseline.
2. **Sentence embeddings + leave-one-out cross-validation** ([`ml/train_embeddings_loocv.py`](../ml/train_embeddings_loocv.py))
   — **positive result**. Logistic regression on embeddings catches 62%
   of flagged clauses (vs. 0% for the baseline) on the binary task.
3. **Threshold tuning on experiment 2's own probabilities** ([`ml/threshold_tuning.py`](../ml/threshold_tuning.py))
   — asks whether precision (0.18 in experiment 2) can be improved for
   free, just by moving the decision cutoff. **Negative result**: 0.5
   turns out to already be close to the best F1 available on this data:
   raising the threshold barely moves precision while recall collapses.

All three are kept and reported, not just the flattering one, because
*why* the first one failed and what specifically fixed it is the actual
finding — a positive number on its own, with no failed attempt to
contrast it against, would tell you nothing about which part of the fix
mattered. Same reasoning for keeping experiment 3's negative result
rather than quietly dropping the script that didn't help.

## Setup

```bash
python -m ml.build_dataset               # -> ml/data/clause_risk_dataset_v1.jsonl
python -m ml.train_risk_classifier       # experiment 1 -> ml/artifacts/{risk_classifier.joblib,results.json}
python -m ml.train_embeddings_loocv      # experiment 2 -> ml/artifacts/results_embeddings_loocv.json
python -m ml.threshold_tuning            # experiment 3 -> ml/artifacts/results_threshold_sweep.json
```

**Dataset** (shared by all three experiments): every PDF under
`sanad/sample_docs/` (10 real Indian rental, freelance, employment, and
legal-services documents), extracted, chunked by the same clause-aware
chunker the live app uses, and labeled by running the real rule engine
over each chunk.

| | count |
|---|---|
| Total clauses | 584 |
| `none` | 571 |
| `high` | 7 |
| `medium` | 4 |
| `low` | 2 |
| Binary `flagged` (high+medium+low) | 13 |

Only **13 of 584 clauses are positive examples** — the rule engine fires
rarely by design (it's deliberately conservative; see the root README's
limitations section). That scarcity is the real story behind both
experiments below.

## Experiment 1: TF-IDF + single split (negative)

**Model**: TF-IDF (word 1-2 grams) + logistic regression,
`class_weight="balanced"`, `random_state=0`, evaluated on a 75/25 split
(stratified where possible) — 146 test clauses, only 3 of them positive.

| 4-class | learned model | majority-class baseline |
|---|---|---|
| accuracy | 0.979 | 0.979 |
| macro-F1 | 0.247 | 0.247 |

| Binary (flagged/none) | learned model | majority-class baseline |
|---|---|---|
| accuracy | 0.979 | 0.979 |
| macro-F1 | 0.495 | 0.495 |
| `flagged` recall | 0.00 | 0.00 |

Identical to the baseline to three decimal places — the model predicted
`none` for every single test example. Checked against three
configurations (unigrams-only, higher-regularization, multinomial Naive
Bayes) — all three scored zero correctly-identified flagged clauses,
ruling out "bad hyperparameter choice" as the explanation. A
probability-ranking diagnostic confirmed it wasn't even close: the
true-flagged test clauses ranked 18th, 38th, and 50th out of 146 by
predicted P(flagged), behind ten false clauses the model was more
confident about. Full detail in this file's git history / the original
write-up structure below.

**Why**: TF-IDF needs near-exact vocabulary overlap between training and
test clauses to generalize. With ~10 positive training examples spread
across different risk categories and phrasings, most categories are
represented by 1-2 examples each — not enough for a bag-of-words model
to learn anything transferable.

## Experiment 2: sentence embeddings + leave-one-out CV (positive)

Two changes from experiment 1, not a retry with different random seeds:

1. **TF-IDF → sentence embeddings** (`all-MiniLM-L6-v2`, the same model
   Sanad's own retrieval already uses — see [`ml/embed.py`](../ml/embed.py)).
   Lets a match fire on semantic similarity instead of shared vocabulary.
2. **A single split → leave-one-out cross-validation.** With 13 positive
   examples total, a single 75/25 split tests on ~3 of them — one of the
   least statistically meaningful ways to evaluate this dataset. LOO-CV
   evaluates every one of the 584 clauses exactly once, trained on all
   583 others each time, so no single lucky/unlucky split determines the
   result.

Three genuinely different model families were tried, not the same model
tuned three ways: a parametric classifier (logistic regression) and two
similarity-based few-shot approaches (1-nearest-neighbor, and 5-nearest
distance-weighted), since few-shot/similarity search is generally the
more appropriate family when there are only a handful of positive
examples.

**Binary (flagged vs. none), n=584, pooled leave-one-out predictions:**

| | majority baseline | logistic regression | 1-NN | 5-NN (distance-weighted) |
|---|---|---|---|---|
| accuracy | 0.978 | 0.928 | 0.967 | 0.978 |
| macro-F1 | 0.494 | **0.619** | 0.579 | 0.494 |
| `flagged` precision | 0.00 | 0.18 | 0.20 | 0.00 |
| `flagged` recall | 0.00 | **0.62** | 0.15 | 0.00 |

**4-class (high/medium/low/none), same procedure:**

| | majority baseline | logistic regression | 1-NN | 5-NN (distance-weighted) |
|---|---|---|---|---|
| accuracy | 0.978 | 0.925 | 0.967 | 0.978 |
| macro-F1 | 0.247 | 0.313 | **0.337** | 0.247 |

**This is a real positive result, honestly obtained**: logistic
regression on embeddings finds **8 of the 13 flagged clauses** (62%
recall) that the TF-IDF version and the baseline both missed entirely,
using the identical labels and identical documents — only the
representation and evaluation procedure changed. 1-NN wins on the finer
4-class task. 5-NN (distance-weighted) shows **no improvement over
baseline** — with only 1-7 examples per class, weighting by 5 neighbors
just dilutes the signal back toward the majority vote, a real negative
data point inside this experiment's own comparison, not swept aside.

## What this result does and does not support

- **Does support**: the representation (embeddings vs. TF-IDF) and the
  evaluation procedure (LOO-CV vs. a single split) were the two things
  actually holding experiment 1 back, not "this task is unlearnable from
  this data." Fixing both, with nothing else changed, flipped 0% recall
  to 62%.
- **Does not support**: that this is a deployable classifier. 0.18
  precision means roughly 5 of every 6 clauses it flags as risky are
  false positives — usable as a *recall-oriented pre-filter* (surface
  candidates for a human or the existing rule engine to check), not as a
  standalone detector. LOO-CV on 13 positive examples also means each
  individual prediction carries real uncertainty — this is a genuine
  signal, not a tight estimate of real-world performance.
- **Does not support**: that embeddings are "correct" and TF-IDF is
  "wrong" in general — only that embeddings generalize better than
  bag-of-words specifically when positive examples are this scarce,
  which is exactly this dataset's situation.

## Experiment 3: threshold tuning (negative)

Experiment 2's logistic regression uses the default 0.5 probability
cutoff to turn a predicted probability into a "flagged"/"none" call.
That's an arbitrary choice, not a tuned one — this experiment sweeps it,
reusing experiment 2's exact pooled leave-one-out probabilities (no new
model, no new data), to see whether a different cutoff trades a little
recall for meaningfully better precision.

| threshold | precision | recall | F1 | # predicted flagged |
|---|---|---|---|---|
| 0.10 | 0.04 | 1.00 | 0.08 | 322 |
| 0.20 | 0.07 | 1.00 | 0.13 | 180 |
| 0.30 | 0.10 | 0.85 | 0.18 | 110 |
| 0.40 | 0.13 | 0.69 | 0.22 | 69 |
| **0.50** | **0.18** | **0.62** | **0.28** | 45 |
| 0.60 | 0.17 | 0.38 | 0.24 | 29 |
| 0.70 | 0.17 | 0.15 | 0.16 | 12 |
| 0.80 | 0.00 | 0.00 | 0.00 | 6 |
| 0.90 | 0.00 | 0.00 | 0.00 | 0 |

**The default 0.5 threshold is already close to the best F1 this dataset
supports.** Raising it doesn't do what you'd hope: precision barely
moves (0.18 → 0.17 at 0.6–0.7) while recall collapses (0.62 → 0.15), and
past 0.8 even the true positives no longer clear the bar, so both
precision and recall hit zero. There's no hidden better operating point
being left on the table by using the default — the probabilities
themselves aren't well-enough separated for a threshold shift alone to
help. This is consistent with (not a contradiction of) experiment 2's
own limitations section: the fix has to be more/better labeled data, not
a knob on the existing model.

## Concrete next steps

1. **More labeled data is still the real lever** — now checked and
   confirmed the only one that plausibly moves precision. More positive
   examples, or a human-reviewed pass over `ml/build_dataset.py`'s
   silver labels, would let the model discriminate flagged-but-different
   clauses from genuinely risky ones; threshold tuning alone (experiment
   3) can't substitute for that on this data.
2. ~~**A tuned decision threshold**~~ — done, see experiment 3 above:
   doesn't help. Worth having actually checked rather than assumed.
3. **Human-labeled data instead of silver labels** would let this
   experiment test something it structurally cannot yet: whether the
   classifier can find risky clauses the *rules themselves* miss, rather
   than only whether it can imitate the rules it was trained to imitate.

## What none of the three experiments show

- Anything about the deterministic rule engine's own accuracy — its
  outputs are the labels for all three experiments, not a ground truth
  being independently validated.
- That any result generalizes to a larger or differently-composed
  dataset — all three are measured on the same 10 sample documents.
