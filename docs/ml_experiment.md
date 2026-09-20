# Risk-severity classifier: four experiments, two fixes that actually mattered

See [`ml/README.md`](../ml/README.md) for what this experiment is and,
just as importantly, what it isn't claiming. In short: does a lightweight
learned model reproduce `sanad/features/risk_flagger.py`'s severity
labels from clause text alone, on clauses it never saw during training?

**Four experiments, run in this order, all real:**

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
4. **LLM-paraphrase augmentation + leave-one-group-out CV** ([`ml/augment_dataset.py`](../ml/augment_dataset.py), [`ml/train_embeddings_grouped_cv.py`](../ml/train_embeddings_grouped_cv.py))
   — experiment 3 confirmed the bottleneck was too few positive examples,
   not a bad cutoff, so this experiment adds more of them the only honest
   way available (paraphrasing the real ones, not fabricating new risk
   categories) and evaluates with a CV scheme that can't leak. **Positive
   result**: precision more than doubles, 0.18 → 0.50, with recall
   improving too (0.62 → 0.66).

All four are kept and reported, not just the flattering ones, because
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
python -m ml.augment_dataset             # experiment 4 -> ml/data/clause_risk_dataset_v1_augmented.jsonl (needs a local Ollama)
python -m ml.train_embeddings_grouped_cv # experiment 4 -> ml/artifacts/results_grouped_cv.json
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

## Experiment 4: paraphrase augmentation + leave-one-group-out CV (positive)

Experiment 3 confirmed the actual bottleneck was too few positive
examples for the model to discriminate genuinely risky clauses from
flagged-but-different ones — not a bad decision threshold. This
experiment adds more positive examples the only way that doesn't
involve either downloading contracts of unknown provenance/licensing or
inventing risk categories that never appeared in a real document: an
LLM (the same local Ollama model Sanad's own features already depend
on) rewrites each of the 13 real flagged clauses in different wording,
4 times each, preserving the exact legal fact (same duration, same
restriction, same party) — 52 synthetic rows total, every one marked
`"is_synthetic": true` in [`ml/data/clause_risk_dataset_v1_augmented.jsonl`](../ml/data/clause_risk_dataset_v1_augmented.jsonl).

**The leakage trap this experiment exists to avoid:** a paraphrase of a
clause is the same underlying fact restated, not independent evidence.
Plain leave-one-out (experiment 2's method) would let a real clause's
own paraphrase sit in the training fold while that same clause is held
out for testing — the model could match near-duplicate phrasing and
score well without generalizing to anything new, producing a
fake-looking improvement. Instead, every synthetic row carries a
`group_id` identical to the real clause it was paraphrased from, and
evaluation uses **leave-one-group-out** cross-validation: a real clause
and every one of its own paraphrases are always held out, or trained
on, together, never split across the train/test boundary. The 571
"none" clauses (never augmented) are each their own singleton group, so
for them this is identical to experiment 2's plain leave-one-out — only
the 13 positive groups actually changed shape.

| | none (precision/recall/F1) | flagged (precision/recall/F1) | macro-F1 |
|---|---|---|---|
| majority-class baseline | 0.90 / 1.00 / 0.95 | 0.00 / 0.00 / 0.00 | 0.473 |
| experiment 2 (no augmentation, LOO-CV) | — | 0.18 / 0.62 / 0.28 | — |
| **experiment 4, logistic regression (augmented, grouped CV)** | 0.96 / 0.92 / 0.94 | **0.50 / 0.66 / 0.57** | **0.756** |
| experiment 4, 1-NN (augmented, grouped CV) | 0.92 / 0.98 / 0.95 | 0.62 / 0.25 / 0.35 | 0.651 |

636 total rows (584 real + 52 synthetic), 584 groups, 65/636 positive
after grouping (up from 13/584 real positives, because each real
positive now has 4 synthetic siblings inflating its group's row count,
not its group count).

**What this does and doesn't show:**
- **Does show**: more labeled positive examples — even synthetic ones
  derived from real clauses, evaluated so they can't leak — measurably
  improve this model's ability to separate flagged from unflagged
  clauses. Precision more than doubled (0.18 → 0.50) without trading
  away recall (0.62 → 0.66).
- **Does not show**: performance on genuinely novel risky clauses the
  paraphraser never saw. Every positive example the model has ever been
  evaluated on, real or synthetic, still traces back to just 13 original
  clauses from 10 sample documents — augmentation multiplies phrasing
  diversity around those 13 facts, it does not add new facts. The real
  fix identified back in experiment 3 (more *human-labeled* clauses from
  more documents) remains untried and is still the strongest form of
  more data.
- 1-NN got worse on recall with augmentation (0.62 → 0.25) despite
  the extra data — a reminder that "more data helps" isn't universal
  across model families; the effect here is specific to logistic
  regression's use of the whole embedding space rather than only the
  single nearest neighbor.

## Concrete next steps

1. ~~**More labeled data is still the real lever**~~ — done, see
   experiment 4 above: paraphrase augmentation of the existing 13
   positives is a real, measured improvement (0.18 → 0.50 precision).
2. ~~**A tuned decision threshold**~~ — done, see experiment 3 above:
   doesn't help. Worth having actually checked rather than assumed.
3. **Human-labeled data from more documents** (rather than more
   paraphrases of the same 13 clauses) is the next lever, and the one
   experiment 4 explicitly can't substitute for: it would let this
   experiment test something it structurally cannot yet — whether the
   classifier can find risky clauses the *rules themselves* miss, from
   genuinely new facts, rather than only recognizing rephrasings of
   facts it already trained on.

## What none of the four experiments show

- Anything about the deterministic rule engine's own accuracy — its
  outputs are the labels for all four experiments, not a ground truth
  being independently validated.
- That any result generalizes to a larger or differently-composed
  dataset — all four are measured on the same 10 sample documents (13
  original positive clauses, with experiment 4 adding phrasing variety
  around those same 13, not new ones).
