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
   — **positive result**. Logistic regression on embeddings catches 58%
   of flagged clauses (vs. 0% for the baseline) on the binary task.
3. **Threshold tuning on experiment 2's own probabilities** ([`ml/threshold_tuning.py`](../ml/threshold_tuning.py))
   — asks whether precision (0.20 in experiment 2) can be improved for
   free, just by moving the decision cutoff. **Small real gain**, and one
   that grew as the dataset grew: 0.7 now beats the default 0.5 (F1 0.40
   vs 0.29) — on an earlier, smaller version of this dataset the same
   sweep found nothing, reported honestly both ways below.
4. **LLM-paraphrase augmentation + leave-one-group-out CV** ([`ml/augment_dataset.py`](../ml/augment_dataset.py), [`ml/train_embeddings_grouped_cv.py`](../ml/train_embeddings_grouped_cv.py))
   — experiment 3 confirmed the bottleneck was too few positive examples,
   not a bad cutoff, so this experiment adds more of them two honest ways
   (paraphrasing the real ones, and adding more real documents) and
   evaluates with a CV scheme that can't leak. **Net-positive but mixed
   result**: precision roughly doubles vs. no augmentation (0.20 → 0.50),
   but adding real documents from increasingly different categories did
   not monotonically improve it further (0.56 at 18 documents → 0.50 at
   32) — reported plainly rather than kept at the better-looking number.

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

**Dataset** (shared by all four experiments): every PDF under
`sanad/sample_docs/` — 32 real Indian legal documents (rental,
employment, freelance/service, NDA, MOU, trademark assignment,
franchise, software license, power of attorney, business takeover,
website terms/privacy, and corporate resolutions), extracted, chunked
by the same clause-aware chunker the live app uses, and labeled by
running the real rule engine over each chunk. The original 10 grew to
18, then to 32, in two rounds of adding real documents from an open,
no-signup directory of Indian legal templates, specifically to grow the
positive class beyond its original 13 examples — see experiment 4 below
for why that mattered, and for an honest result that didn't go the way
more data "should."

| | count |
|---|---|
| Total clauses | 760 |
| `none` | 741 |
| `high` | 12 |
| `medium` | 5 |
| `low` | 2 |
| Binary `flagged` (high+medium+low) | 19 |

Only **19 of 760 clauses are positive examples** — the rule engine fires
rarely by design (it's deliberately conservative; see the root README's
limitations section). That scarcity is the real story behind the
experiments below.

## Experiment 1: TF-IDF + single split (negative)

**Model**: TF-IDF (word 1-2 grams) + logistic regression,
`class_weight="balanced"`, `random_state=0`, evaluated on a 75/25 split
(stratified where possible) — 190 test clauses, only 5 of them positive.

| 4-class | learned model | majority-class baseline |
|---|---|---|
| accuracy | 0.968 | 0.968 |
| macro-F1 | 0.246 | 0.246 |

| Binary (flagged/none) | learned model | majority-class baseline |
|---|---|---|
| accuracy | 0.963 | 0.974 |
| macro-F1 | 0.491 | 0.493 |
| `flagged` recall | 0.00 | 0.00 |

Identical to the baseline in substance (both predict `none` for every
test example) — the tiny accuracy/macro-F1 gap is the balanced model
occasionally guessing `flagged` on a true negative, not a real signal.

**Why**: TF-IDF needs near-exact vocabulary overlap between training and
test clauses to generalize. With only a handful of positive training
examples spread across different risk categories and phrasings, most
categories are represented by 1-2 examples each — not enough for a
bag-of-words model to learn anything transferable.

## Experiment 2: sentence embeddings + leave-one-out CV (positive)

Two changes from experiment 1, not a retry with different random seeds:

1. **TF-IDF → sentence embeddings** (`all-MiniLM-L6-v2`, the same model
   Sanad's own retrieval already uses — see [`ml/embed.py`](../ml/embed.py)).
   Lets a match fire on semantic similarity instead of shared vocabulary.
2. **A single split → leave-one-out cross-validation.** With only a
   handful of positive examples total, a single 75/25 split tests on
   just a few of them — one of the least statistically meaningful ways
   to evaluate this dataset. LOO-CV evaluates every one of the 760
   clauses exactly once, trained on all 759 others each time, so no
   single lucky/unlucky split determines the result.

Three genuinely different model families were tried, not the same model
tuned three ways: a parametric classifier (logistic regression) and two
similarity-based few-shot approaches (1-nearest-neighbor, and 5-nearest
distance-weighted), since few-shot/similarity search is generally the
more appropriate family when there are only a handful of positive
examples.

**Binary (flagged vs. none), n=760, pooled leave-one-out predictions:**

| | majority baseline | logistic regression | 1-NN | 5-NN (distance-weighted) |
|---|---|---|---|---|
| accuracy | 0.975 | 0.930 | 0.957 | 0.976 |
| macro-F1 | 0.494 | **0.628** | 0.543 | 0.544 |
| `flagged` precision | 0.00 | 0.20 | 0.11 | 1.00 |
| `flagged` recall | 0.00 | **0.58** | 0.11 | 0.05 |

**4-class (high/medium/low/none), same procedure:** logistic regression
remains ahead of the majority baseline (macro-F1 0.247) but the exact
numbers shift clause-by-clause as the dataset grows — see
`ml/artifacts/results_embeddings_loocv.json` for the current run's full
breakdown rather than a table that would need updating every time the
sample set changes.

**This is a real positive result, honestly obtained**: logistic
regression on embeddings recovers 58% recall on flagged clauses that
the TF-IDF version and the baseline both missed entirely, using the
identical labels and identical documents — only the representation and
evaluation procedure changed. 5-NN (distance-weighted) trades recall
for near-perfect precision (1.00) at 0.05 recall — a real, different
trade-off point, not simply "worse." Worth noting plainly: recall
dropped slightly from an earlier, smaller version of this dataset (62%
at 16 positives → 58% at 19) — adding more, more heterogeneous document
categories didn't purely help, it also made the positive class harder
to discriminate as a whole. Reported here rather than only keeping the
better-looking number from an earlier run.

## What this result does and does not support

- **Does support**: the representation (embeddings vs. TF-IDF) and the
  evaluation procedure (LOO-CV vs. a single split) were the two things
  actually holding experiment 1 back, not "this task is unlearnable from
  this data." Fixing both, with nothing else changed, flipped 0% recall
  to well over half.
- **Does not support**: that this is a deployable classifier. 0.20
  precision means roughly 4 of every 5 clauses it flags as risky are
  false positives — usable as a *recall-oriented pre-filter* (surface
  candidates for a human or the existing rule engine to check), not as a
  standalone detector. LOO-CV on 19 positive examples also means each
  individual prediction carries real uncertainty — this is a genuine
  signal, not a tight estimate of real-world performance.
- **Does not support**: that embeddings are "correct" and TF-IDF is
  "wrong" in general — only that embeddings generalize better than
  bag-of-words specifically when positive examples are this scarce,
  which is exactly this dataset's situation.
- **Does not support**: that adding more real documents monotonically
  improves this classifier. It doesn't, straightforwardly — see the
  recall dip noted above and experiment 4's update below.

## Experiment 3: threshold tuning (small real gain, revised from an earlier negative result)

Experiment 2's logistic regression uses the default 0.5 probability
cutoff to turn a predicted probability into a "flagged"/"none" call.
That's an arbitrary choice, not a tuned one — this experiment sweeps it,
reusing experiment 2's exact pooled leave-one-out probabilities (no new
model, no new data), to see whether a different cutoff trades a little
recall for meaningfully better precision.

| threshold | precision | recall | F1 | # predicted flagged |
|---|---|---|---|---|
| 0.10 | 0.04 | 0.95 | 0.07 | 466 |
| 0.20 | 0.06 | 0.89 | 0.11 | 297 |
| 0.30 | 0.10 | 0.89 | 0.18 | 165 |
| 0.40 | 0.14 | 0.68 | 0.23 | 95 |
| 0.50 (default) | 0.20 | 0.58 | 0.29 | 56 |
| 0.60 | 0.30 | 0.53 | 0.38 | 33 |
| **0.70** | **0.44** | 0.37 | **0.40** | 16 |
| 0.80 | 0.43 | 0.16 | 0.23 | 7 |
| 0.90 | 0.00 | 0.00 | 0.00 | 0 |

**This result changed again as the dataset grew, and that keeps being
worth reporting honestly rather than freezing on whichever run looked
best.** On the original 13-positive dataset, the default 0.5 threshold
was already close to optimal. At 16 positives, 0.6 became better (F1
0.34). At 19 positives, 0.7 is better still (F1 0.40, precision 0.44) —
the ranking of "best threshold" keeps moving as more data arrives,
which is itself the finding: there is no single stable optimal cutoff
for this dataset yet, only a trend that more data gives the probability
estimates more separation to exploit. The improvement is still modest
(0.44 precision still means more than half of flagged clauses are false
flagged clauses are false positives) and this cutoff was chosen by
looking at these same 16 positives, so it's a plausible operating point
to try, not a cutoff validated on held-out data. The core conclusion
from the original run stands regardless: a threshold knob is a much
smaller lever than more labeled data (experiment 4), not a substitute
for it.

## Experiment 4: paraphrase augmentation + leave-one-group-out CV (positive)

Experiment 3 confirmed the actual bottleneck was too few positive
examples for the model to discriminate genuinely risky clauses from
flagged-but-different ones — not a bad decision threshold. This
experiment adds more positive examples the only way that doesn't
involve either downloading contracts of unknown provenance/licensing or
inventing risk categories that never appeared in a real document: an
LLM (the same local Ollama model Sanad's own features already depend
on) rewrites each real flagged clause in different wording, 4 times
each, preserving the exact legal fact (same duration, same restriction,
same party) — every synthetic row marked `"is_synthetic": true` in
[`ml/data/clause_risk_dataset_v1_augmented.jsonl`](../ml/data/clause_risk_dataset_v1_augmented.jsonl).

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
on, together, never split across the train/test boundary. The "none"
clauses (never augmented) are each their own singleton group, so for
them this is identical to experiment 2's plain leave-one-out — only the
positive groups actually changed shape.

**Update: real documents were added in two rounds, not just paraphrases
— and the second round is an honest mixed result, not a repeat win.**
Round 1 (13 → 16 real positives, 10 → 18 documents, 5 new categories:
NDA, MOU, trademark assignment, franchise, software license) took
precision from 0.18 to 0.56. Round 2 (16 → 19 real positives, 18 → 32
documents, adding power of attorney, business takeover, website
terms/privacy, corporate resolutions, and more employment variants)
grew the dataset further but precision came back down to 0.50 — more
data did not mean a strictly better number this time. The table below
is the current, 32-document, fully real+leakage-safe result:

| | none (precision/recall/F1) | flagged (precision/recall/F1) | macro-F1 |
|---|---|---|---|
| majority-class baseline | 0.89 / 1.00 / 0.94 | 0.00 / 0.00 / 0.00 | 0.470 |
| experiment 2 (no augmentation, LOO-CV, 19 real positives) | — | 0.20 / 0.58 / 0.29 | — |
| **experiment 4, logistic regression (32 docs, augmented, grouped CV)** | 0.95 / 0.92 / 0.94 | **0.50 / 0.60 / 0.55** | **0.742** |
| experiment 4, 1-NN (32 docs, augmented, grouped CV) | 0.90 / 0.97 / 0.94 | 0.44 / 0.17 / 0.24 | 0.590 |

836 total rows (760 real + 76 synthetic), 760 groups, 95/836 positive
after grouping (up from 19/760 real positives, since each real positive
now has up to 4 synthetic siblings inflating its group's row count, not
its group count). See
[`ml/artifacts/results_grouped_cv.json`](../ml/artifacts/results_grouped_cv.json)
for the exact current numbers whenever this is re-run.

**What this does and doesn't show:**
- **Does show**: more labeled positive examples — both more real
  documents and more synthetic paraphrases of existing ones, evaluated
  so neither can leak — measurably improve this model's ability to
  separate flagged from unflagged clauses **compared to no augmentation
  at all** (0.20 → 0.50 precision, same 32-document dataset). That part
  of the story replicated across both rounds.
- **Does not show**: that adding more real documents is a one-way
  ratchet on precision. Round 2's raw number (0.50) is lower than round
  1's (0.56) on a smaller dataset — plausible reasons include the new
  categories (power of attorney, corporate resolutions) being
  structurally different from the rental/employment/freelance clauses
  the rule engine's regex patterns were originally tuned against, so
  their "flagged" examples may be harder for embeddings to separate
  from surrounding "none" boilerplate. This wasn't diagnosed further —
  reported as an open, honest question rather than quietly kept at the
  better-looking round-1 number.
- **Does not show**: performance on genuinely novel risky clauses beyond
  what these 32 documents and their paraphrases contain. The dataset is
  now considerably bigger and more diverse than the original
  10-document/13-positive version, but it is still small in absolute
  terms (19 real positive clauses).
- 1-NN's recall stayed weak with augmentation (0.17) despite the extra
  data — a reminder that "more data helps" isn't universal across model
  families; the effect here is specific to logistic regression's use of
  the whole embedding space rather than only the nearest neighbor.

## Concrete next steps

1. ~~**More labeled data is still the real lever**~~ — partially: it's
   the biggest lever for turning 0% recall into real detection
   (experiment 2) and augmentation vs. no augmentation on a fixed
   dataset (experiment 4), but adding real documents from increasingly
   different categories does not straightforwardly keep improving
   precision (0.56 at 18 docs → 0.50 at 32 docs). More data helps up to
   a point, and diversity beyond the original clause "genres" needs its
   own investigation, not an assumption.
2. ~~**A tuned decision threshold**~~ — done, see experiment 3 above: a
   real gain that keeps growing as data grows (0.29 → 0.34 → 0.40 F1
   across three dataset sizes), still a smaller lever than more data
   and not yet stable enough to call a fixed "best" threshold.
3. ~~**Human-labeled data from more documents**~~ — done in two rounds
   (13 → 16 → 19 real positives across 10 → 18 → 32 documents). The
   open question this still doesn't answer: whether the classifier can
   find risky clauses the *rules themselves* miss — that needs
   human-labeled data, not rule-engine silver labels, since right now
   both the labels and the thing being learned come from the same rules
   (see [`ml/README.md`](../ml/README.md)'s "What this is NOT" section).
4. **Why round 2 underperformed round 1 on precision** is now the
   concrete open question this experiment surfaced but didn't answer —
   worth investigating (e.g. per-category breakdown of which new
   documents' positives the model gets wrong) before adding a third
   round of documents on faith that more will keep helping.

## What none of the four experiments show

- Anything about the deterministic rule engine's own accuracy — its
  outputs are the labels for all four experiments, not a ground truth
  being independently validated.
- That any result generalizes beyond this project's own sample set —
  all four experiments are measured on the same 32 documents (19 real
  positive clauses, with experiment 4 also adding phrasing variety
  around those clauses via paraphrasing). Considerably larger than the
  original 10-document/13-positive version, but still a small,
  project-curated set, not an independent benchmark — and, per
  experiment 4's round 2, "larger" isn't the same as "more separable."
