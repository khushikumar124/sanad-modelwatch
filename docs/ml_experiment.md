# Risk-severity classifier: a real (negative) result

See [`ml/README.md`](../ml/README.md) for what this experiment is and,
just as importantly, what it isn't claiming. In short: does a lightweight
learned model reproduce `sanad/features/risk_flagger.py`'s severity
labels from clause text alone, on clauses it never saw during training?

**Headline result: no, not with the data currently in this repo.** The
learned model is statistically indistinguishable from a majority-class
baseline that always predicts "none" — on both the original 4-class task
and a coarser binary "flagged vs. none" reformulation. This is reported
as a negative result because it is one, not reframed into something more
flattering. Root README's own framing applies here too: "including the
results that came out worse than hoped."

## Setup

```bash
python -m ml.build_dataset          # -> ml/data/clause_risk_dataset_v1.jsonl
python -m ml.train_risk_classifier  # -> ml/artifacts/{risk_classifier.joblib,results.json}
```

**Dataset**: every PDF under `sanad/sample_docs/` (10 real Indian rental,
freelance, employment, and legal-services documents), extracted, chunked
by the same clause-aware chunker the live app uses, and labeled by
running the real rule engine over each chunk.

| | count |
|---|---|
| Total clauses | 584 |
| `none` | 571 |
| `high` | 7 |
| `medium` | 4 |
| `low` | 2 |
| Binary `flagged` (high+medium+low) | 13 |

584 clauses sounds like a reasonable dataset size, but only **13 of them
are positive examples** for the thing being learned — the rule engine
fires rarely by design (it's deliberately conservative; see the root
README's limitations section). Split 75/25 (stratified where possible),
that leaves roughly **10 positive training examples and 3 positive test
examples**. That number is the real story of this experiment.

**Model**: TF-IDF (word 1-2 grams) + logistic regression,
`class_weight="balanced"`, `random_state=0`. Compared against a
`DummyClassifier(strategy="most_frequent")` baseline on the identical
train/test split, so the comparison is apples-to-apples.

## Measured results

**4-class** (`high` / `medium` / `low` / `none`), n_test=146:

| | learned model | majority-class baseline |
|---|---|---|
| accuracy | 0.979 | 0.979 |
| macro-F1 | 0.247 | 0.247 |

Identical to four decimal places because the learned model predicted
`none` for every single test example — the class weighting didn't
overcome having 1-2 training examples for `low`/`medium` and roughly 5
for `high`.

**Binary** (`flagged` vs. `none`), same split, n_test=146, 3 positive:

| | learned model | majority-class baseline |
|---|---|---|
| accuracy | 0.979 | 0.979 |
| macro-F1 | 0.495 | 0.495 |
| `flagged` recall | 0.00 | 0.00 |

Same outcome even after collapsing to the coarser, more learnable
question. This was checked against **three different configurations**
(unigrams-only logistic regression, higher-regularization logistic
regression, and multinomial Naive Bayes) — all three produced zero
correctly-identified flagged clauses in the test set. This rules out "bad
hyperparameter choice" as the explanation.

## Is there *any* learned signal at all?

Worth checking directly rather than assuming: ranking every test clause
by the model's predicted P(flagged) and looking at where the true
positives land answers whether the model is close-but-under-threshold, or
has genuinely learned nothing.

```
Top 10 by predicted P(flagged): all 10 are true "none" clauses (highest p=0.455)

True-flagged test clauses, by rank out of 146:
  rank 18   p=0.210   "TERMINATION ... Either Party can terminate..."
  rank 38   p=0.166   "MISCELLANEOUS ... Notwithstanding any other..."
  rank 50   p=0.156   "5. INDEMNIFICATION FOR DAMAGES, TAXES..."
```

The true positives don't even cluster near the top — they're scattered
in the middle of the ranking, below ten false clauses the model was more
confident about. That's a genuine "no meaningful signal learned" result,
not a "right idea, wrong threshold" one.

## Why, and what would actually fix it

TF-IDF + logistic regression needs to see a phrase (or a close lexical
variant of one) in multiple positive examples to learn it's predictive.
With ~10 positive training examples spread across three severity levels
and several different risk categories (non-compete, termination-without-
notice, unilateral-amendment, etc. — see `risk_flagger.py`'s rule list),
most categories are represented by 1-2 examples each, which is not enough
for a bag-of-words model to generalize from.

Concrete next steps, in order of expected impact:

1. **More labeled data is the actual bottleneck**, not a better model.
   The fix is more contracts through `ml/build_dataset.py`, not a fancier
   classifier over the same 584 clauses.
2. **Sentence-embedding features instead of TF-IDF** (e.g. the same
   `all-MiniLM-L6-v2` model Sanad's retrieval already uses) would let the
   model generalize by *semantic* similarity to the ~10 known positive
   examples, rather than requiring near-exact vocabulary overlap — this
   is the single change most likely to help at this dataset size, and is
   a natural next experiment since the embedding model is already a
   dependency elsewhere in this repo.
3. **Human-labeled data instead of silver labels** would also let this
   experiment test something the current setup structurally cannot: it
   currently cannot show the classifier is "more correct" than the rules,
   only whether it can imitate them — a human-labeled set would let it
   find clauses the rules miss entirely, which is the more interesting
   long-term goal.

## What this does NOT show

- That a learned classifier for this task is impossible — only that it
  didn't work with the data volume and representation tried here.
- Anything about the deterministic rule engine's own accuracy. The rule
  engine's outputs are the labels, not a ground truth being validated
  independently. Also see: this repo's `high`/`medium`/`low` severities
  are for a handful of manually-authored rules over 10 sample documents,
  not a broad legal corpus.
- That this generalizes to a larger or differently-composed dataset —
  see "what would actually fix it," above.
