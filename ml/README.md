# Risk-severity classifier (offline ML experiment)

A genuine supervised learning experiment, kept deliberately separate
from both running services (Sanad, ModelWatch): it is not wired into
either app's API, so neither one's Docker image or dependency list
carries scikit-learn's weight for a model nothing serves in production.

## What this is, precisely

**Question:** can a lightweight learned model reproduce
[`sanad/features/risk_flagger.py`](../sanad/features/risk_flagger.py)'s
severity calls (`high` / `medium` / `low` / `none`) from a clause's text
alone, on clauses it never saw during training?

**What this is NOT:** a claim that this classifier is more accurate than
the rule engine, or a second, independent risk detector. Its labels come
from the rule engine itself (a "silver label" dataset, not human
annotation) — see [`build_dataset.py`](build_dataset.py)'s docstring —
so by construction it cannot be validated as "more correct" than the
rules it was trained to imitate. What it *can* show is whether the
signal the rules react to (specific phrasing) is learnable well enough
from a bag-of-words representation to generalize to held-out clauses,
which is a real, useful question if the eventual goal is a classifier
that catches phrasing the fixed regex patterns don't anticipate.

## Three experiments

1. **`train_risk_classifier.py`** — TF-IDF + logistic regression, a
   single train/test split. **Negative result**: statistically
   indistinguishable from a majority-class baseline.
2. **`train_embeddings_loocv.py`** — sentence embeddings (the same
   `all-MiniLM-L6-v2` Sanad's retrieval already uses) + leave-one-out
   cross-validation, three model families compared. **Positive result**:
   logistic regression catches 62% of flagged clauses the first
   experiment missed entirely.
3. **`threshold_tuning.py`** — sweeps experiment 2's decision cutoff to
   see if precision (0.18) can be improved for free. **Negative result**:
   0.5 is already close to the best F1 available; the fix has to be more
   data, not a different cutoff.

All three are kept, not just the flattering one — *why* the first failed
and what specifically fixed it (representation, not hyperparameters) is
the real finding, and experiment 3 shows a plausible-sounding quick fix
that turned out not to work rather than quietly dropping it. See
[`docs/ml_experiment.md`](../docs/ml_experiment.md) for the full measured
numbers and honest limitations of each (in particular: experiment 2's
precision is low — it's a recall-oriented pre-filter, not a deployable
standalone detector).

## Running it

```bash
# from the repository root, with the main .venv activated
pip install -r ml/requirements.txt

python -m ml.build_dataset             # extracts + chunks + rule-flags every
                                        # sample PDF, writes ml/data/clause_risk_dataset_v1.jsonl
python -m ml.train_risk_classifier     # experiment 1 -> ml/artifacts/{risk_classifier.joblib,results.json}
python -m ml.train_embeddings_loocv    # experiment 2 -> ml/artifacts/results_embeddings_loocv.json
python -m ml.threshold_tuning          # experiment 3 -> ml/artifacts/results_threshold_sweep.json
```

Tests (fast — synthetic toy data, not the real PDFs or the real
sentence-transformer model):

```bash
pytest ml/ -v
```
