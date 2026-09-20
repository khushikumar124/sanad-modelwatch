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

## Four experiments

1. **`train_risk_classifier.py`** — TF-IDF + logistic regression, a
   single train/test split. **Negative result**: statistically
   indistinguishable from a majority-class baseline.
2. **`train_embeddings_loocv.py`** — sentence embeddings (the same
   `all-MiniLM-L6-v2` Sanad's retrieval already uses) + leave-one-out
   cross-validation, three model families compared. **Positive result**:
   logistic regression catches most of the flagged clauses the first
   experiment missed entirely (58% recall on the current 32-document
   dataset; this exact number has moved between 58-62% as the dataset
   grew — see `docs/ml_experiment.md` for why).
3. **`threshold_tuning.py`** — sweeps experiment 2's decision cutoff to
   see if precision can be improved for free. **Growing real gain**: on
   the original 13-positive dataset this found nothing; it now shows a
   real, growing improvement as the dataset has grown (F1 0.29 → 0.34 →
   0.40 across three dataset sizes) — a result that changed as more data
   arrived, reported honestly at each size rather than only once.
4. **`augment_dataset.py` + `train_embeddings_grouped_cv.py`** —
   LLM-paraphrases the real flagged clauses (4 variants each, marked
   `is_synthetic`), evaluated with leave-one-**group**-out CV so a real
   clause and its own paraphrases can never land on opposite sides of
   the train/test split. Also where more real documents were added in
   two rounds (10 → 18 → 32 documents, 13 → 16 → 19 real positives).
   **Mixed but net-positive result**: augmentation vs. none on the same
   dataset consistently roughly doubles precision (0.20 → 0.50 on the
   current 32-doc dataset); adding more real documents from increasingly
   different categories, however, did *not* monotonically improve
   precision (0.56 at 18 docs → 0.50 at 32 docs) — reported as an open
   question, not smoothed over.

All four are kept, not just the flattering ones — *why* the first failed
and what specifically fixed it (representation, not hyperparameters) is
the real finding, and experiment 3 shows a plausible-sounding quick fix
that only started working once there was enough data, rather than
quietly dropping the earlier negative result. See
[`docs/ml_experiment.md`](../docs/ml_experiment.md) for the full measured
numbers and honest limitations of each (in particular: experiment 2's
precision is low — it's a recall-oriented pre-filter, not a deployable
standalone detector; experiment 4 improves that same number substantially
via augmentation, but growing the raw document count further didn't keep
improving it — the dataset, while much bigger than the original 10
documents, is still small — 19 real positive clauses).

## Running it

```bash
# from the repository root, with the main .venv activated
pip install -r ml/requirements.txt

python -m ml.build_dataset               # extracts + chunks + rule-flags every
                                          # sample PDF, writes ml/data/clause_risk_dataset_v1.jsonl
python -m ml.train_risk_classifier       # experiment 1 -> ml/artifacts/{risk_classifier.joblib,results.json}
python -m ml.train_embeddings_loocv      # experiment 2 -> ml/artifacts/results_embeddings_loocv.json
python -m ml.threshold_tuning            # experiment 3 -> ml/artifacts/results_threshold_sweep.json
python -m ml.augment_dataset             # experiment 4 -> ml/data/clause_risk_dataset_v1_augmented.jsonl (needs local Ollama)
python -m ml.train_embeddings_grouped_cv # experiment 4 -> ml/artifacts/results_grouped_cv.json
```

Tests (fast — synthetic toy data, not the real PDFs or the real
sentence-transformer model):

```bash
pytest ml/ -v
```
