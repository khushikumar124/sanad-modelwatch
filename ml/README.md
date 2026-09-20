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

## Running it

```bash
# from the repository root, with the main .venv activated
pip install -r ml/requirements.txt

python -m ml.build_dataset          # extracts + chunks + rule-flags every
                                     # sample PDF, writes ml/data/clause_risk_dataset_v1.jsonl
python -m ml.train_risk_classifier  # trains, evaluates against a held-out
                                     # split and a majority-class baseline,
                                     # writes ml/artifacts/{risk_classifier.joblib,results.json}
```

Tests (fast — a synthetic toy dataset, not the real PDFs):

```bash
pytest ml/ -v
```

See [`docs/ml_experiment.md`](../docs/ml_experiment.md) for the actual
measured results, dataset composition, and honest limitations of this
specific run.
