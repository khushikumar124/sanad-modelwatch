# Classifier example

Monitors a tabular classifier with ModelWatch's `ClassifierAdapter`,
using only `modelwatch-client`'s public SDK -- the same "connect any
model" story as [`examples/independent_rag/`](../independent_rag/), for
a completely different model type (tabular features, not a RAG
pipeline).

## What's synthetic and what's real

**Synthetic**: there is no real tabular classifier anywhere in this
project. `age` and `income` are drawn from hand-picked Gaussian
distributions, and predictions/labels are generated to hit a target
accuracy. None of this is presented as a real model's real data.

**Real**: everything ModelWatch does with that data. `client.check()`
runs an actual Kolmogorov-Smirnov two-sample test per feature (scipy,
not a canned verdict), with a real Bonferroni correction across
features, and computes a real accuracy number from the predictions/
labels. The clean batch and the shifted batch produce genuinely
different KS statistics because the underlying samples are genuinely
different -- nothing here is staged to look dramatic.

## Running it

```bash
./run.sh                              # starts ModelWatch on :8000
pip install -e ./modelwatch-client
python examples/classifier/run_example.py
```

## What you should see

A clean batch (same distribution as baseline) does not flag drift. A
batch drawn from a shifted distribution (older, higher-income
applicants) does -- both `age` and `income` come back with a KS
p-value near zero, correctly rejecting the "same distribution" null.
