"""ClassifierAdapter tests against controlled ground truth: data is
generated with a known distribution, so whether drift *should* be detected
is known ahead of time rather than inferred from the test's own output.
"""
import numpy as np

from modelwatch.adapters.classifier_adapter import ClassifierAdapter


def _baseline_data(seed: int = 0, n: int = 500) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "features": {
            "age": rng.normal(loc=35, scale=8, size=n).tolist(),
            "income": rng.normal(loc=50000, scale=12000, size=n).tolist(),
        }
    }


def test_clean_batch_is_not_flagged():
    adapter = ClassifierAdapter()
    baseline = adapter.build_baseline(_baseline_data(seed=0))

    rng = np.random.default_rng(1)
    clean_batch = {
        "features": {
            "age": rng.normal(loc=35, scale=8, size=200).tolist(),
            "income": rng.normal(loc=50000, scale=12000, size=200).tolist(),
        }
    }

    result = adapter.check_drift(baseline, clean_batch)

    assert result.is_drifted is False
    assert result.drift_score == 0.0
    assert all(not s.is_drifted for s in result.signals)


def test_drifted_batch_is_flagged():
    adapter = ClassifierAdapter()
    baseline = adapter.build_baseline(_baseline_data(seed=0))

    rng = np.random.default_rng(2)
    drifted_batch = {
        "features": {
            # shifted ~5 standard deviations -- unmistakably a different distribution
            "age": rng.normal(loc=75, scale=8, size=200).tolist(),
            "income": rng.normal(loc=120000, scale=12000, size=200).tolist(),
        }
    }

    result = adapter.check_drift(baseline, drifted_batch)

    assert result.is_drifted is True
    assert result.drift_score == 1.0
    assert all(s.is_drifted for s in result.signals)
    ages_signal = next(s for s in result.signals if s.name == "age")
    assert ages_signal.detail["pvalue"] < 0.05


def test_quality_reflects_real_accuracy_when_labels_given():
    adapter = ClassifierAdapter()
    baseline = adapter.build_baseline(_baseline_data(seed=0))

    rng = np.random.default_rng(3)
    n = 100
    labels = [0, 1] * (n // 2)
    # Construct predictions with an exactly known accuracy: first 80 correct, last 20 wrong.
    predictions = labels[:80] + [1 - l for l in labels[80:]]

    batch = {
        "features": {
            "age": rng.normal(loc=35, scale=8, size=n).tolist(),
            "income": rng.normal(loc=50000, scale=12000, size=n).tolist(),
        },
        "predictions": predictions,
        "labels": labels,
    }

    result = adapter.check_drift(baseline, batch)

    assert result.quality_score == 0.8


def test_quality_is_none_without_labels():
    adapter = ClassifierAdapter()
    baseline = adapter.build_baseline(_baseline_data(seed=0))
    rng = np.random.default_rng(4)
    batch = {
        "features": {
            "age": rng.normal(loc=35, scale=8, size=50).tolist(),
            "income": rng.normal(loc=50000, scale=12000, size=50).tolist(),
        }
    }

    result = adapter.check_drift(baseline, batch)

    assert result.quality_score is None
