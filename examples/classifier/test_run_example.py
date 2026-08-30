"""Tests for the classifier example's synthetic-data helpers (no
ModelWatch server needed -- these check the helpers produce real,
correctly-shaped statistical inputs; the server-dependent flow is
verified live by actually running run_example.py, not here).

Loads run_example.py by explicit file path rather than a bare
`import run_example` -- examples/independent_rag/ has its own
run_example.py, and pytest's default (non-package) import mode keys
modules in sys.modules by bare basename, so two same-named modules
under different, __init__.py-less directories can silently collide
(whichever one pytest happens to import first "wins", and the other
test file quietly gets the wrong module). Explicit loading sidesteps
that instead of relying on import-order luck."""
import importlib.util
import random
from pathlib import Path

_spec = importlib.util.spec_from_file_location("classifier_run_example", Path(__file__).parent / "run_example.py")
_run_example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run_example)
_gaussian_feature = _run_example._gaussian_feature
_synthetic_predictions_and_labels = _run_example._synthetic_predictions_and_labels


def test_gaussian_feature_has_the_right_length():
    values = _gaussian_feature(random.Random(1), mean=10, stdev=2, n=50)
    assert len(values) == 50


def test_gaussian_feature_is_roughly_centered_on_the_given_mean():
    values = _gaussian_feature(random.Random(1), mean=100, stdev=5, n=2000)
    mean = sum(values) / len(values)
    assert 95 < mean < 105  # generous bound -- this is a statistical check, not exact equality


def test_synthetic_predictions_and_labels_are_binary():
    predictions, labels = _synthetic_predictions_and_labels(random.Random(1), n=100, accuracy=0.9)
    assert all(p in (0, 1) for p in predictions)
    assert all(l in (0, 1) for l in labels)
    assert len(predictions) == len(labels) == 100


def test_synthetic_accuracy_is_approximately_the_requested_rate():
    predictions, labels = _synthetic_predictions_and_labels(random.Random(1), n=5000, accuracy=0.9)
    accuracy = sum(1 for p, l in zip(predictions, labels) if p == l) / len(labels)
    assert 0.87 < accuracy < 0.93  # generous bound around the requested 0.9


def test_different_seeds_produce_different_samples():
    a = _gaussian_feature(random.Random(1), mean=0, stdev=1, n=20)
    b = _gaussian_feature(random.Random(2), mean=0, stdev=1, n=20)
    assert a != b
