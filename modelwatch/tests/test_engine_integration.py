"""Engine-level integration tests: register a model through the real
MonitoringEngine + SQLite Storage + a real adapter, and check the full
register -> check -> alert -> retrain -> recover story end to end.
"""
import numpy as np
import pytest

from modelwatch.adapters.classifier_adapter import ClassifierAdapter
from modelwatch.adapters.llm_adapter import LLMAdapter
from modelwatch.core.engine import MonitoringEngine
from modelwatch.core.storage import Storage

GOLDEN_SET = [
    {"prompt": "What is the notice period?", "expected_answer": "The notice period is thirty days."},
    {"prompt": "What is the monthly rent?", "expected_answer": "The monthly rent is twenty five thousand rupees."},
]


@pytest.fixture
def engine(tmp_path):
    storage = Storage(str(tmp_path / "test.db"))
    yield MonitoringEngine(storage)
    storage.close()


def _features(seed, loc_age=35, loc_income=50000, n=200):
    rng = np.random.default_rng(seed)
    return {
        "features": {
            "age": rng.normal(loc=loc_age, scale=8, size=n).tolist(),
            "income": rng.normal(loc=loc_income, scale=12000, size=n).tolist(),
        }
    }


def test_register_model_persists_and_returns_config(engine):
    config = {"embedding_model": "all-MiniLM-L6-v2", "retrieval_top_k": 6, "prompt_version": "v3"}
    model = engine.register_model("clf-cfg", "test classifier", ClassifierAdapter(), _features(seed=0, n=200), config=config)
    assert model["config"] == config

    # persisted, not just returned from the register call
    assert engine.get_model("clf-cfg")["config"] == config
    assert engine.list_models()[-1]["config"] == config


def test_register_model_without_config_defaults_to_empty_dict(engine):
    model = engine.register_model("clf-nocfg", "test classifier", ClassifierAdapter(), _features(seed=0, n=200))
    assert model["config"] == {}


def test_clean_check_produces_no_alert(engine):
    engine.register_model("clf-1", "test classifier", ClassifierAdapter(), _features(seed=0, n=500))

    result = engine.run_check("clf-1", _features(seed=1))

    assert result["is_drifted"] is False
    assert result["alert_id"] is None
    assert engine.get_alerts("clf-1") == []
    assert len(engine.get_history("clf-1")) == 1


def test_drifted_check_produces_exactly_one_alert(engine):
    engine.register_model("clf-2", "test classifier", ClassifierAdapter(), _features(seed=0, n=500))

    result = engine.run_check("clf-2", _features(seed=2, loc_age=75, loc_income=120000))

    assert result["is_drifted"] is True
    assert result["alert_id"] is not None
    alerts = engine.get_alerts("clf-2")
    assert len(alerts) == 1
    assert alerts[0]["resolved"] is False


def test_llm_adapter_drifted_check_produces_exactly_one_alert(engine):
    engine.register_model("llm-1", "test chatbot", LLMAdapter(), GOLDEN_SET)

    drifted_batch = [
        {"prompt": g["prompt"], "actual_answer": "Bananas grow in tropical climates."} for g in GOLDEN_SET
    ]
    result = engine.run_check("llm-1", drifted_batch)

    assert result["is_drifted"] is True
    alerts = engine.get_alerts("llm-1")
    assert len(alerts) == 1


def test_trigger_retrain_resets_baseline_bumps_version_and_resolves_alerts(engine):
    engine.register_model("clf-3", "test classifier", ClassifierAdapter(), _features(seed=0, n=500))
    engine.run_check("clf-3", _features(seed=2, loc_age=75, loc_income=120000))
    assert len(engine.get_alerts("clf-3", active_only=True)) == 1

    retrain_calls = []
    fresh_data = _features(seed=2, loc_age=75, loc_income=120000, n=500)
    engine.trigger_retrain("clf-3", retrain_calls.append, fresh_data)

    assert retrain_calls == [fresh_data]
    model = engine.get_model("clf-3")
    assert model["current_version"] == 2
    assert len(engine.get_versions("clf-3")) == 2
    assert engine.get_alerts("clf-3", active_only=True) == []

    # a check against data matching the new baseline should now read as clean
    result = engine.run_check("clf-3", _features(seed=3, loc_age=75, loc_income=120000))
    assert result["is_drifted"] is False


def test_health_state_is_healthy_by_default_before_any_check(engine):
    engine.register_model("clf-health-0", "test classifier", ClassifierAdapter(), _features(seed=0, n=500))
    health = engine.get_health("clf-health-0")
    assert health["state"] == "healthy"
    assert health["consecutive_drifted"] == 0


def test_default_thresholds_alert_immediately_and_recover_after_confirmation(engine):
    """Default MODELWATCH_DEGRADED_AFTER_CONSECUTIVE=1 reproduces the
    original one-shot alerting -- checked at the engine level, not just
    the pure state machine in test_health.py. Recovery always passes
    through one RECOVERING check (a single clean batch isn't taken as
    proof the incident is over) before the alert auto-resolves."""
    engine.register_model("clf-health-1", "test classifier", ClassifierAdapter(), _features(seed=0, n=500))

    result = engine.run_check("clf-health-1", _features(seed=2, loc_age=75, loc_income=120000))
    assert result["health_state"] == "degraded"
    assert result["alert_id"] is not None

    result = engine.run_check("clf-health-1", _features(seed=3))  # clean batch #1
    assert result["health_state"] == "recovering"
    assert engine.get_alerts("clf-health-1", active_only=True)[0]["resolved"] is False

    result = engine.run_check("clf-health-1", _features(seed=4))  # clean batch #2 confirms recovery
    assert result["health_state"] == "healthy"
    assert engine.get_alerts("clf-health-1", active_only=True) == []


def test_hysteresis_suppresses_a_single_drifted_batch(engine, monkeypatch):
    """With degraded_after_consecutive=2, one bad batch shows as a warning
    (no alert), and only a second consecutive bad batch raises one."""
    import modelwatch.core.engine as engine_module
    from dataclasses import replace

    monkeypatch.setattr(
        engine_module, "config", replace(engine_module.config, health_degraded_after_consecutive=2)
    )

    engine.register_model("clf-health-2", "test classifier", ClassifierAdapter(), _features(seed=0, n=500))

    result = engine.run_check("clf-health-2", _features(seed=2, loc_age=75, loc_income=120000))
    assert result["health_state"] == "warning"
    assert result["alert_id"] is None
    assert engine.get_alerts("clf-health-2", active_only=True) == []

    result = engine.run_check("clf-health-2", _features(seed=4, loc_age=75, loc_income=120000))
    assert result["health_state"] == "degraded"
    assert result["alert_id"] is not None
    assert len(engine.get_alerts("clf-health-2", active_only=True)) == 1
