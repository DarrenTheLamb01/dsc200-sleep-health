"""Unit tests for src/models: train, evaluate, explain, save/load.

Uses a small synthetic dataset so the tests do not need the Kaggle CSV.
"""
import numpy as np
import pandas as pd
import pytest

from src.models import model as M
from src.models.features import (
    CATEGORICAL_FEATURES,
    CLASS_ORDER,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)

CATEGORIES = {
    "gender": ["Female", "Male", "Other"],
    "occupation": ["Student", "Nurse", "Retired"],
    "chronotype": ["Morning", "Neutral", "Evening"],
    "mental_health_condition": ["Healthy", "Anxiety", "Depression", "Both"],
    "day_type": ["Weekday", "Weekend"],
}


def _synthetic_df(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in NUMERIC_FEATURES})
    for c in CATEGORICAL_FEATURES:
        df[c] = rng.choice(CATEGORIES[c], size=n)
    # Make the label depend on stress so there is something to learn.
    bins = np.digitize(df["stress_score"], [-0.3, 0.8, 1.5])
    df[TARGET] = [CLASS_ORDER[b] for b in bins]
    return df


@pytest.fixture(scope="module")
def fitted():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(M, "PARAM_GRID", [{"learning_rate": 0.1, "max_leaf_nodes": 15}])
        train_df = _synthetic_df(600, seed=0)
        val_df = _synthetic_df(200, seed=1)
        return M.train(train_df, val_df), val_df


def test_train_returns_model(fitted):
    model, _ = fitted
    assert isinstance(model, M.SleepRiskModel)
    assert set(model.classes) == set(CLASS_ORDER)
    assert set(model.baseline) == set(FEATURES)
    assert 0.0 <= model.val_macro_f1 <= 1.0


def test_evaluate_metrics_in_range(fitted):
    model, val_df = fitted
    metrics = M.evaluate(model, val_df)
    for key in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "roc_auc_ovr"):
        assert key in metrics
    for cls in CLASS_ORDER:
        assert f"recall_{cls.lower()}" in metrics
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in metrics.values())


def test_predict_with_explanation_shape(fitted):
    model, val_df = fitted
    person = val_df.iloc[0][FEATURES].to_dict()
    out = M.predict_with_explanation(model, person)

    assert 0.0 <= out["risk_score"] <= 1.0
    assert 1 <= len(out["top_factors"]) <= M.TOP_K
    for name, contribution in out["top_factors"]:
        assert name in FEATURES
        assert isinstance(contribution, float)
    assert out["predicted_class"] in CLASS_ORDER
    assert sum(out["class_probabilities"].values()) == pytest.approx(1.0)


def test_high_stress_scores_higher_risk(fitted):
    model, _ = fitted
    low = M.predict_with_explanation(model, {"stress_score": -2.0})["risk_score"]
    high = M.predict_with_explanation(model, {"stress_score": 3.0})["risk_score"]
    assert high > low


def test_missing_fields_use_baseline(fitted):
    model, _ = fitted
    out = M.predict_with_explanation(model, {"stress_score": 1.0})
    assert 0.0 <= out["risk_score"] <= 1.0
    empty = M.predict_with_explanation(model, {})
    assert 0.0 <= empty["risk_score"] <= 1.0


def test_save_load_round_trip(fitted, tmp_path):
    model, val_df = fitted
    path = M.save_model(model, tmp_path / "m.joblib")
    loaded = M.load_model(path)
    np.testing.assert_allclose(
        loaded.predict_proba(val_df), model.predict_proba(val_df)
    )


def test_load_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        M.load_model(tmp_path / "nope.joblib")
