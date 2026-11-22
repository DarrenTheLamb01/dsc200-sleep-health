"""End-to-end checks on the real Kaggle CSV and the saved model.

These are skipped on a fresh clone. To enable them:
    1. put the CSV at data/sleep_health_dataset.csv
    2. run: python -m src.models.train
"""
from pathlib import Path

import pytest

from src.data.pipeline import clean_data, load_data, split_data
from src.models.features import CLASS_ORDER, FEATURES, TARGET
from src.models.model import evaluate, load_model, predict_with_explanation

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "sleep_health_dataset.csv"
MODEL = ROOT / "models" / "sleep_risk_model.joblib"

needs_data = pytest.mark.skipif(not CSV.exists(), reason="Kaggle CSV not in data/")
needs_model = pytest.mark.skipif(
    not (CSV.exists() and MODEL.exists()),
    reason="needs the CSV and a trained model (python -m src.models.train)",
)


@pytest.fixture(scope="module")
def splits():
    df = clean_data(load_data(CSV))
    return split_data(df, TARGET, 42)


@needs_data
def test_real_csv_loads_cleans_and_splits(splits):
    train_df, val_df, test_df = splits
    assert len(train_df) + len(val_df) + len(test_df) >= 90_000
    assert len(train_df) > 4 * len(val_df)  # roughly 70 / 15 / 15
    for part in splits:
        assert set(FEATURES) <= set(part.columns)
        assert part[TARGET].notna().all()
        assert set(part[TARGET]) == set(CLASS_ORDER)


@needs_model
def test_saved_model_meets_quality_floor_on_held_out_test_set(splits):
    metrics = evaluate(load_model(MODEL), splits[2])
    assert metrics["macro_f1"] >= 0.85
    assert metrics["roc_auc_ovr"] >= 0.97


@needs_model
def test_real_model_scores_a_stressed_short_sleeper_above_a_rested_one():
    model = load_model(MODEL)
    stressed = {"stress_score": 9.0, "sleep_duration_hrs": 4.0, "caffeine_mg_before_bed": 300.0}
    rested = {"stress_score": 1.0, "sleep_duration_hrs": 8.0, "caffeine_mg_before_bed": 0.0}
    hi = predict_with_explanation(model, stressed)
    lo = predict_with_explanation(model, rested)
    assert hi["risk_score"] > lo["risk_score"]
    assert hi["predicted_class"] in CLASS_ORDER
