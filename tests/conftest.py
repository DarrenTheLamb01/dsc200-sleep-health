"""Shared fixtures.

Everything here is built from a small synthetic dataset, so the tests need
neither the Kaggle CSV (data/ is gitignored) nor a saved model file
(models/*.joblib is gitignored). They run on a fresh clone.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import model as M
from src.models.features import (
    CATEGORICAL_FEATURES,
    CLASS_ORDER,
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


def make_synthetic_df(n: int, seed: int) -> pd.DataFrame:
    """Random rows whose label depends on stress_score, so there is signal."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in NUMERIC_FEATURES})
    for c in CATEGORICAL_FEATURES:
        df[c] = rng.choice(CATEGORIES[c], size=n)
    bins = np.digitize(df["stress_score"], [-0.3, 0.8, 1.5])
    df[TARGET] = [CLASS_ORDER[b] for b in bins]
    return df


@pytest.fixture(scope="session")
def tiny_model():
    """A fast-to-train SleepRiskModel (one hyperparameter setting)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(M, "PARAM_GRID", [{"learning_rate": 0.1, "max_leaf_nodes": 15}])
        return M.train(make_synthetic_df(600, 0), make_synthetic_df(200, 1))


@pytest.fixture(scope="session")
def app_module(tiny_model):
    """Import src.app.app with load_model stubbed to return `tiny_model`.

    src/app/app.py calls create_app() at import time, which would otherwise
    require models/sleep_risk_model.joblib to exist.
    """
    # The stub is only active during the import. app.py keeps its own reference
    # to it, and M.load_model is restored for every other test straight after.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(M, "load_model", lambda *args, **kwargs: tiny_model)
        sys.modules.pop("src.app.app", None)
        module = importlib.import_module("src.app.app")
    # Do not depend on whether the real CSV happens to be on disk.
    module.DATA_PATH = Path("no-such-file.csv")
    yield module
    sys.modules.pop("src.app.app", None)
