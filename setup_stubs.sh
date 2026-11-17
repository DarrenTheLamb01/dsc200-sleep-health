#!/usr/bin/env bash
# Run from the repo root. Creates stub modules, tests, and pytest config.
set -e
mkdir -p src/data src/models src/app tests/unit tests/functional
touch src/__init__.py src/data/__init__.py src/models/__init__.py src/app/__init__.py

cat > pytest.ini <<'PY'
[pytest]
pythonpath = .
testpaths = tests
PY

cat > src/data/pipeline.py <<'PY'
"""Data loading, cleaning, and splitting. Owner: Data lead."""
from typing import Tuple

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load the raw Sleep Health CSV from `path` into a DataFrame."""
    raise NotImplementedError


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of `df` (missing values, outliers, dtypes)."""
    raise NotImplementedError


def split_data(
    df: pd.DataFrame, target: str, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/validation/test split on `target`.

    Returns (train_df, val_df, test_df). Must not leak test data into
    any preprocessing fit.
    """
    raise NotImplementedError
PY

cat > src/models/model.py <<'PY'
"""Model training, evaluation, and explanations. Owner: Modeling lead."""
from typing import Any, Dict

import pandas as pd


def train(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Any:
    """Fit the sleep disorder risk classifier and return the fitted model."""
    raise NotImplementedError


def evaluate(model: Any, df: pd.DataFrame) -> Dict[str, float]:
    """Return a dict of metrics (e.g. accuracy, macro-F1) for `model` on `df`."""
    raise NotImplementedError


def predict_with_explanation(model: Any, input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Predict risk for one person.

    Returns {"risk_score": float in [0, 1], "top_factors": list of
    (feature_name, contribution) tuples}.
    """
    raise NotImplementedError
PY

cat > src/app/app.py <<'PY'
"""Dash app: input form, risk score, and charts. Owner: App lead."""
from dash import Dash, html


def create_app() -> Dash:
    """Build and return the Dash app (placeholder layout)."""
    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H1("Sleep, Lifestyle, and Daily Performance"),
            html.Div(id="input-form", children="Input form goes here."),
            html.Div(id="results", children="Risk score and charts go here."),
        ]
    )
    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
PY

cat > tests/unit/test_interfaces.py <<'PY'
"""Unit tests: agreed function interfaces exist and are callable."""
import inspect

from src.data import pipeline
from src.models import model


def test_data_interface():
    for name in ("load_data", "clean_data", "split_data"):
        assert callable(getattr(pipeline, name))


def test_split_data_signature():
    params = inspect.signature(pipeline.split_data).parameters
    assert list(params) == ["df", "target", "seed"]


def test_model_interface():
    for name in ("train", "evaluate", "predict_with_explanation"):
        assert callable(getattr(model, name))
PY

cat > tests/functional/test_app_smoke.py <<'PY'
"""Functional smoke test: the Dash app builds and has a layout."""
from src.app.app import create_app


def test_app_builds_with_layout():
    app = create_app()
    assert app.layout is not None
PY

echo "Done. Run: pytest"
