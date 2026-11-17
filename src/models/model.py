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
