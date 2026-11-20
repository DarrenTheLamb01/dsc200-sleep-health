"""Model training, evaluation, and explanations. Owner: Modeling lead."""
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.models.features import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_preprocessor,
)

SEED = 42

# Small grid searched on the validation set. Kept small so training stays fast.
PARAM_GRID: List[Dict[str, Any]] = [
    {"learning_rate": 0.1, "max_leaf_nodes": 31},
    {"learning_rate": 0.1, "max_leaf_nodes": 63},
    {"learning_rate": 0.05, "max_leaf_nodes": 31},
]


@dataclass
class SleepRiskModel:
    """Fitted pipeline plus what we need to explain its predictions."""

    pipeline: Pipeline
    params: Dict[str, Any]
    baseline: Dict[str, Any]  # typical value per feature, from training data
    val_macro_f1: float
    search_results: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def classes(self) -> List[str]:
        return list(self.pipeline.classes_)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(df[FEATURES])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(df[FEATURES])


def _make_pipeline(params: Dict[str, Any]) -> Pipeline:
    clf = HistGradientBoostingClassifier(
        class_weight="balanced",  # classes are imbalanced (Severe is ~4%)
        max_iter=300,
        early_stopping=True,
        random_state=SEED,
        **params,
    )
    return Pipeline([("pre", build_preprocessor()), ("clf", clf)])


def _baseline(df: pd.DataFrame) -> Dict[str, Any]:
    """Median for numeric features, most common value for categorical ones."""
    base: Dict[str, Any] = {c: float(df[c].median()) for c in NUMERIC_FEATURES}
    base.update({c: df[c].mode().iloc[0] for c in CATEGORICAL_FEATURES})
    return base


def train(train_df: pd.DataFrame, val_df: pd.DataFrame) -> SleepRiskModel:
    """Fit the sleep disorder risk classifier and return the fitted model.

    Tries each setting in PARAM_GRID and keeps the one with the best macro-F1
    on `val_df`. The test set is never touched here.
    """
    X_tr, y_tr = train_df[FEATURES], train_df[TARGET]
    X_val, y_val = val_df[FEATURES], val_df[TARGET]

    results = []
    best = None
    for params in PARAM_GRID:
        pipe = _make_pipeline(params).fit(X_tr, y_tr)
        score = f1_score(y_val, pipe.predict(X_val), average="macro")
        results.append({**params, "val_macro_f1": round(float(score), 4)})
        if best is None or score > best[0]:
            best = (score, params, pipe)

    score, params, pipe = best
    return SleepRiskModel(
        pipeline=pipe,
        params=params,
        baseline=_baseline(train_df),
        val_macro_f1=float(score),
        search_results=results,
    )


def evaluate(model: SleepRiskModel, df: pd.DataFrame) -> Dict[str, float]:
    """Return a dict of metrics for `model` on `df`.

    Macro-F1 is the headline metric because the classes are imbalanced and
    the rare classes (Moderate, Severe) matter most.
    """
    y_true = df[TARGET]
    y_pred = model.predict(df)
    proba = model.predict_proba(df)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc_ovr": roc_auc_score(
            y_true, proba, multi_class="ovr", average="macro", labels=model.classes
        ),
    }
    recalls = recall_score(
        y_true, y_pred, labels=model.classes, average=None, zero_division=0
    )
    for cls, r in zip(model.classes, recalls):
        metrics[f"recall_{cls.lower()}"] = r

    return {k: float(v) for k, v in metrics.items()}


def predict_with_explanation(model: Any, input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Predict risk for one person.

    Returns {"risk_score": float in [0, 1], "top_factors": list of
    (feature_name, contribution) tuples}.
    """
    raise NotImplementedError
