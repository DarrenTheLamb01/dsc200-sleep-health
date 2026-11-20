"""Model training, evaluation, and explanations. Owner: Modeling lead."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

import joblib
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
    CLASS_ORDER,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_preprocessor,
)

SEED = 42

DEFAULT_MODEL_PATH = Path("models") / "sleep_risk_model.joblib"

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


def save_model(
    model: SleepRiskModel, path: Union[str, Path] = DEFAULT_MODEL_PATH
) -> Path:
    """Save a trained model to disk and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Union[str, Path] = DEFAULT_MODEL_PATH) -> SleepRiskModel:
    """Load a model saved with save_model."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No model at {path}. Train one with: python -m src.models.train"
        )
    return joblib.load(path)


TOP_K = 5


def _risk_scores(model: SleepRiskModel, df: pd.DataFrame) -> np.ndarray:
    """Expected severity in [0, 1]: Healthy=0, Mild=1/3, Moderate=2/3, Severe=1."""
    proba = model.predict_proba(df)
    severity = np.array([CLASS_ORDER.index(c) for c in model.classes])
    return proba @ severity / (len(CLASS_ORDER) - 1)


def predict_with_explanation(
    model: SleepRiskModel, input_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Predict risk for one person.

    Returns {"risk_score": float in [0, 1], "top_factors": list of
    (feature_name, contribution) tuples, "predicted_class": str,
    "class_probabilities": {class: prob}}.

    A factor's contribution is how much the risk score drops (or rises) if
    that one feature is reset to a typical value from the training data.
    Positive means the feature is pushing this person's risk up. Fields
    missing from `input_dict` are filled with the typical value.
    """
    person = {f: input_dict.get(f, model.baseline[f]) for f in FEATURES}

    # Row 0 is the person as entered; row i+1 has feature i reset to baseline.
    rows = [person] + [{**person, f: model.baseline[f]} for f in FEATURES]
    batch = pd.DataFrame(rows, columns=FEATURES)
    scores = _risk_scores(model, batch)

    risk = float(scores[0])
    contributions = {f: risk - float(s) for f, s in zip(FEATURES, scores[1:])}
    top = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)

    proba = model.predict_proba(batch.iloc[[0]])[0]
    return {
        "risk_score": risk,
        "top_factors": [(f, round(c, 4) + 0.0) for f, c in top[:TOP_K]],  # +0.0 avoids -0.0
        "predicted_class": model.classes[int(np.argmax(proba))],
        "class_probabilities": {c: float(p) for c, p in zip(model.classes, proba)},
    }
