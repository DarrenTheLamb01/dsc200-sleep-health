"""Data loading, cleaning, and splitting for the sleep risk project.

Which columns the model uses lives in src/models/features.py. This module
does NOT drop model-excluded columns, and it does not impute or scale
(the model's preprocessing does that, fit on train only).
"""
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "sleep_disorder_risk"
ID_COLUMN = "person_id"

# Physically plausible bounds (not fitted to the data). Values outside a bound
# are clipped to it. Chosen so valid values in the dataset are never altered.
VALID_RANGES = {
    "age": (0, 120),
    "bmi": (10, 70),
    "sleep_duration_hrs": (0, 24),
    "sleep_quality_score": (1, 10),
    "rem_percentage": (0, 100),
    "deep_sleep_percentage": (0, 100),
    "sleep_latency_mins": (0, 600),
    "wake_episodes_per_night": (0, 50),
    "caffeine_mg_before_bed": (0, 2000),
    "alcohol_units_before_bed": (0, 50),
    "screen_time_before_bed_mins": (0, 1440),
    "steps_that_day": (0, 100000),
    "nap_duration_mins": (0, 1440),
    "stress_score": (1, 10),
    "work_hours_that_day": (0, 24),
    "heart_rate_resting_bpm": (25, 220),
    "room_temperature_celsius": (-10, 50),
    "weekend_sleep_diff_hrs": (-24, 24),
    "cognitive_performance_score": (0, 100),
}


def load_data(path) -> pd.DataFrame:
    """Read the raw CSV and return a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Download the CSV from the Kaggle "
            "link in the README and place it in the top-level data/ folder."
        )
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of df.

    1. Strip whitespace from column names and text values.
    2. Drop rows with a missing target (unusable for training).
    3. Drop duplicate rows, ignoring the ID column.
    4. Clip numeric columns to VALID_RANGES (outlier handling). The target
       and ID are never modified.
    Missing feature values are left for the model's imputer.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            df[col] = df[col].str.strip()

    if TARGET in df.columns:
        df = df.dropna(subset=[TARGET])

    dedupe_cols = [c for c in df.columns if c != ID_COLUMN]
    df = df.drop_duplicates(subset=dedupe_cols)

    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    return df


def split_data(
    df: pd.DataFrame, target: str, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 split. Returns (train, val, test).

    Uses the same two-step calls as train.py's fallback, so the same seed
    gives identical rows. Original indices are preserved.
    """
    train_df, rest = train_test_split(
        df, test_size=0.30, stratify=df[target], random_state=seed
    )
    val_df, test_df = train_test_split(
        rest, test_size=0.50, stratify=rest[target], random_state=seed
    )
    return train_df, val_df, test_df