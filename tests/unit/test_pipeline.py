"""Unit tests for src/data/pipeline.py: load, clean, split."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_synthetic_df
from src.data import pipeline as P
from src.models import train as T

TARGET = P.TARGET


def _raw(**overrides) -> pd.DataFrame:
    """Four tidy rows with an ID column; overrides replace whole columns."""
    df = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "age": [25, 40, 33, 58],
            "stress_score": [3, 5, 7, 9],
            "occupation": ["Nurse", "Student", "Nurse", "Retired"],
            TARGET: ["Healthy", "Mild", "Moderate", "Severe"],
        }
    )
    for col, values in overrides.items():
        df[col] = values
    return df


def test_load_data_reads_csv(tmp_path):
    csv = tmp_path / "sleep.csv"
    _raw().to_csv(csv, index=False)
    df = P.load_data(csv)
    assert len(df) == 4
    assert list(df.columns) == ["person_id", "age", "stress_score", "occupation", TARGET]


def test_load_data_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        P.load_data(tmp_path / "nope.csv")


def test_clean_strips_whitespace_from_names_and_values():
    raw = _raw(occupation=[" Nurse ", "Student", "Nurse\t", "Retired"])
    raw = raw.rename(columns={"age": " age "})
    out = P.clean_data(raw)
    assert "age" in out.columns and " age " not in out.columns
    assert set(out["occupation"]) == {"Nurse", "Student", "Retired"}


def test_clean_drops_rows_with_missing_target():
    out = P.clean_data(_raw(**{TARGET: ["Healthy", None, "Moderate", "Severe"]}))
    assert len(out) == 3
    assert out[TARGET].notna().all()


def test_clean_drops_duplicates_ignoring_person_id():
    raw = pd.concat([_raw(), _raw().iloc[[0]].assign(person_id=99)], ignore_index=True)
    assert len(raw) == 5
    assert len(P.clean_data(raw)) == 4


def test_clean_clips_out_of_range_values_but_not_valid_ones():
    out = P.clean_data(_raw(age=[25, 200, 33, -5], stress_score=[3, 5, 7, 9]))
    assert list(out["age"]) == [25, 120, 33, 0]
    assert list(out["stress_score"]) == [3, 5, 7, 9]


def test_clean_leaves_missing_feature_values_and_target_unchanged():
    raw = _raw(age=[25, np.nan, 33, 58])
    out = P.clean_data(raw)
    assert out["age"].isna().sum() == 1  # left for the model's imputer
    assert list(out[TARGET]) == list(raw[TARGET])
    assert list(out["person_id"]) == [1, 2, 3, 4]


def test_clean_does_not_modify_its_input():
    raw = _raw(age=[25, 200, 33, 58])
    before = raw.copy()
    P.clean_data(raw)
    pd.testing.assert_frame_equal(raw, before)


@pytest.fixture(scope="module")
def df():
    return make_synthetic_df(1000, seed=5)


def test_split_is_70_15_15_with_no_overlap(df):
    train_df, val_df, test_df = P.split_data(df, TARGET, seed=42)
    assert (len(train_df), len(val_df), len(test_df)) == (700, 150, 150)
    idx = [set(p.index) for p in (train_df, val_df, test_df)]
    assert idx[0].isdisjoint(idx[1]) and idx[0].isdisjoint(idx[2]) and idx[1].isdisjoint(idx[2])
    assert set().union(*idx) == set(df.index)


def test_split_is_stratified_by_target(df):
    overall = df[TARGET].value_counts(normalize=True)
    for part in P.split_data(df, TARGET, seed=42):
        share = part[TARGET].value_counts(normalize=True)
        for cls, p in overall.items():
            assert share[cls] == pytest.approx(p, abs=0.03)


def test_split_same_seed_same_rows_and_matches_train_fallback(df):
    a = P.split_data(df, TARGET, seed=42)
    b = P.split_data(df, TARGET, seed=42)
    fallback = T._fallback_split(df, TARGET, seed=42)
    for x, y, z in zip(a, b, fallback):
        assert list(x.index) == list(y.index) == list(z.index)
