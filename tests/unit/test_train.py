"""Unit tests for src/models/train.py: the fallback split and load_splits."""
import pytest

from conftest import make_synthetic_df
from src.models import train as T
from src.models.features import TARGET


@pytest.fixture(scope="module")
def df():
    return make_synthetic_df(1000, seed=3).reset_index(drop=True)


def test_fallback_split_is_70_15_15(df):
    train_df, val_df, test_df = T._fallback_split(df, TARGET, seed=42)
    assert len(train_df) == 700
    assert len(val_df) == 150
    assert len(test_df) == 150


def test_fallback_split_has_no_overlap_and_loses_no_rows(df):
    parts = T._fallback_split(df, TARGET, seed=42)
    indexes = [set(p.index) for p in parts]
    assert indexes[0].isdisjoint(indexes[1])
    assert indexes[0].isdisjoint(indexes[2])
    assert indexes[1].isdisjoint(indexes[2])
    assert set().union(*indexes) == set(df.index)


def test_fallback_split_keeps_class_balance(df):
    overall = df[TARGET].value_counts(normalize=True)
    for part in T._fallback_split(df, TARGET, seed=42):
        share = part[TARGET].value_counts(normalize=True)
        for cls, p in overall.items():
            assert share[cls] == pytest.approx(p, abs=0.03)


def test_fallback_split_is_reproducible(df):
    a = T._fallback_split(df, TARGET, seed=42)[2]
    b = T._fallback_split(df, TARGET, seed=42)[2]
    assert list(a.index) == list(b.index)


def test_load_splits_returns_three_parts_covering_all_rows(df, tmp_path):
    csv = tmp_path / "sleep.csv"
    df.to_csv(csv, index=False)
    train_df, val_df, test_df = T.load_splits(str(csv))
    assert len(train_df) + len(val_df) + len(test_df) == len(df)
