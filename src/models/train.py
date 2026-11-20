"""Train the sleep disorder risk model, report metrics, and save it.

Usage:
    python -m src.models.train [--data PATH] [--out PATH]
"""
import argparse
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.features import TARGET
from src.models.model import DEFAULT_MODEL_PATH, SEED, evaluate, save_model, train

DEFAULT_DATA_PATH = "data/sleep_health_dataset.csv"


def _fallback_split(
    df: pd.DataFrame, target: str, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70/15/15 stratified split, used until src/data/pipeline.py exists."""
    train_df, rest = train_test_split(
        df, test_size=0.30, stratify=df[target], random_state=seed
    )
    val_df, test_df = train_test_split(
        rest, test_size=0.50, stratify=rest[target], random_state=seed
    )
    return train_df, val_df, test_df


def load_splits(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use the data lead's pipeline when available, otherwise a local fallback."""
    try:
        from src.data.pipeline import clean_data, load_data, split_data
    except ImportError:
        print("src.data.pipeline not found; using fallback loading and split.")
        df = pd.read_csv(path)
        return _fallback_split(df, TARGET, SEED)

    df = clean_data(load_data(path))
    return split_data(df, TARGET, SEED)


def _print_metrics(name: str, metrics: dict) -> None:
    print(f"\n{name}")
    for k, v in metrics.items():
        print(f"  {k:20s} {v:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--out", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    train_df, val_df, test_df = load_splits(args.data)
    print(f"Rows: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    model = train(train_df, val_df)
    print("\nHyperparameter search (validation macro-F1):")
    for r in model.search_results:
        print(f"  {r}")
    print(f"Chosen: {model.params}")

    _print_metrics("Validation", evaluate(model, val_df))
    _print_metrics("Test", evaluate(model, test_df))

    path = save_model(model, args.out)
    print(f"\nSaved model to {path}")


if __name__ == "__main__":
    main()
