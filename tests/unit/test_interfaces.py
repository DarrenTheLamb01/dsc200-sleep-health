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
