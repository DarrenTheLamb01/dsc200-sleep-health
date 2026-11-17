"""Functional smoke test: the Dash app builds and has a layout."""
from src.app.app import create_app


def test_app_builds_with_layout():
    app = create_app()
    assert app.layout is not None
