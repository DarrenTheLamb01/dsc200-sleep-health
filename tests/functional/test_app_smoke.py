"""Functional tests: the Dash app, driven through its real HTTP callback.

Covers the user flow from the activity diagram: open app -> enter sleep and
lifestyle information -> click Analyze -> see risk prediction + charts, or an
error message if the prediction fails.
"""
import re

import pytest

from src.models.features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES

BINARY = {"exercise_day", "sleep_aid_used", "shift_work"}


@pytest.fixture
def app(app_module):
    return app_module.create_app()


@pytest.fixture
def default_inputs(tiny_model):
    """What the form shows on first load: the model's typical value per field."""
    return {
        f: (float(v) if f in NUMERIC_FEATURES else str(v))
        for f, v in tiny_model.baseline.items()
    }


def click_analyze(app, values):
    """Simulate pressing 'Analyze my sleep risk' and return the `results` tree."""
    payload = {
        "output": "results.children",
        "outputs": {"id": "results", "property": "children"},
        "inputs": [{"id": "predict-button", "property": "n_clicks", "value": 1}],
        "state": [
            {"id": f"field-{f}", "property": "value", "value": values.get(f)}
            for f in FEATURES
        ],
        "changedPropIds": ["predict-button.n_clicks"],
    }
    resp = app.server.test_client().post("/_dash-update-component", json=payload)
    assert resp.status_code == 200, resp.get_data(as_text=True)
    return resp.get_json()["response"]["results"]["children"]


def walk(node):
    """Yield every component dict in a serialized Dash tree."""
    if isinstance(node, dict):
        yield node
        yield from walk(node.get("props", {}).get("children"))
    elif isinstance(node, list):
        for item in node:
            yield from walk(item)


def text_of(tree):
    """All plain strings in the tree, in order."""
    out = []

    def rec(n):
        if isinstance(n, str):
            out.append(n)
        elif isinstance(n, dict):
            rec(n.get("props", {}).get("children"))
        elif isinstance(n, list):
            for item in n:
                rec(item)

    rec(tree)
    return out


def risk_number(tree):
    """The 0-100 score shown to the user."""
    return float(
        next(
            n["props"]["children"]
            for n in walk(tree)
            if n.get("props", {}).get("className") == "risk-number"
        )
    )


def test_app_builds_with_layout(app):
    assert app.layout is not None


def test_form_has_an_input_for_every_model_feature(app):
    ids = {c.id for c in app.layout._traverse() if getattr(c, "id", None)}
    missing = [f for f in FEATURES if f"field-{f}" not in ids]
    assert not missing, f"form is missing inputs for: {missing}"
    assert "predict-button" in ids
    assert "results" in ids


def test_form_defaults_to_typical_values(app, tiny_model):
    by_id = {c.id: c for c in app.layout._traverse() if getattr(c, "id", None)}
    for f in NUMERIC_FEATURES:
        if f not in BINARY:
            assert by_id[f"field-{f}"].value == pytest.approx(tiny_model.baseline[f])
    for f in CATEGORICAL_FEATURES:
        assert by_id[f"field-{f}"].value == str(tiny_model.baseline[f])


def test_analyze_shows_score_class_and_charts(app, default_inputs):
    tree = click_analyze(app, default_inputs)
    score = risk_number(tree)
    assert 0.0 <= score <= 100.0

    text = " ".join(text_of(tree))
    match = re.search(r"Predicted class: (\w+)", text)
    assert match and match.group(1) in {"Healthy", "Mild", "Moderate", "Severe"}

    graphs = [n for n in walk(tree) if n.get("type") == "Graph"]
    assert len(graphs) == 2  # class probabilities + top contributing factors


def test_higher_stress_gives_higher_displayed_score(app, default_inputs):
    low = risk_number(click_analyze(app, {**default_inputs, "stress_score": -2.0}))
    high = risk_number(click_analyze(app, {**default_inputs, "stress_score": 3.0}))
    assert high > low


def test_blank_fields_still_produce_a_prediction(app, default_inputs):
    values = {**default_inputs, "age": None, "stress_score": "", "occupation": None}
    tree = click_analyze(app, values)
    assert 0.0 <= risk_number(tree) <= 100.0


def test_prediction_failure_shows_error_message_not_results(
    app, app_module, default_inputs, monkeypatch
):
    def boom(*args, **kwargs):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(app_module, "predict_with_explanation", boom)
    tree = click_analyze(app, default_inputs)
    text = " ".join(text_of(tree))
    assert "Prediction unavailable" in text
    assert "model exploded" in text
    assert not [n for n in walk(tree) if n.get("type") == "Graph"]
