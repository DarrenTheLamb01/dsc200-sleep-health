"""Unit tests for the chart builders in src/app/app.py."""
import pytest

from src.models.features import CLASS_ORDER


def test_probability_chart_has_one_bar_per_class_in_severity_order(app_module):
    probs = {"Healthy": 0.7, "Mild": 0.2, "Moderate": 0.07, "Severe": 0.03}
    fig = app_module.probability_chart(probs)

    bar = fig.data[0]
    assert list(bar.x) == CLASS_ORDER
    assert list(bar.y) == pytest.approx([70.0, 20.0, 7.0, 3.0])


def test_probability_chart_treats_missing_class_as_zero(app_module):
    fig = app_module.probability_chart({"Healthy": 1.0})
    assert list(fig.data[0].y) == pytest.approx([100.0, 0.0, 0.0, 0.0])


def test_factors_chart_colors_by_sign_and_uses_readable_labels(app_module):
    factors = [("stress_score", 0.12), ("sleep_duration_hrs", -0.05)]
    fig = app_module.factors_chart(factors)

    bar = fig.data[0]
    # The chart reverses the list so the biggest factor is drawn on top.
    assert list(bar.y) == ["Sleep duration (hours)", "Stress score"]
    assert list(bar.x) == pytest.approx([-0.05, 0.12])
    assert bar.marker.color[0] != bar.marker.color[1]  # negative vs positive
