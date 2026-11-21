
"""Sleep, Lifestyle, and Daily Performance dashboard."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

from src.models.features import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    CLASS_ORDER,
)
from src.models.model import load_model, predict_with_explanation


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "sleep_risk_model.joblib"
DATA_PATH = ROOT / "data" / "sleep_health_dataset.csv"

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

BINARY_FEATURES = {
    "exercise_day",
    "sleep_aid_used",
    "shift_work",
}

LABELS = {
    "age": "Age (years)",
    "bmi": "BMI",
    "sleep_duration_hrs": "Sleep duration (hours)",
    "sleep_latency_mins": "Time to fall asleep (minutes)",
    "wake_episodes_per_night": "Nighttime awakenings",
    "caffeine_mg_before_bed": "Caffeine before bed (mg)",
    "alcohol_units_before_bed": "Alcohol before bed (units)",
    "screen_time_before_bed_mins": "Screen time before bed (minutes)",
    "exercise_day": "Did you exercise today?",
    "steps_that_day": "Steps today",
    "nap_duration_mins": "Nap duration (minutes)",
    "stress_score": "Stress score",
    "work_hours_that_day": "Hours worked today",
    "heart_rate_resting_bpm": "Resting heart rate (bpm)",
    "sleep_aid_used": "Did you use a sleep aid?",
    "shift_work": "Did you work a shift?",
    "room_temperature_celsius": "Room temperature (°C)",
    "weekend_sleep_diff_hrs": "Weekend sleep difference (hours)",
    "gender": "Gender",
    "occupation": "Occupation",
    "chronotype": "Chronotype",
    "mental_health_condition": "Mental health condition",
    "day_type": "Day type",
}

SECTIONS = {
    "About you": [
        "age", "gender", "occupation", "bmi",
    ],
    "Your sleep": [
        "sleep_duration_hrs",
        "sleep_latency_mins",
        "wake_episodes_per_night",
        "nap_duration_mins",
        "chronotype",
        "weekend_sleep_diff_hrs",
        "sleep_aid_used",
    ],
    "Daily lifestyle": [
        "caffeine_mg_before_bed",
        "alcohol_units_before_bed",
        "screen_time_before_bed_mins",
        "exercise_day",
        "steps_that_day",
        "stress_score",
        "work_hours_that_day",
        "shift_work",
        "day_type",
    ],
    "Health and environment": [
        "heart_rate_resting_bpm",
        "room_temperature_celsius",
        "mental_health_condition",
    ],
}


def read_data():
    """Read the dataset to obtain valid input values."""
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


def make_field(feature, model, df):
    """Build an input matching the trained model's feature."""
    label = LABELS.get(feature, feature)

    baseline = model.baseline.get(feature)

    if feature in BINARY_FEATURES:
        control = dcc.Dropdown(
            id=f"field-{feature}",
            options=[
                {"label": "Yes", "value": 1},
                {"label": "No", "value": 0},
            ],
            value=int(baseline) if baseline is not None else 0,
            clearable=False,
            className="form-dropdown",
        )

    elif feature in CATEGORICAL_FEATURES:
        if df is not None and feature in df.columns:
            categories = sorted(
                df[feature].dropna().astype(str).unique().tolist()
            )
        else:
            categories = [str(baseline)] if baseline is not None else []

        control = dcc.Dropdown(
            id=f"field-{feature}",
            options=[
                {"label": item, "value": item}
                for item in categories
            ],
            value=str(baseline) if baseline is not None else None,
            clearable=False,
            className="form-dropdown",
        )

    else:
        control = dcc.Input(
            id=f"field-{feature}",
            type="number",
            value=float(baseline) if baseline is not None else None,
            className="form-input",
            debounce=True,
        )

    return html.Div(
        [
            html.Label(label, className="field-label"),
            control,
        ],
        className="form-field",
    )


def make_section(title, features, model, df):
    return html.Div(
        [
            html.H3(title, className="section-title"),
            html.Div(
                [
                    make_field(feature, model, df)
                    for feature in features
                ],
                className="form-grid",
            ),
        ],
        className="form-section",
    )


def probability_chart(probabilities):
    values = [
        float(probabilities.get(name, 0)) * 100
        for name in CLASS_ORDER
    ]

    fig = go.Figure(
        go.Bar(
            x=CLASS_ORDER,
            y=values,
            marker_color=[
                "#84c9ad",
                "#a79be7",
                "#edbf80",
                "#e58b9b",
            ],
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=310,
        margin=dict(l=35, r=15, t=25, b=35),
        yaxis=dict(title="Probability (%)", range=[0, 110]),
        xaxis_title=None,
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        font=dict(family="Arial"),
    )

    return fig


def factors_chart(factors):
    names = [
        LABELS.get(name, name.replace("_", " ").title())
        for name, _ in factors
    ][::-1]

    values = [float(value) for _, value in factors][::-1]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=[
                "#e58b9b" if value > 0 else "#8274d8"
                for value in values
            ],
            hovertemplate="%{y}: %{x:+.4f}<extra></extra>",
        )
    )

    fig.add_vline(x=0, line_color="#aaa6ba")

    fig.update_layout(
        template="plotly_white",
        height=350,
        margin=dict(l=10, r=20, t=20, b=40),
        xaxis_title="Change in expected severity",
        yaxis_title=None,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial"),
    )

    return fig


def create_app() -> Dash:
    model = load_model(MODEL_PATH)
    df = read_data()

    app = Dash(
        __name__,
        assets_folder=str(Path(__file__).resolve().parent / "assets"),
    )

    app.title = "SleepWell | Sleep Health Dashboard"

    app.layout = html.Div(
        [
            html.Header(
                [
                    html.Div(
                        [
                            html.Span("☾", className="logo-icon"),
                            html.Span("SleepWell", className="logo-text"),
                        ],
                        className="brand",
                    ),
                    html.Span(
                        "UC San Diego · DSC 200",
                        className="nav-subtitle",
                    ),
                ],
                className="navbar",
            ),

            html.Main(
                [
                    html.Div(
                        [
                            html.Span(
                                "SLEEP HEALTH ANALYTICS",
                                className="eyebrow",
                            ),
                            html.H1(
                                "Better sleep starts with understanding."
                            ),
                            html.P(
                                "Explore how your daily habits relate "
                                "to sleep health. Enter your lifestyle "
                                "information to see your predicted "
                                "sleep disorder risk."
                            ),
                        ],
                        className="hero",
                    ),

                    html.Div(
                        [
                            html.Section(
                                [
                                    html.Div(
                                        [
                                            html.Span(
                                                "01 / YOUR PROFILE",
                                                className="eyebrow",
                                            ),
                                            html.H2("Lifestyle information"),
                                            html.P(
                                                "Adjust the values below "
                                                "to match your daily habits."
                                            ),
                                        ],
                                        className="card-header",
                                    ),

                                    *[
                                        make_section(
                                            title, fields, model, df
                                        )
                                        for title, fields in SECTIONS.items()
                                    ],

                                    html.Button(
                                        "Analyze my sleep risk",
                                        id="predict-button",
                                        n_clicks=0,
                                        className="primary-button",
                                    ),
                                ],
                                className="card input-panel",
                            ),

                            html.Section(
                                [
                                    html.Div(
                                        [
                                            html.Span(
                                                "02 / YOUR RESULTS",
                                                className="eyebrow",
                                            ),
                                            html.H2("Your sleep analysis"),
                                            html.P(
                                                "Your model-generated "
                                                "sleep health overview."
                                            ),
                                        ],
                                        className="card-header",
                                    ),

                                    dcc.Loading(
                                        html.Div(
                                            [
                                                html.Div(
                                                    "☾",
                                                    className="empty-icon",
                                                ),
                                                html.H3(
                                                    "Ready when you are"
                                                ),
                                                html.P(
                                                    "Adjust your lifestyle "
                                                    "information and click "
                                                    "Analyze to see your results."
                                                ),
                                            ],
                                            id="results",
                                            className="results-content",
                                        ),
                                        type="circle",
                                    ),
                                ],
                                className="card results-panel",
                            ),
                        ],
                        className="dashboard-grid",
                    ),

                    html.Footer(
                        [
                            html.Span(
                                "SleepWell · DSC 200 Group Project"
                            ),
                            html.Span(
                                "Synthetic data · Educational use only · "
                                "Not medical advice"
                            ),
                        ],
                        className="footer",
                    ),
                ],
                className="container",
            ),
        ]
    )

    states = [
        State(f"field-{feature}", "value")
        for feature in FEATURES
    ]

    @app.callback(
        Output("results", "children"),
        Input("predict-button", "n_clicks"),
        *states,
        prevent_initial_call=True,
    )
    def update_prediction(n_clicks, *values):
        user_inputs = {}

        for feature, value in zip(FEATURES, values):
            if value is None or value == "":
                continue

            if feature in NUMERIC_FEATURES:
                user_inputs[feature] = float(value)
            else:
                user_inputs[feature] = str(value)

        try:
            result = predict_with_explanation(
                model, user_inputs
            )
        except Exception as exc:
            return html.Div(
                [
                    html.H3("Prediction unavailable"),
                    html.P(str(exc)),
                ],
                className="error-message",
            )

        score = float(result["risk_score"])
        category = result["predicted_class"]
        probabilities = result["class_probabilities"]
        factors = result["top_factors"]

        return html.Div(
            [
                html.Div(
                    [
                        html.Span(
                            "YOUR SLEEP RISK",
                            className="eyebrow",
                        ),
                        html.Div(
                            f"{score * 100:.1f}",
                            className="risk-number",
                        ),
                        html.P(
                            "Expected severity score out of 100",
                            className="risk-label",
                        ),
                        html.Div(
                            f"Predicted class: {category}",
                            className="risk-category",
                        ),
                    ],
                    className="risk-summary",
                ),

                html.P(
                    "The severity score summarizes the model's "
                    "predictions across four risk categories. "
                    "It is not the probability of a disorder.",
                    className="info-note",
                ),

                html.H3(
                    "Class probabilities",
                    className="chart-title",
                ),
                dcc.Graph(
                    figure=probability_chart(probabilities),
                    config={"displayModeBar": False},
                    responsive=True,
                ),

                html.H3(
                    "Top contributing factors",
                    className="chart-title",
                ),
                dcc.Graph(
                    figure=factors_chart(factors),
                    config={"displayModeBar": False},
                    responsive=True,
                ),

                html.P(
                    "Positive contributions increase modeled "
                    "severity relative to typical values. "
                    "Negative contributions decrease it. "
                    "These comparisons do not prove causation.",
                    className="info-note",
                ),
            ],
            className="prediction-output",
        )

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
