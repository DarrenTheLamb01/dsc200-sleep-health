"""Dash app: input form, risk score, and charts. Owner: App lead."""
from dash import Dash, html


def create_app() -> Dash:
    """Build and return the Dash app (placeholder layout)."""
    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H1("Sleep, Lifestyle, and Daily Performance"),
            html.Div(id="input-form", children="Input form goes here."),
            html.Div(id="results", children="Risk score and charts go here."),
        ]
    )
    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
