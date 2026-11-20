"""Feature definitions and preprocessing for the sleep disorder risk model.

Column names match the raw Kaggle CSV (sleep_health_dataset.csv).
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "sleep_disorder_risk"

# Ordered from least to most severe. The risk score is built from this order.
CLASS_ORDER = ["Healthy", "Mild", "Moderate", "Severe"]

# Things a person can reasonably report about themselves in the app form.
NUMERIC_FEATURES = [
    "age",
    "bmi",
    "sleep_duration_hrs",
    "sleep_latency_mins",
    "wake_episodes_per_night",
    "caffeine_mg_before_bed",
    "alcohol_units_before_bed",
    "screen_time_before_bed_mins",
    "exercise_day",
    "steps_that_day",
    "nap_duration_mins",
    "stress_score",
    "work_hours_that_day",
    "heart_rate_resting_bpm",
    "sleep_aid_used",
    "shift_work",
    "room_temperature_celsius",
    "weekend_sleep_diff_hrs",
]

CATEGORICAL_FEATURES = [
    "gender",
    "occupation",
    "chronotype",
    "mental_health_condition",
    "day_type",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns we deliberately leave out, and why.
EXCLUDED = {
    "person_id": "identifier",
    "sleep_quality_score": "outcome of sleep, not a lifestyle input (leaks target)",
    "cognitive_performance_score": "outcome of sleep, not a lifestyle input (leaks target)",
    "felt_rested": "outcome of sleep, not a lifestyle input",
    "rem_percentage": "needs a sleep tracker; users cannot self-report",
    "deep_sleep_percentage": "needs a sleep tracker; users cannot self-report",
    "country": "no relationship with the target in this data",
    "season": "no relationship with the target in this data",
}


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric columns and one-hot encode categorical columns."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
