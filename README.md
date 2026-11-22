# Sleep, Lifestyle, and Daily Performance

DSC 200 group project (UCSD, Fall 2026).

## Overview

We train a classification model to predict a person's risk of sleep disorder from lifestyle factors (stress, caffeine, work, physical activity, and others). An interactive Dash app lets a user enter their own information and see a predicted risk score, the factors driving it, and charts exploring how lifestyle relates to sleep disorder risk.

**Data:** [Sleep Health and Daily Performance Dataset](https://www.kaggle.com/datasets/mohankrishnathalla/sleep-health-and-daily-performance-dataset) (Kaggle, ~100,000 synthetic records, public domain).

**Disclaimer:** This is a class project built on synthetic data. It is not medical advice.

## Team

| Role | Owner | Owns |
|---|---|---|
| Data lead | Ethan Arnn | `src/data/` |
| Modeling lead | Ethan Hershman | `src/models/` |
| App / visualization lead | Tammy Nguyen | `src/app/` |
| Integration / spec lead | Darren Lam | repo, tests runner, spec |

Teammates: Ethan Arnn, Tammy Nguyen, Ethan Hershman.

## Setup

Requires Python 3.11.

```bash
git clone https://github.com/DarrenTheLamb01/dsc200-sleep-health
cd dsc200-sleep-health
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Get the data:** download the CSV from the Kaggle link above and put it in `data/`. The `data/` folder is gitignored, so do not commit the file.

## Run

```bash
pytest                        # run all tests
python -m src.app.app         # start the Dash app
```

## Project layout

```
README.md
requirements.txt
data/            # raw CSV goes here (gitignored)
src/
  data/          # loading, cleaning, splitting
  models/        # training, evaluation, explanations
  app/           # Dash app, charts, input form
tests/
  unit/
  functional/
notebooks/       # exploration only; real logic lives in src/
models/          # saved model artifacts
docs/            # spec drafts and diagrams
```

## Interfaces between roles

Code against these signatures so the pieces fit together. If one needs to change, tell the team first.

| Module | Function | Returns |
|---|---|---|
| `src/data` | `load_data(path)` | DataFrame |
| `src/data` | `clean_data(df)` | DataFrame |
| `src/data` | `split_data(df, target, seed)` | train, val, test DataFrames |
| `src/models` | `train(train_df, val_df)` | fitted model |
| `src/models` | `evaluate(model, df)` | dict of metrics |
| `src/models` | `predict_with_explanation(model, input_dict)` | `{risk_score, top_factors}` |

## Model

**Classifier:** `HistGradientBoostingClassifier` with balanced class weights, predicting `sleep_disorder_risk` (Healthy / Mild / Moderate / Severe). Three hyperparameter settings are tried and the best by validation macro-F1 is kept. It beat logistic regression (macro-F1 0.72) and random forest (0.85) on the same features.

**Features** (defined in `src/models/features.py`): only things a person can report about themselves, such as age, BMI, sleep duration, sleep latency, wake-ups, caffeine, alcohol, screen time, exercise, steps, naps, stress, work hours, resting heart rate, sleep aids, shift work, room temperature, weekend sleep difference, gender, occupation, chronotype, mental health condition and day type.

**Excluded columns:**

| Column | Reason |
|---|---|
| `sleep_quality_score`, `cognitive_performance_score`, `felt_rested` | Outcomes of sleep, not lifestyle inputs; they leak the target |
| `rem_percentage`, `deep_sleep_percentage` | Need a sleep tracker; users can't self-report them |
| `country`, `season` | No relationship with the target in this data |
| `person_id` | Identifier |

**Risk score:** expected severity across the four classes, scaled to [0, 1] (Healthy = 0, Mild = 1/3, Moderate = 2/3, Severe = 1).

**Top factors:** for each feature, the change in risk score when that feature is reset to a typical training value (median, or most common category). The five largest by size are returned; positive means the feature raises this person's risk. Fields left out of `input_dict` are filled with typical values, so a partial form still works.

**Train and save:**

```bash
python -m src.models.train    # writes models/sleep_risk_model.joblib
```

The model file is gitignored, so each person runs this once locally. The app loads it with `load_model()`. The script uses `src/data/pipeline.py` when it exists, and otherwise does its own 70/15/15 stratified split.

**Test set results** (15,000 held-out rows):

| Metric | Value |
|---|---|
| Macro-F1 | 0.867 |
| Accuracy | 0.943 |
| ROC AUC (one-vs-rest) | 0.991 |
| Recall: Healthy / Mild / Moderate / Severe | 0.996 / 0.907 / 0.791 / 0.831 |

## Workflow

- Make a branch named `<area>/<task>` (for example `data/cleaning`, `app/input-form`).
- Open a pull request into `main`. The integration lead reviews and merges.
- Commit small and often.
- Requirement IDs (FR-1, NFR-1) and test IDs (UT-1, FT-1) are defined in `docs/`.
