# Sleep, Lifestyle, and Daily Performance

DSC 200 group project (UCSD, Fall 2026).

## Overview

We train a classification model to predict a person's risk of sleep disorder from lifestyle factors (stress, caffeine, work, physical activity, and others). An interactive Dash app lets a user enter their own information and see a predicted risk score, the factors driving it, and charts exploring how lifestyle relates to sleep disorder risk.

**Data:** [Sleep Health and Daily Performance Dataset](https://www.kaggle.com/datasets/mohankrishnathalla/sleep-health-and-daily-performance-dataset) (Kaggle, ~100,000 synthetic records, public domain).

**Disclaimer:** This is a class project built on synthetic data. It is not medical advice.

## Team

| Role | Owner | Owns |
|---|---|---|
| Data lead | _TBD_ | `src/data/` |
| Modeling lead | _TBD_ | `src/models/` |
| App / visualization lead | _TBD_ | `src/app/` |
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

## Workflow

- Never push directly to `main`.
- Make a branch named `<area>/<task>` (for example `data/cleaning`, `app/input-form`).
- Open a pull request into `main`. The integration lead reviews and merges.
- Commit small and often.
- Requirement IDs (FR-1, NFR-1) and test IDs (UT-1, FT-1) are defined in `docs/`.
