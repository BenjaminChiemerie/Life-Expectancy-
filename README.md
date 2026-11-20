# Life Expectancy Predictor

A [Streamlit](https://streamlit.io) web app for estimating **life expectancy at birth** from health, economic and social indicators, powered by a pre-trained scikit-learn pipeline (median imputer + XGBoost regressor).

## Features

- **Single prediction**: enter indicators in the sidebar and get a predicted life expectancy, a category (Very low → Very high) and the difference from an all-typical baseline. Includes presets, the exact model features used, and global feature importances.
- **What-if explorer**: vary one input across a range, holding the others fixed, and plot the effect on the prediction.
- **Batch predictions**: upload a CSV, get predictions for every row, and download the results. A CSV template is built into the app.
- **Missing values handled**: tick **n/a** on any field and the model's imputer fills in the training median.

## Project structure

```
.
├── app.py                         # Streamlit application
├── life_expectancy_model.joblib   # Trained sklearn Pipeline (imputer + XGBRegressor)
├── requirements.txt
└── README.md
```

## Getting started

### Prerequisites

- Python 3.9 – 3.13

### Installation

```bash
git clone <your-repo-url>
cd <your-repo-folder>

python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

The app opens at <http://localhost:8501>.

> **Note:** the model was saved with scikit-learn **1.6.1**, which is pinned in `requirements.txt`. Using a different version may produce warnings or subtly different results.

## Model

| | |
|---|---|
| Type | `sklearn.pipeline.Pipeline` |
| Steps | `SimpleImputer(strategy="median")` → `XGBRegressor` (300 trees, max depth 4, learning rate 0.05) |
| Target | Life expectancy (years) |
| Input | 15 engineered features |

### Model features

`year_idx`, `status_developed`, `adult_mortality`, `alcohol`, `percentage_expenditure_log`, `total_expenditure`, `hiv_aids_log`, `gdp_log`, `population_log`, `infant_deaths_log`, `measles_log`, `immunization_avg`, `thinness_avg`, `income_composition_of_resources`, `schooling`

### How UI inputs are converted

The app accepts human-readable values and converts them in `to_features()`:

| Model feature | Derived from |
|---|---|
| `year_idx` | `year − 2000` |
| `status_developed` | `1` if Developed, else `0` |
| `*_log` features | `log(1 + value)` of the raw value |
| `immunization_avg` | mean of Hepatitis B, Polio and Diphtheria coverage |
| `thinness_avg` | mean of thinness (ages 10–19) and thinness (ages 5–9) |

> These transformations were inferred from the feature names. If your training pipeline differed, edit `to_features()` in `app.py`.

## Batch CSV format

Download the template from the **Batch (CSV)** tab, or use these raw columns (any missing column or cell is imputed):

`status`, `year`, `adult_mortality`, `infant_deaths`, `hiv_aids`, `measles`, `thinness_1_19`, `thinness_5_9`, `hepatitis_b`, `polio`, `diphtheria`, `gdp`, `population`, `percentage_expenditure`, `total_expenditure`, `alcohol`, `income_composition`, `schooling`

`status` is `Developed` or `Developing`. Alternatively, upload a CSV that already contains the 15 model features listed above.

The output includes all of your original columns plus `predicted_life_expectancy`.

## Limitations

- The prediction is a **point estimate** with no uncertainty interval.
- The model appears to be trained on country-level data from roughly 2000–2015; it is not meant for individual-level predictions, and inputs far outside the training range are unreliable.
- Tree-based models produce step-like responses, so small input changes can have no effect or a sudden one.

## Tech stack

[Streamlit](https://streamlit.io) · [scikit-learn](https://scikit-learn.org) · [XGBoost](https://xgboost.ai) · [pandas](https://pandas.pydata.org) · [joblib](https://joblib.readthedocs.io)

## License

Add a license of your choice (e.g. MIT) before publishing.
