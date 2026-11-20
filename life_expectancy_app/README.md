# Life Expectancy Predictor (Streamlit)

Streamlit UI for `life_expectancy_model.joblib` (sklearn Pipeline: median imputer + XGBRegressor).

## Run
```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
streamlit run app.py
```
Keep `life_expectancy_model.joblib` in the same folder as `app.py`.

## Features
- **Single prediction** – sidebar inputs, presets, "n/a" per field (imputed with the training median), feature table and importances.
- **What-if explorer** – sweep one input and plot predicted life expectancy.
- **Batch (CSV)** – upload raw-column or engineered-feature CSV; download predictions. A template is provided in the app.

## Assumptions
Feature engineering is inferred from the model's feature names (`year_idx = year - 2000`, `log1p` for `*_log`,
mean of hep B/polio/diphtheria for `immunization_avg`, mean of the two thinness columns for `thinness_avg`).
If your training code differed, edit `to_features()` in `app.py`.
