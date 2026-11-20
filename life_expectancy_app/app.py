"""Life Expectancy Predictor - Streamlit UI for life_expectancy_model.joblib

Run:  streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "life_expectancy_model.joblib"

# Exact feature order the trained pipeline expects.
FEATURES = [
    "year_idx", "status_developed", "adult_mortality", "alcohol",
    "percentage_expenditure_log", "total_expenditure", "hiv_aids_log",
    "gdp_log", "population_log", "infant_deaths_log", "measles_log",
    "immunization_avg", "thinness_avg", "income_composition_of_resources",
    "schooling",
]

# ---------------------------------------------------------------------------
# Input definitions (human-friendly raw values; transformed in to_features()).
# key: (label, min, max, default, step, help)
# Defaults are roughly the training-data medians used by the model's imputer.
# ---------------------------------------------------------------------------
GROUPS = {
    "Country & year": [
        ("year", "Year", 2000, 2030, 2015, 1, "Calendar year (model was trained on 2000-2015)."),
    ],
    "Health & mortality": [
        ("adult_mortality", "Adult mortality", 0.0, 800.0, 139.0, 1.0,
         "Deaths per 1,000 adults aged 15-60."),
        ("infant_deaths", "Infant deaths", 0.0, 2000.0, 3.0, 1.0,
         "Infant deaths per 1,000 population."),
        ("hiv_aids", "HIV/AIDS deaths", 0.0, 60.0, 0.1, 0.1,
         "Deaths per 1,000 live births (ages 0-4)."),
        ("measles", "Measles cases", 0.0, 250000.0, 19.0, 10.0,
         "Reported cases per 1,000 population."),
        ("thinness_1_19", "Thinness, ages 10-19 (%)", 0.0, 30.0, 3.3, 0.1,
         "Prevalence of thinness among ages 10-19."),
        ("thinness_5_9", "Thinness, ages 5-9 (%)", 0.0, 30.0, 3.3, 0.1,
         "Prevalence of thinness among ages 5-9."),
    ],
    "Immunization (% coverage)": [
        ("hepatitis_b", "Hepatitis B", 0.0, 100.0, 92.0, 1.0, "Coverage among 1-year-olds."),
        ("polio", "Polio", 0.0, 100.0, 93.0, 1.0, "Coverage among 1-year-olds."),
        ("diphtheria", "Diphtheria", 0.0, 100.0, 92.0, 1.0, "DTP3 coverage among 1-year-olds."),
    ],
    "Economy & spending": [
        ("gdp", "GDP per capita (USD)", 0.0, 120000.0, 1815.0, 100.0, "Gross domestic product per capita."),
        ("population", "Population", 0.0, 1_500_000_000.0, 1_400_000.0, 100_000.0, "Total population."),
        ("percentage_expenditure", "Health expenditure (% of GDP per capita)", 0.0, 20000.0, 79.0, 10.0,
         "Expenditure on health as a percentage of GDP per capita."),
        ("total_expenditure", "Total health spending (% of gov. spending)", 0.0, 20.0, 5.7, 0.1,
         "General government expenditure on health as % of total government expenditure."),
    ],
    "Lifestyle & development": [
        ("alcohol", "Alcohol (litres per capita)", 0.0, 20.0, 3.8, 0.1, "Recorded consumption, ages 15+."),
        ("income_composition", "Income composition of resources", 0.0, 1.0, 0.70, 0.01,
         "Human Development Index in terms of income composition (0-1)."),
        ("schooling", "Schooling (years)", 0.0, 22.0, 12.5, 0.1, "Average years of schooling."),
    ],
}
FIELDS = {f[0]: f for fields in GROUPS.values() for f in fields}

PRESETS = {
    "Typical developing country": dict(
        status="Developing", year=2015, adult_mortality=190.0, infant_deaths=15.0, hiv_aids=0.5,
        measles=200.0, thinness_1_19=5.5, thinness_5_9=5.5, hepatitis_b=80.0, polio=85.0,
        diphtheria=85.0, gdp=1500.0, population=12_000_000.0, percentage_expenditure=40.0,
        total_expenditure=5.0, alcohol=3.0, income_composition=0.60, schooling=11.0),
    "Typical developed country": dict(
        status="Developed", year=2015, adult_mortality=70.0, infant_deaths=1.0, hiv_aids=0.1,
        measles=0.0, thinness_1_19=1.5, thinness_5_9=1.5, hepatitis_b=90.0, polio=95.0,
        diphtheria=95.0, gdp=35000.0, population=10_000_000.0, percentage_expenditure=3000.0,
        total_expenditure=8.5, alcohol=10.0, income_composition=0.88, schooling=15.5),
    "Dataset medians": dict(
        status="Developing", **{k: v[4] for k, v in FIELDS.items()}),
}


# ---------------------------------------------------------------------------
# Model + feature engineering
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load(MODEL_PATH)


def to_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert raw, human-readable columns to the model's engineered features.

    Missing values stay NaN; the pipeline's median imputer fills them in.
    """
    r = raw.copy()
    for col in list(FIELDS) + ["status"]:
        if col not in r:
            r[col] = np.nan
    num = lambda c: pd.to_numeric(r[c], errors="coerce")
    log = lambda c: np.log1p(num(c).clip(lower=0))

    out = pd.DataFrame(index=r.index)
    out["year_idx"] = num("year") - 2000
    status = r["status"].astype("string").str.strip().str.lower()
    out["status_developed"] = status.map({"developed": 1.0, "developing": 0.0}).astype(float)
    out["adult_mortality"] = num("adult_mortality")
    out["alcohol"] = num("alcohol")
    out["percentage_expenditure_log"] = log("percentage_expenditure")
    out["total_expenditure"] = num("total_expenditure")
    out["hiv_aids_log"] = log("hiv_aids")
    out["gdp_log"] = log("gdp")
    out["population_log"] = log("population")
    out["infant_deaths_log"] = log("infant_deaths")
    out["measles_log"] = log("measles")
    out["immunization_avg"] = pd.concat(
        [num("hepatitis_b"), num("polio"), num("diphtheria")], axis=1).mean(axis=1)
    out["thinness_avg"] = pd.concat(
        [num("thinness_1_19"), num("thinness_5_9")], axis=1).mean(axis=1)
    out["income_composition_of_resources"] = num("income_composition")
    out["schooling"] = num("schooling")
    return out[FEATURES].astype(float)


def predict(model, raw: pd.DataFrame) -> np.ndarray:
    return model.predict(to_features(raw))


def band(years: float) -> str:
    if years < 55:
        return "Very low"
    if years < 65:
        return "Low"
    if years < 75:
        return "Moderate"
    if years < 82:
        return "High"
    return "Very high"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Life Expectancy Predictor", page_icon="🌍", layout="wide")

try:
    model = load_model()
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load model at `{MODEL_PATH}`.\n\n{e}")
    st.stop()


def apply_preset(name: str):
    p = PRESETS[name]
    st.session_state["status"] = p["status"]
    for k in FIELDS:
        st.session_state[k] = type(FIELDS[k][4])(p[k])
        st.session_state[f"{k}_na"] = False


# Initialise widget state once.
if "status" not in st.session_state:
    apply_preset("Dataset medians")

st.title("🌍 Life Expectancy Predictor")
st.caption("Estimate life expectancy at birth (years) from health, economic and social indicators.")

tab_single, tab_whatif, tab_batch, tab_about = st.tabs(
    ["Single prediction", "What-if explorer", "Batch (CSV)", "About"])

# -------------------------- Sidebar: inputs --------------------------------
with st.sidebar:
    st.header("Inputs")
    preset = st.selectbox("Load a preset", list(PRESETS), index=None, placeholder="Choose...", key="preset")
    if preset:
        st.button("Apply preset", on_click=apply_preset, args=(preset,), width="stretch")

    st.radio("Development status", ["Developing", "Developed"], key="status", horizontal=True)

    for group, fields in GROUPS.items():
        with st.expander(group, expanded=group in ("Health & mortality", "Lifestyle & development")):
            for key, label, lo, hi, default, step, help_ in fields:
                cols = st.columns([4, 1.2])
                is_int = isinstance(default, int)
                cols[0].number_input(
                    label, min_value=type(default)(lo), max_value=type(default)(hi),
                    step=type(default)(step), key=key, help=help_,
                    disabled=st.session_state.get(f"{key}_na", False),
                    format="%d" if is_int else None)
                cols[1].checkbox("n/a", key=f"{key}_na",
                                 help="Unknown - the model fills in the typical (median) value.")

    st.caption("Tick **n/a** for any value you don't know; it's imputed with the training median.")


def current_raw() -> dict:
    raw = {"status": st.session_state["status"]}
    for k in FIELDS:
        raw[k] = np.nan if st.session_state.get(f"{k}_na") else st.session_state[k]
    return raw


raw_now = current_raw()
raw_df = pd.DataFrame([raw_now])

# ----------------------------- Single prediction ---------------------------
with tab_single:
    pred = float(predict(model, raw_df)[0])
    baseline = float(predict(model, pd.DataFrame([{"status": "Developing"}]))[0])

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted life expectancy", f"{pred:.1f} years")
    c2.metric("Category", band(pred))
    c3.metric("vs. all-typical inputs", f"{pred - baseline:+.1f} years",
              help=f"Baseline = model prediction when every input is unknown ({baseline:.1f} years).")

    st.progress(min(max((pred - 40) / 50, 0.0), 1.0), text="Scale: 40 → 90 years")

    left, right = st.columns(2)
    with left:
        st.subheader("Model features used")
        feats = to_features(raw_df).T.rename(columns={0: "value"})
        feats["value"] = feats["value"].map(lambda v: "imputed (median)" if pd.isna(v) else f"{v:.3f}")
        st.dataframe(feats, width="stretch")
    with right:
        st.subheader("Feature importance (global)")
        imp = pd.Series(model[-1].feature_importances_, index=FEATURES).sort_values()
        st.bar_chart(imp, horizontal=True)

# ----------------------------- What-if explorer ----------------------------
with tab_whatif:
    st.write("Vary one input across its range while holding the others at the values in the sidebar.")
    key = st.selectbox("Input to vary", list(FIELDS), format_func=lambda k: FIELDS[k][1], index=1)
    _, label, lo, hi, default, step, _ = FIELDS[key]
    lo_v, hi_v = st.slider(f"Range for {label}", float(lo), float(hi), (float(lo), float(hi)))
    xs = np.linspace(lo_v, hi_v, 60)
    sweep = pd.DataFrame([{**raw_now, key: x} for x in xs])
    ys = predict(model, sweep)
    chart = pd.DataFrame({label: xs, "Predicted life expectancy (years)": ys}).set_index(label)
    st.line_chart(chart)
    st.caption("Tree models give step-like curves, and values outside the training range are unreliable.")

# ----------------------------- Batch ---------------------------------------
with tab_batch:
    st.write(
        "Upload a CSV to predict many rows at once. Use the raw column names below "
        "(missing columns/cells are imputed), or supply the model's 15 engineered features directly.")
    template_cols = ["status"] + list(FIELDS)
    template = pd.DataFrame([PRESETS["Typical developing country"], PRESETS["Typical developed country"]])[template_cols]
    st.download_button("Download CSV template", template.to_csv(index=False),
                       "life_expectancy_template.csv", "text/csv")

    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            df = pd.read_csv(up)
            if set(FEATURES).issubset(df.columns):
                X = df[FEATURES].astype(float)
            else:
                known = set(template_cols)
                if not known & set(df.columns):
                    raise ValueError("No recognised columns found. Download the template for the expected names.")
                X = to_features(df)
            result = df.copy()
            result["predicted_life_expectancy"] = model.predict(X).round(2)
            st.success(f"Predicted {len(result):,} rows.")
            st.dataframe(result, width="stretch")
            st.download_button("Download predictions", result.to_csv(index=False),
                               "life_expectancy_predictions.csv", "text/csv")
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not process file: {e}")

# ----------------------------- About ---------------------------------------
with tab_about:
    st.markdown(
        """
**Model:** scikit-learn `Pipeline` → median `SimpleImputer` → `XGBRegressor` (300 trees, depth 4).

**How inputs are converted** (assumed from the feature names; adjust `to_features()` if your
training code differed):

| Model feature | From the UI |
|---|---|
| `year_idx` | year − 2000 |
| `status_developed` | 1 if Developed, else 0 |
| `*_log` features | `log(1 + value)` of the raw value |
| `immunization_avg` | mean of Hepatitis B, Polio, Diphtheria |
| `thinness_avg` | mean of thinness 10-19 and 5-9 |

**Caveats:** the output is a point estimate with no uncertainty interval, and the model was trained on
country-level data from 2000-2015, so individual-level or out-of-range use is not meaningful.
        """
    )
