"""Interactive Streamlit app for the Amazon review Score predictor.

Run:  streamlit run app.py
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

from features import BASE_FEATURES, NUMERIC_FEATURES, build_numeric, combine_text

PREP_PATH = os.path.join("models", "preprocessor.pkl")
MODEL_PATHS = {
    "Best (auto)": os.path.join("models", "best_lgb_model.pkl"),
    "Logistic Regression": os.path.join("models", "lr_model.pkl"),
    "LightGBM": os.path.join("models", "lgbm_model.pkl"),
}
DATA_PATH = os.path.join("data", "processed", "cleaned_features.parquet")

st.set_page_config(page_title="Amazon Review Score Predictor", layout="wide")
st.title("Amazon Product Review — Score Predictor")


@st.cache_resource
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_sample_rows(n=500):
    df = pd.read_parquet(DATA_PATH)
    return df.sample(n=min(n, len(df)), random_state=42)


prep = load_pickle(PREP_PATH)
scaler, tfidf, svd = prep["scaler"], prep["tfidf"], prep["svd"]
time_reference = prep["time_reference"]

with st.sidebar:
    st.header("Model")
    choice = st.radio("Select model", list(MODEL_PATHS), index=0)
    bundle = load_pickle(MODEL_PATHS[choice])
    model, le = bundle["model"], bundle["label_encoder"]
    st.caption(f"Active: **{bundle.get('model_name', 'best_lgb_model')}**")
    if "metrics" in bundle:
        st.caption(f"Holdout F1: {bundle['metrics']['f1_weighted']:.3f} | "
                   f"Acc: {bundle['metrics']['accuracy']:.3f}")

    st.header("Input mode")
    mode = st.radio("Provide features via", ["Manual inputs", "Sample rows"])


def make_features(num_vals, summary, text):
    """Build the model input vector from raw values."""
    row = {f: float(num_vals.get(f, 0.0)) for f in BASE_FEATURES}
    row["Summary"] = summary
    row["Text"] = text
    df1 = pd.DataFrame([row])

    X_num = scaler.transform(build_numeric(df1, time_reference=time_reference))
    X_txt = svd.transform(tfidf.transform(combine_text(df1)))
    return np.hstack([X_num, X_txt]).astype(np.float32)


# Numeric input specs: (min, max, default) — all floats for Streamlit consistency
SPECS = {
    "HelpfulnessNumerator": (0.0, 500.0, 0.0),
    "HelpfulnessDenominator": (0.0, 500.0, 0.0),
    "Time": (0.0, float(time_reference), float(time_reference)),
    "Text_Length": (0.0, 10000.0, 200.0),
    "Summary_Length": (0.0, 2000.0, 50.0),
    "Word_Count": (0.0, 2000.0, 30.0),
    "Helpful_Ratio": (0.0, 1.0, 0.5),
    "Sentiment_Polarity": (-1.0, 1.0, 0.0),
    "Sentiment_Subjectivity": (0.0, 1.0, 0.5),
}
SLIDERS = {"Helpful_Ratio", "Sentiment_Polarity", "Sentiment_Subjectivity"}


def numeric_inputs(defaults=None, key_suffix=""):
    defaults = defaults or {}
    cols = st.columns(3)
    vals = {}
    for i, feat in enumerate(BASE_FEATURES):
        lo, hi, dflt = SPECS[feat]
        default = float(defaults.get(feat, dflt))
        default = min(max(default, lo), hi)
        key = f"{feat}_{key_suffix}" if key_suffix else None
        with cols[i % 3]:
            if feat in SLIDERS:
                vals[feat] = st.slider(feat, lo, hi, default, 0.01, key=key)
            else:
                vals[feat] = st.number_input(
                    feat, min_value=float(lo), max_value=float(hi),
                    value=float(default), step=1.0, format="%.0f", key=key)
    return vals


summary_text, text_body = "", ""
if mode == "Sample rows":
    sample = load_sample_rows(500)
    idx = st.slider("Sample index", 0, len(sample) - 1, 0)
    row = sample.iloc[idx]
    st.write("**Actual Score:**", int(row["Score"]))
    s = str(row["Summary"]) if pd.notna(row["Summary"]) else ""
    t = str(row["Text"]) if pd.notna(row["Text"]) else ""
    summary_text = st.text_area("Summary", value=s, height=60,
                                key=f"sum_{idx}")
    text_body = st.text_area("Review text", value=t, height=150,
                             key=f"txt_{idx}")
    num_vals = numeric_inputs(defaults={f: float(row[f]) for f in BASE_FEATURES},
                              key_suffix=f"s{idx}")
else:
    num_vals = numeric_inputs()
    summary_text = st.text_area("Summary", height=60)
    text_body = st.text_area("Review text", height=120)

if st.button("Predict", type="primary"):
    X = make_features(num_vals, summary_text, text_body)
    proba = model.predict_proba(X)[0]
    pred_label = int(le.inverse_transform(model.predict(X))[0])

    st.subheader(f"Predicted Score: {pred_label} / 5")
    st.metric("Confidence", f"{proba.max() * 100:.1f}%")

    labels = [f"{s} star" + ("s" if s != 1 else "")
              for s in le.inverse_transform(range(len(le.classes_)))]
    st.bar_chart(pd.DataFrame({"Probability": proba}, index=labels))

with st.expander("Holdout metrics"):
    try:
        import json
        with open(os.path.join("models", "metrics.json")) as f:
            st.json(json.load(f))
    except FileNotFoundError:
        st.write("metrics.json not found — run train_model.py first.")
