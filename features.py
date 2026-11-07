"""Shared feature engineering for train_model.py and app.py."""

import numpy as np
import pandas as pd

BASE_FEATURES = [
    "HelpfulnessNumerator",
    "HelpfulnessDenominator",
    "Time",
    "Text_Length",
    "Summary_Length",
    "Word_Count",
    "Helpful_Ratio",
    "Sentiment_Polarity",
    "Sentiment_Subjectivity",
]

ENGINEERED_FEATURES = [
    "Has_Summary",
    "Has_Text",
    "Helpful_Votes",
    "Time_Age_Years",
]

NUMERIC_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES
SECONDS_PER_YEAR = 365.25 * 24 * 3600


def build_numeric(df, time_reference=None):
    """Return a DataFrame with NUMERIC_FEATURES columns.

    df must contain the BASE_FEATURES columns plus 'Summary' and 'Text'
    (raw text, may be NaN). time_reference defaults to df['Time'].max().
    """
    out = df[BASE_FEATURES].copy().astype(np.float32)

    summary = df["Summary"].fillna("").astype(str)
    text = df["Text"].fillna("").astype(str)
    out["Has_Summary"] = ((summary.str.len() > 0)).astype(np.float32)
    out["Has_Text"] = ((text.str.len() > 0)).astype(np.float32)
    out["Helpful_Votes"] = (df["HelpfulnessDenominator"] > 0).astype(np.float32)

    ref = df["Time"].max() if time_reference is None else time_reference
    out["Time_Age_Years"] = ((ref - df["Time"]) / SECONDS_PER_YEAR).astype(np.float32)

    return out[NUMERIC_FEATURES]


def combine_text(df):
    """Single text column for TF-IDF: summary + review body."""
    return (df["Summary"].fillna("").astype(str) + " " +
            df["Text"].fillna("").astype(str))
