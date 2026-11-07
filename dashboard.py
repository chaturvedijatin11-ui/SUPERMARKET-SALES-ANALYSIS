"""EDA dashboard for the Amazon Product Reviews dataset.

Run:  streamlit run dashboard.py --server.port 8502
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

DATA_PATH = os.path.join("data", "processed", "cleaned_features.parquet")

st.set_page_config(page_title="Reviews Dataset Dashboard", layout="wide")
st.title("Amazon Product Reviews — Dataset Dashboard")


@st.cache_data
def load_data():
    return pd.read_parquet(DATA_PATH)


df = load_data()

# ---------------------------------------------------------------- overview
tab_overview, tab_target, tab_text, tab_sentiment, tab_activity = st.tabs([
    "Overview", "Target (Score)", "Text Features", "Sentiment", "Activity & Entities",
])

with tab_overview:
    st.subheader("Dataset profile")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    c4.metric("Score range", f"{df['Score'].min()}–{df['Score'].max()}")

    st.subheader("Columns")
    st.dataframe(
        pd.DataFrame({
            "column": df.columns,
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "null_pct": (df.isna().mean() * 100).round(2),
        }),
        use_container_width=True, hide_index=True,
    )

    st.subheader("Numeric summary")
    st.dataframe(df.describe().T, use_container_width=True)

with tab_target:
    st.subheader("Score distribution")
    counts = df["Score"].value_counts().sort_index()
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(counts.rename("Reviews"))
    with col2:
        st.dataframe(pd.DataFrame({
            "Score": counts.index,
            "Reviews": counts.values,
            "Share %": (counts.values / len(df) * 100).round(2),
        }), use_container_width=True, hide_index=True)
    st.caption("The dataset is heavily imbalanced toward 5-star reviews — "
               "models are trained with class weights to compensate.")

with tab_text:
    st.subheader("Review text statistics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg review length", f"{df['Text_Length'].mean():.0f} chars")
    c2.metric("Avg word count", f"{df['Word_Count'].mean():.0f} words")
    c3.metric("Avg helpfulness ratio", f"{df['Helpful_Ratio'].mean():.2f}")

    st.subheader("Feature distributions by score")
    feats = st.multiselect("Features to compare", [
        "Text_Length", "Word_Count", "Helpful_Ratio",
        "Sentiment_Polarity", "Sentiment_Subjectivity",
    ], default=["Text_Length", "Helpful_Ratio", "Sentiment_Polarity"])
    for feat in feats:
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.boxplot(data=df, x="Score", y=feat, ax=ax, palette="viridis")
        ax.set_title(f"{feat} by Score")
        st.pyplot(fig)
        plt.close(fig)

with tab_sentiment:
    st.subheader("Sentiment vs. rating")
    st.write("TextBlob polarity/subjectivity mined from review text "
             "(see `eda_and_preprocess.py`).")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(data=df, x="Score", y="Sentiment_Polarity", ax=axes[0],
                palette="viridis")
    axes[0].set_title("Polarity by Score")
    sns.boxplot(data=df, x="Score", y="Sentiment_Subjectivity", ax=axes[1],
                palette="viridis")
    axes[1].set_title("Subjectivity by Score")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Sample: polarity vs. subjectivity (colored by score)")
    sample = df.sample(min(5000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(sample["Sentiment_Polarity"],
                         sample["Sentiment_Subjectivity"],
                         c=sample["Score"], cmap="viridis", s=8, alpha=0.5)
    plt.colorbar(scatter, ax=ax, label="Score")
    ax.set_xlabel("Polarity"); ax.set_ylabel("Subjectivity")
    st.pyplot(fig)
    plt.close(fig)

with tab_activity:
    st.subheader("Review activity over time")
    df_year = df.copy()
    df_year["Year"] = pd.to_datetime(df_year["Time"], unit="s").dt.year
    per_year = df_year.groupby("Year").agg(
        reviews=("Score", "size"), avg_score=("Score", "mean"),
    ).reset_index()
    c1, c2 = st.columns(2)
    with c1:
        st.bar_chart(per_year.set_index("Year")["reviews"].rename("Reviews"))
    with c2:
        st.line_chart(per_year.set_index("Year")["avg_score"].rename("Avg score"))

    st.subheader("Top products by review count")
    top_products = (df.groupby("ProductId").size()
                    .sort_values(ascending=False).head(15))
    st.bar_chart(top_products.rename("Reviews"))

    st.subheader("Most active reviewers")
    top_users = (df.groupby("ProfileName").size()
                 .sort_values(ascending=False).head(15))
    st.bar_chart(top_users.rename("Reviews"))

    st.subheader("Helpfulness votes")
    voted = df[df["HelpfulnessDenominator"] > 0]
    c3, c4 = st.columns(2)
    c3.metric("Reviews with votes", f"{len(voted):,}")
    c4.metric("Total helpful votes", f"{int(df['HelpfulnessNumerator'].sum()):,}")
