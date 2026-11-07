# Amazon Product Review — Score Predictor

Company-grade data science project built on the **Amazon Product Reviews** dataset
(~500k reviews; we use a 50k stratified sample). It predicts the star `Score` (1–5)
a reviewer gave, using review text, helpfulness votes, review length, and
sentiment features — and ships with a trained model, an interactive predictor,
and an EDA dashboard.

## Repository layout

```
DS_Project/
├── app.py                      # Streamlit inference app (Score predictor)
├── dashboard.py                # Streamlit EDA dashboard for the dataset
├── train_model.py              # Trains LR baseline + LightGBM challenger
├── features.py                 # Shared feature engineering (train + app)
├── eda_and_preprocess.py       # Phase 1: raw CSV → cleaned parquet
│
├── data/
│   ├── raw/Reviews.csv                    # Original Kaggle dataset (~300 MB)
│   └── processed/cleaned_features.parquet # 50k rows × 16 cols, engineered
│
├── models/
│   ├── preprocessor.pkl       # scaler + TF-IDF + SVD (fit on train split only)
│   ├── lr_model.pkl            # Logistic Regression bundle
│   ├── lgbm_model.pkl          # LightGBM bundle
│   ├── best_lgb_model.pkl      # Winning model (currently LightGBM)
│   └── metrics.json            # Holdout metrics for all models
│
├── assets/target_distribution.png
├── docs/STREAMLIT_APP.md       # Field-by-field guide to app.py
├── requirements.txt
└── README.md
```

## Pipeline overview

```
data/raw/Reviews.csv
      │  eda_and_preprocess.py   (chunked load → 50k sample → clean →
      │                          engineer lengths/helpfulness/sentiment)
      ▼
data/processed/cleaned_features.parquet
      │  train_model.py          (stratified 80/10/10 split)
      │    numeric features ──► StandardScaler
      │    Summary+Text   ──► TF-IDF (20k, 1–2 grams) ──► SVD (150 comps)
      │    → 163-dim feature matrix, inverse-frequency class weights
      ▼
models/*.pkl + metrics.json
      │  app.py / dashboard.py   (Streamlit)
      ▼
Interactive prediction + dataset exploration
```

## Models

| Model | Role | Holdout accuracy | Holdout F1 (weighted) | Log loss |
|---|---|---|---|---|
| Majority class | Baseline floor | 0.642 | — | — |
| Logistic Regression | Baseline | 0.630 | 0.658 | 0.975 |
| **LightGBM** | **Challenger → winner** | **0.730** | **0.712** | **0.725** |

- Task: 5-class classification (`Score` ∈ {1,…,5}), heavily imbalanced
  (~64% are 5★) → handled with inverse-frequency `sample_weight`.
- LightGBM uses early stopping on a validation split (`multi_logloss`).
- Winner is selected by weighted F1 (log-loss tiebreak) and exported as
  `models/best_lgb_model.pkl` via atomic writes (temp file + `os.replace`),
  so a crash mid-save can't corrupt the artifact.

## Quick start

```bash
# 1. Environment (already set up in .venv)
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Rebuild data (only if raw CSV changes)
python eda_and_preprocess.py

# 3. Retrain models (~5 min)
python train_model.py

# 4. Launch apps
streamlit run app.py --server.port 8501        # predictor
streamlit run dashboard.py --server.port 8502  # dataset dashboard
```

## Key files explained

- **`features.py`** — single source of truth for features. `build_numeric()`
  produces the 13 numeric columns (9 base + `Has_Summary`, `Has_Text`,
  `Helpful_Votes`, `Time_Age_Years`); `combine_text()` merges Summary+Text for
  TF-IDF. Used identically by training and inference — no train/serve skew.
- **`train_model.py`** — trains both models, evaluates on a untouched test set,
  writes all artifacts under `models/`.
- **`app.py`** — inference UI; see [`docs/STREAMLIT_APP.md`](docs/STREAMLIT_APP.md)
  for a field-by-field walkthrough.
- **`dashboard.py`** — read-only EDA dashboard: dataset profile, score
  distribution, text/sentiment analysis, review activity, top products/users.

## Notes & limitations

- The 50k sample is a random draw from the full ~500k-review CSV; metrics are
  on a held-out 20% of that sample.
- Text features are TF-IDF→SVD (linear), not a transformer — fast and
  dependency-light, but a fine-tuned model would likely score higher.
- `Ecommerce_Rating.py` is an empty placeholder.
