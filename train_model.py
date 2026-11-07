"""Train baseline Logistic Regression and challenger LightGBM models to predict
Amazon review 'Score' (1-5) from cleaned_features.parquet.

Features: engineered numeric features + TF-IDF/SVD of Summary+Text.
Class imbalance handled via inverse-frequency sample weights.

Exports (atomic writes):
    preprocessor.pkl    - scaler, TF-IDF vectorizer, SVD, feature metadata
    lr_model.pkl        - logistic regression bundle
    lgbm_model.pkl      - LightGBM bundle
    best_lgb_model.pkl  - winning model bundle
    metrics.json        - holdout metrics for both models + majority baseline
"""

import json
import os
import pickle
import tempfile

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from features import NUMERIC_FEATURES, build_numeric, combine_text

DATA_PATH = os.path.join("data", "processed", "cleaned_features.parquet")
MODELS_DIR = "models"
RANDOM_STATE = 42
N_SVD_COMPONENTS = 150


def atomic_pickle_dump(obj, path):
    """Write pickle to a temp file in the same dir, then os.replace for atomicity."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)  # atomic on same filesystem
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def evaluate(model, X_test, y_test, le, name):
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    y_pred_labels = le.inverse_transform(y_pred)
    y_true_labels = le.inverse_transform(y_test)
    return {
        "model": name,
        "n_features": int(X_test.shape[1]),
        "accuracy": float(accuracy_score(y_true_labels, y_pred_labels)),
        "f1_weighted": float(f1_score(y_true_labels, y_pred_labels, average="weighted")),
        "log_loss": float(log_loss(y_test, proba, labels=np.arange(len(le.classes_)))),
    }


def main():
    df = pd.read_parquet(DATA_PATH)
    df = df.dropna(subset=["Score"])

    le = LabelEncoder()
    y = le.fit_transform(df["Score"].astype(int))  # 0..4 for LightGBM

    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    idx_tr, idx_val = train_test_split(
        idx_train, test_size=0.15, stratify=y[idx_train], random_state=RANDOM_STATE
    )

    df_train_text = df.iloc[idx_tr]
    time_reference = float(df["Time"].max())

    # --- Preprocessing: numeric scaler + TF-IDF/SVD on text (fit on train only) ---
    X_num_all = build_numeric(df, time_reference=time_reference)
    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(X_num_all.iloc[idx_tr])

    tfidf = TfidfVectorizer(
        max_features=20000, ngram_range=(1, 2), min_df=5,
        sublinear_tf=True, strip_accents="unicode",
    )
    svd = TruncatedSVD(n_components=N_SVD_COMPONENTS, algorithm="randomized",
                       random_state=RANDOM_STATE)
    X_text_train = svd.fit_transform(tfidf.fit_transform(combine_text(df_train_text)))

    def make_X(idx):
        num = scaler.transform(X_num_all.iloc[idx])
        txt = svd.transform(tfidf.transform(combine_text(df.iloc[idx])))
        return np.hstack([num, txt]).astype(np.float32)

    X_train, X_val, X_test = make_X(idx_tr), make_X(idx_val), make_X(idx_test)
    y_train, y_val, y_test = y[idx_tr], y[idx_val], y[idx_test]
    print(f"Feature matrix: {X_train.shape[1]} cols "
          f"({len(NUMERIC_FEATURES)} numeric + {N_SVD_COMPONENTS} SVD)")

    # Inverse-frequency class weights for imbalance
    counts = np.bincount(y_train, minlength=len(le.classes_))
    weights = len(y_train) / (len(le.classes_) * counts)
    sample_weight = weights[y_train]

    # --- Baseline: Logistic Regression ---
    lr = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    lr.fit(X_train, y_train, sample_weight=sample_weight)

    # --- Challenger: LightGBM multiclass with early stopping ---
    lgbm = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(le.classes_),
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    # --- Evaluate on held-out test set ---
    lr_metrics = evaluate(lr, X_test, y_test, le, "logistic_regression")
    lgb_metrics = evaluate(lgbm, X_test, y_test, le, "lightgbm")

    majority = float(np.bincount(y_test, minlength=len(le.classes_)).max() / len(y_test))
    metrics = {
        "lr": lr_metrics,
        "lgbm": lgb_metrics,
        "majority_baseline_accuracy": majority,
        "class_distribution_test": {
            str(int(c)): int(n) for c, n in zip(le.classes_, np.bincount(y_test))
        },
    }
    print(json.dumps(metrics, indent=2))

    # Pick winner by weighted F1, tiebreak on log loss
    best_name = "lgbm" if (
        lgb_metrics["f1_weighted"] > lr_metrics["f1_weighted"]
        or (lgb_metrics["f1_weighted"] == lr_metrics["f1_weighted"]
            and lgb_metrics["log_loss"] < lr_metrics["log_loss"])
    ) else "lr"
    best_model, best_metrics = (lgbm, lgb_metrics) if best_name == "lgbm" else (lr, lr_metrics)
    print(f"Best model: {best_metrics['model']} (f1={best_metrics['f1_weighted']:.4f})")

    # Preprocessor bundle (shared by app.py)
    prep = {
        "scaler": scaler,
        "tfidf": tfidf,
        "svd": svd,
        "numeric_features": NUMERIC_FEATURES,
        "time_reference": time_reference,
        "n_svd_components": N_SVD_COMPONENTS,
    }
    # Keep SVD components float32 to reduce pickle size
    prep["svd"].components_ = prep["svd"].components_.astype(np.float32)

    atomic_pickle_dump(prep, os.path.join(MODELS_DIR, "preprocessor.pkl"))

    def model_bundle(model, name, key):
        return {"model": model, "model_name": name, "label_encoder": le,
                "metrics": metrics[key]}

    atomic_pickle_dump(model_bundle(lr, "logistic_regression", "lr"),
                       os.path.join(MODELS_DIR, "lr_model.pkl"))
    atomic_pickle_dump(model_bundle(lgbm, "lightgbm", "lgbm"),
                       os.path.join(MODELS_DIR, "lgbm_model.pkl"))
    atomic_pickle_dump(model_bundle(best_model, best_metrics["model"], best_name),
                       os.path.join(MODELS_DIR, "best_lgb_model.pkl"))

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Exported models/preprocessor.pkl, models/lr_model.pkl, "
          "models/lgbm_model.pkl, models/best_lgb_model.pkl, models/metrics.json")


if __name__ == "__main__":
    main()
