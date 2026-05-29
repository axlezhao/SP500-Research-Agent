from __future__ import annotations

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import MODEL_DIR


NUMERIC_FEATURES = ["return_5d", "return_20d", "volatility_20d", "momentum_20d", "sentiment_20d", "news_count_20d", "market_cap", "pe_ratio", "revenue", "profit_margin", "debt_to_equity", "roe"]
CATEGORICAL_FEATURES = ["sector"]


def available_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    return [col for col in NUMERIC_FEATURES if col in df.columns], [col for col in CATEGORICAL_FEATURES if col in df.columns]


def train_return_direction_model(features: pd.DataFrame, max_rows: int = 200_000) -> tuple[Pipeline, str]:
    numeric, categorical = available_features(features)
    model_df = features.dropna(subset=["target_up_5d"]).copy()
    original_rows = len(model_df)
    if len(model_df) > max_rows:
        model_df = model_df.sample(n=max_rows, random_state=42)
    x = model_df[numeric + categorical]
    y = model_df["target_up_5d"].astype(int)
    stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=stratify)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ],
        remainder="drop",
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", RandomForestClassifier(n_estimators=120, min_samples_leaf=10, random_state=42, class_weight="balanced_subsample", n_jobs=-1))])
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    report = classification_report(y_test, predictions, zero_division=0)
    if y_test.nunique() == 2:
        report += f"\nROC AUC: {roc_auc_score(y_test, pipeline.predict_proba(x_test)[:, 1]):.3f}\n"
    if original_rows > len(model_df):
        report += f"\nTraining sample: {len(model_df):,} of {original_rows:,} rows.\n"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "numeric": numeric, "categorical": categorical}, MODEL_DIR / "return_direction_model.joblib")
    return pipeline, report


def load_model():
    return joblib.load(MODEL_DIR / "return_direction_model.joblib")


def score_latest(features: pd.DataFrame, model_bundle: dict) -> pd.DataFrame:
    columns = model_bundle["numeric"] + model_bundle["categorical"]
    latest = features.sort_values("date").groupby("ticker", as_index=False).tail(1).copy()
    latest["up_probability_5d"] = model_bundle["pipeline"].predict_proba(latest[columns])[:, 1]
    return latest.sort_values("up_probability_5d", ascending=False)

