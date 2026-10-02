from __future__ import annotations

from pathlib import Path
from typing import Callable

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import MODEL_PATH


# Only features that were known on each row's date. Snapshot fundamentals are excluded
# because they come from the download date and would leak future information into history.
NUMERIC_FEATURES = [
    "return_5d",
    "return_20d",
    "momentum_60d",
    "volatility_20d",
    "ma_gap_50d",
    "volume_ratio_20d",
    "sentiment_20d",
    "news_count_20d",
    "return_20d_xs_rank",
    "momentum_60d_xs_rank",
    "volatility_20d_xs_rank",
    "return_20d_vs_sector",
]
CATEGORICAL_FEATURES = ["sector"]
TARGET = "target_up_5d"
# Tickers whose last price is older than this (relative to the newest price) are not ranked.
MAX_STALENESS_DAYS = 7

MODEL_FACTORIES: dict[str, Callable[[], ClassifierMixin]] = {
    "logistic_regression": lambda: LogisticRegression(C=0.5, max_iter=2000),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=150, min_samples_leaf=50, max_features="sqrt", n_jobs=-1, random_state=42
    ),
    "gradient_boosting": lambda: HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=42
    ),
}
DEFAULT_MODEL = "random_forest"


def available_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    return [col for col in NUMERIC_FEATURES if col in df.columns], [col for col in CATEGORICAL_FEATURES if col in df.columns]


def make_pipeline(model_name: str, numeric: list[str], categorical: list[str]) -> Pipeline:
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model {model_name!r}. Choose from: {', '.join(MODEL_FACTORIES)}")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessor", preprocessor), ("model", MODEL_FACTORIES[model_name]())])


def sample_rows(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    return df.sample(n=max_rows, random_state=42) if len(df) > max_rows else df


def fit_model(model_name: str, rows: pd.DataFrame, numeric: list[str], categorical: list[str], max_rows: int = 200_000) -> Pipeline:
    rows = sample_rows(rows.dropna(subset=[TARGET]), max_rows)
    pipeline = make_pipeline(model_name, numeric, categorical)
    pipeline.fit(rows[numeric + categorical], rows[TARGET].astype(int))
    return pipeline


def train_final_model(
    features: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
    validation: dict | None = None,
    max_rows: int = 200_000,
    model_path: Path | None = MODEL_PATH,
) -> dict:
    """Fit on every labelled row so current predictions use the most recent data, and save the bundle.

    `validation` holds the walk-forward metrics that justify this model; they are stored with it.
    """
    numeric, categorical = available_features(features)
    pipeline = fit_model(model_name, features, numeric, categorical, max_rows)
    bundle = {
        "pipeline": pipeline,
        "model_name": model_name,
        "numeric": numeric,
        "categorical": categorical,
        "trained_through": pd.Timestamp(features.dropna(subset=[TARGET])["date"].max()),
        "metrics": validation or {},
    }
    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, model_path)
    return bundle


def load_model(path: Path = MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run python run_pipeline.py first.")
    return joblib.load(path)


def score_latest(features: pd.DataFrame, model_bundle: dict, max_staleness_days: int = MAX_STALENESS_DAYS) -> pd.DataFrame:
    """Score each ticker's most recent row, ranked by predicted probability (rank 1 = highest)."""
    columns = model_bundle["numeric"] + model_bundle["categorical"]
    latest = features.sort_values("date").groupby("ticker").tail(1)
    # Delisted or stale tickers would otherwise be ranked on old data alongside current ones.
    latest = latest[latest["date"] >= latest["date"].max() - pd.Timedelta(days=max_staleness_days)].copy()
    latest["up_probability_5d"] = model_bundle["pipeline"].predict_proba(latest[columns])[:, 1]
    latest = latest.sort_values("up_probability_5d", ascending=False).reset_index(drop=True)
    latest["rank"] = latest.index + 1
    return latest
