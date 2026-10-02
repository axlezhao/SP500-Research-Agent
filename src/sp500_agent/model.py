from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import HORIZON_DAYS, MODEL_PATH


# Only features that were known on each row's date. Snapshot fundamentals are excluded
# because they come from the download date and would leak future information into history.
NUMERIC_FEATURES = ["return_5d", "return_20d", "momentum_60d", "volatility_20d", "sentiment_20d", "news_count_20d"]
CATEGORICAL_FEATURES = ["sector"]
TARGET = "target_up_5d"
# Tickers whose last price is older than this (relative to the newest price) are not ranked.
MAX_STALENESS_DAYS = 7


def available_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    return [col for col in NUMERIC_FEATURES if col in df.columns], [col for col in CATEGORICAL_FEATURES if col in df.columns]


def _make_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessor", preprocessor), ("model", RandomForestClassifier(n_estimators=120, min_samples_leaf=10, random_state=42, class_weight="balanced_subsample", n_jobs=-1))])


def time_split(df: pd.DataFrame, test_fraction: float = 0.2, embargo_days: int = HORIZON_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on earlier dates and test on later ones.

    The first `embargo_days` sessions after the cut are skipped: training rows near the cut have
    targets that look `HORIZON_DAYS` ahead, so testing on those sessions would overlap their labels.
    """
    dates = np.sort(df["date"].unique())
    cut = int(len(dates) * (1 - test_fraction))
    if cut < 1 or cut + embargo_days >= len(dates):
        raise ValueError(f"Not enough distinct dates ({len(dates)}) for a time split with a {embargo_days}-day gap.")
    train = df[df["date"] < dates[cut]]
    test = df[df["date"] >= dates[cut + embargo_days]]
    return train, test


def _sample(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    return df.sample(n=max_rows, random_state=42) if len(df) > max_rows else df


def train_return_direction_model(
    features: pd.DataFrame,
    max_rows: int = 200_000,
    model_path: Path | None = MODEL_PATH,
) -> tuple[dict, str]:
    """Evaluate on a time-ordered holdout, then refit on all labelled rows and save the bundle."""
    numeric, categorical = available_features(features)
    columns = numeric + categorical
    labelled = features.dropna(subset=[TARGET])
    train, test = time_split(labelled)

    sampled_train = _sample(train, max_rows)
    holdout_pipeline = _make_pipeline(numeric, categorical)
    holdout_pipeline.fit(sampled_train[columns], sampled_train[TARGET].astype(int))
    y_test = test[TARGET].astype(int)
    predictions = holdout_pipeline.predict(test[columns])
    majority_class = int(train[TARGET].mean() >= 0.5)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "baseline_accuracy": accuracy_score(y_test, np.full(len(y_test), majority_class)),
        "auc": roc_auc_score(y_test, holdout_pipeline.predict_proba(test[columns])[:, 1]) if y_test.nunique() == 2 else float("nan"),
        "train_end": pd.Timestamp(train["date"].max()),
        "test_start": pd.Timestamp(test["date"].min()),
        "test_end": pd.Timestamp(test["date"].max()),
    }

    report = (
        f"Time-ordered holdout: train through {metrics['train_end'].date()}, "
        f"test {metrics['test_start'].date()} to {metrics['test_end'].date()} "
        f"({HORIZON_DAYS}-session gap between them).\n\n"
        + classification_report(y_test, predictions, zero_division=0)
        + f"\nROC AUC: {metrics['auc']:.3f}  (0.500 = no skill)"
        + f"\nAccuracy: {metrics['accuracy']:.3f}  vs. always predicting "
        + f"'{'up' if majority_class else 'down'}': {metrics['baseline_accuracy']:.3f}\n"
    )
    if len(sampled_train) < len(train):
        report += f"Holdout training sample: {len(sampled_train):,} of {len(train):,} rows.\n"

    # The saved model uses every labelled row, including the holdout period, so predictions use the most recent data.
    final_rows = _sample(labelled, max_rows)
    pipeline = _make_pipeline(numeric, categorical)
    pipeline.fit(final_rows[columns], final_rows[TARGET].astype(int))
    bundle = {"pipeline": pipeline, "numeric": numeric, "categorical": categorical, "metrics": metrics}
    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, model_path)
    return bundle, report


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
