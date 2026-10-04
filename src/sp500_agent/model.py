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

from .config import MODEL_PATH, PRODUCTION_SPEC, ResearchSpec


# Only features that were known on each row's date. Snapshot fundamentals are excluded
# because they come from the download date and would leak future information into history.
NUMERIC_FEATURES = [
    "return_5d",
    "return_20d",
    "momentum_60d",
    "momentum_12_1",
    "volatility_20d",
    "ma_gap_50d",
    "volume_ratio_20d",
    "sentiment_20d",
    "news_count_20d",
    "return_20d_xs_rank",
    "momentum_60d_xs_rank",
    "volatility_20d_xs_rank",
    "momentum_12_1_xs_rank",
    "return_20d_vs_sector",
    # Point-in-time fundamentals (live SEC data only), as cross-sectional ranks: value, quality, growth, size.
    "earnings_yield_xs_rank",
    "sales_yield_xs_rank",
    "book_to_market_xs_rank",
    "profit_margin_xs_rank",
    "roe_xs_rank",
    "revenue_growth_yoy_xs_rank",
    "market_cap_xs_rank",
    # Market conditions (FRED), the same for every stock on a date.
    "vix",
    "vix_change_20d",
    "term_spread",
]
CATEGORICAL_FEATURES = ["sector"]
# Features missing on more than this share of training rows are left out rather than imputed.
MAX_MISSING_SHARE = 0.6
# The production target column; functions take a ResearchSpec to use another one.
TARGET = PRODUCTION_SPEC.target_column
# Tickers whose last price is older than this (relative to the newest price) are not ranked.
MAX_STALENESS_DAYS = 7
# A daily move this large in the last few sessions is either an unadjusted corporate action
# (spin-off, split) or a shock outside what the model has seen; such stocks are left out of the ranking.
SUSPECT_DAILY_MOVE = 0.4

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
    """Model inputs present in `df` with enough data to learn from (not mostly missing, not constant)."""
    numeric = [
        col
        for col in NUMERIC_FEATURES
        if col in df.columns and df[col].notna().mean() >= 1 - MAX_MISSING_SHARE and df[col].nunique(dropna=True) > 1
    ]
    return numeric, [col for col in CATEGORICAL_FEATURES if col in df.columns]


def investable(df: pd.DataFrame) -> pd.DataFrame:
    """Rows for stocks that were index members on that date, when membership is known."""
    return df[df["in_index"]] if "in_index" in df.columns else df


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


def fit_model(model_name: str, rows: pd.DataFrame, numeric: list[str], categorical: list[str], max_rows: int = 200_000, target: str = TARGET) -> Pipeline:
    rows = sample_rows(rows.dropna(subset=[target]), max_rows)
    pipeline = make_pipeline(model_name, numeric, categorical)
    pipeline.fit(rows[numeric + categorical], rows[target].astype(int))
    return pipeline


def train_final_model(
    features: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
    validation: dict | None = None,
    max_rows: int = 200_000,
    model_path: Path | None = MODEL_PATH,
    spec: ResearchSpec = PRODUCTION_SPEC,
) -> dict:
    """Fit on every labelled row so current predictions use the most recent data, and save the bundle.

    `validation` holds the walk-forward metrics that justify this model; they are stored with it.
    """
    features = investable(features)
    labelled = features.dropna(subset=[spec.target_column])
    numeric, categorical = available_features(labelled)
    pipeline = fit_model(model_name, features, numeric, categorical, max_rows, target=spec.target_column)
    bundle = {
        "pipeline": pipeline,
        "model_name": model_name,
        "numeric": numeric,
        "categorical": categorical,
        "trained_through": pd.Timestamp(labelled["date"].max()),
        "spec": spec.as_dict(),
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


def suspect_moves(features: pd.DataFrame, sessions: int = 5) -> dict[str, str]:
    """Tickers with a daily move above SUSPECT_DAILY_MOVE in their last `sessions` rows, with an explanation."""
    if "return_1d" not in features.columns:
        return {}
    recent = features.sort_values("date").groupby("ticker").tail(sessions)
    flagged = recent[recent["return_1d"].abs() > SUSPECT_DAILY_MOVE]
    worst = flagged.loc[flagged["return_1d"].abs().groupby(flagged["ticker"]).idxmax()]
    return {
        row.ticker: f"a {row.return_1d:+.0%} daily move on {row.date.date()} (possibly a corporate action the price source hasn't adjusted, or a shock outside the model's experience)"
        for row in worst.itertuples()
    }


def score_latest(features: pd.DataFrame, model_bundle: dict, max_staleness_days: int = MAX_STALENESS_DAYS) -> pd.DataFrame:
    """Score each ticker's most recent row, ranked by predicted probability (rank 1 = highest).

    Tickers left out because of a suspect recent move are listed in `result.attrs["excluded"]`.
    """
    columns = model_bundle["numeric"] + model_bundle["categorical"]
    members = investable(features)
    latest = members.sort_values("date").groupby("ticker").tail(1)
    # Delisted or stale tickers would otherwise be ranked on old data alongside current ones.
    latest = latest[latest["date"] >= latest["date"].max() - pd.Timedelta(days=max_staleness_days)].copy()
    excluded = suspect_moves(members[members["ticker"].isin(latest["ticker"])])
    latest = latest[~latest["ticker"].isin(excluded)]
    latest["model_probability"] = model_bundle["pipeline"].predict_proba(latest[columns])[:, 1]
    latest = latest.sort_values("model_probability", ascending=False).reset_index(drop=True)
    latest["rank"] = latest.index + 1
    latest.attrs["excluded"] = excluded
    return latest
