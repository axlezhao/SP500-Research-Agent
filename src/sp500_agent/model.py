from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import MODEL_PATH, PRODUCTION_SPEC, ResearchSpec
from .features import EVENT_SIGNALS, POINT_IN_TIME_RATIOS, PRICE_SIGNALS
from .stats import information_coefficients


# Only features that were known on each row's date, each normalised within its date (rank -> standard
# normal), so the model compares stocks with each other rather than with their own past. Snapshot
# fundamentals never qualify: they come from the download date and would leak the future.
NUMERIC_FEATURES = [f"{col}_z" for col in PRICE_SIGNALS + EVENT_SIGNALS + POINT_IN_TIME_RATIOS] + [
    # Market conditions only change rankings through interactions (see features.add_cross_sectional_features);
    # the regime itself is kept for the tree models, which can split on it.
    "reversal_x_vix",
    "momentum_x_vix",
    "vix_regime",
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

# The equal-weight baseline: documented anomalies with signs fixed from the literature, not fitted.
# If a fitted model can't beat this, its extra complexity isn't earning anything.
COMPOSITE_SIGNS = {
    "return_5d_vs_industry_z": -1,  # short-term reversal within industry
    "return_20d_vs_industry_z": -1,  # one-month reversal
    "momentum_12_1_z": 1,
    "residual_momentum_z": 1,
    "industry_momentum_z": 1,
    "high_52w_z": 1,
    "volatility_20d_z": -1,  # low-risk anomaly
    "idio_vol_63_z": -1,
    "max_return_21_z": -1,
    "beta_252_z": -1,
    "earnings_yield_z": 1,
    "book_to_market_z": 1,
    "gross_profitability_z": 1,
    "accruals_z": -1,
    "asset_growth_z": -1,
    "net_issuance_z": -1,
    "sue_z": 1,
    "ear_z": 1,
    "insider_purchases_90d_z": 1,
}


class _SignComposite(BaseEstimator):
    """Average of the signed, standardised inputs; `signs` is aligned with the numeric columns (0 = unused)."""

    def __init__(self, signs=None):
        self.signs = signs

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        return self

    def _composite(self, X) -> np.ndarray:
        signs = np.asarray(self.signs, dtype=float)
        X = np.asarray(X, dtype=float)[:, : len(signs)]
        used = max(int((signs != 0).sum()), 1)
        # An average of k roughly independent standard normals has sd 1/sqrt(k); rescale, then map to (0, 1).
        return norm.cdf(X @ signs / used * np.sqrt(used))


class CompositeClassifier(ClassifierMixin, _SignComposite):
    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return super().fit(X, y)

    def predict_proba(self, X):
        p = self._composite(X)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self._composite(X) >= 0.5).astype(int)


class CompositeRegressor(RegressorMixin, _SignComposite):
    def predict(self, X):
        return self._composite(X)


def _forest(kind, min_samples_leaf=100):
    cls = RandomForestClassifier if kind == "binary" else RandomForestRegressor
    return cls(n_estimators=100, min_samples_leaf=min_samples_leaf, max_features="sqrt", n_jobs=-1, random_state=42)


def _boosting(kind, learning_rate=0.05, max_leaf_nodes=15):
    cls = HistGradientBoostingClassifier if kind == "binary" else HistGradientBoostingRegressor
    return cls(max_iter=200, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes, l2_regularization=1.0, random_state=42)


# name -> target kinds it supports, constructor(kind, signs, **params), and the small grid searched
# inside each training window (choosing by rank IC on the window's own last quarter).
MODEL_FACTORIES = {
    "logistic_regression": {"kinds": ("binary",), "make": lambda kind, signs, C=0.5: LogisticRegression(C=C, max_iter=2000), "grid": [{"C": 0.01}, {"C": 1.0}]},
    "ridge": {"kinds": ("rank",), "make": lambda kind, signs, alpha=1000.0: Ridge(alpha=alpha), "grid": [{"alpha": 10.0}, {"alpha": 1e3}, {"alpha": 1e5}]},
    "random_forest": {"kinds": ("binary", "rank"), "make": lambda kind, signs, **p: _forest(kind, **p), "grid": [{"min_samples_leaf": 100}, {"min_samples_leaf": 500}]},
    "gradient_boosting": {"kinds": ("binary", "rank"), "make": lambda kind, signs, **p: _boosting(kind, **p), "grid": [{"learning_rate": 0.05, "max_leaf_nodes": 15}, {"learning_rate": 0.03, "max_leaf_nodes": 7}]},
    "composite": {"kinds": ("binary", "rank"), "make": lambda kind, signs: (CompositeClassifier if kind == "binary" else CompositeRegressor)(signs=signs), "grid": []},
}
DEFAULT_MODEL = "random_forest"


def models_for(spec: ResearchSpec) -> list[str]:
    """Model names that can learn this spec's target (classifiers for yes/no labels, regressors for ranks)."""
    return [name for name, entry in MODEL_FACTORIES.items() if spec.target in entry["kinds"]]


def linear_model_for(spec: ResearchSpec) -> str:
    return "ridge" if spec.is_rank else "logistic_regression"


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


def make_pipeline(model_name: str, numeric: list[str], categorical: list[str], kind: str = "binary", params: dict | None = None) -> Pipeline:
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model {model_name!r}. Choose from: {', '.join(MODEL_FACTORIES)}")
    entry = MODEL_FACTORIES[model_name]
    if kind not in entry["kinds"]:
        raise ValueError(f"{model_name} can't learn a {kind} target. Models for it: {', '.join(n for n, e in MODEL_FACTORIES.items() if kind in e['kinds'])}")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
        ],
        remainder="drop",
    )
    signs = [COMPOSITE_SIGNS.get(col, 0) for col in numeric]
    return Pipeline([("preprocessor", preprocessor), ("model", entry["make"](kind, signs, **(params or {})))])


def predict_score(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """P(event) for classifiers; for rank regressors the expected percentile, i.e. P(beating a random peer)."""
    model = pipeline[-1]
    if hasattr(model, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    return np.clip(pipeline.predict(X), 0.0, 1.0)


def sample_rows(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    return df.sample(n=max_rows, random_state=42) if len(df) > max_rows else df


def non_overlapping(rows: pd.DataFrame, step: int) -> pd.DataFrame:
    """Every `step`-th date (counting back from the latest), so no two training labels share a forward window.

    Labels on consecutive days overlap by step-1 of their step days: they are nearly copies, and treating
    them as independent makes a model fit noise. Thinning keeps one independent observation per stock per window.
    """
    if step <= 1:
        return rows
    dates = np.sort(rows["date"].unique())[::-1][::step]
    return rows[rows["date"].isin(dates)]


def fit_model(
    model_name: str,
    rows: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    max_rows: int = 200_000,
    target: str = TARGET,
    kind: str = "binary",
    params: dict | None = None,
    step: int = 1,
) -> Pipeline:
    rows = sample_rows(non_overlapping(rows.dropna(subset=[target]), step), max_rows)
    pipeline = make_pipeline(model_name, numeric, categorical, kind, params)
    y = rows[target].astype(int) if kind == "binary" else rows[target].astype(float)
    pipeline.fit(rows[numeric + categorical], y)
    return pipeline


def tune_params(model_name: str, rows: pd.DataFrame, numeric: list[str], categorical: list[str], spec: ResearchSpec, max_rows: int = 100_000) -> dict:
    """Pick hyperparameters using only the training window: fit on its first 75% of dates, score the rank IC
    on the last 25% (after a gap of one horizon). Returns {} when the model has nothing to tune."""
    grid = MODEL_FACTORIES[model_name]["grid"]
    if len(grid) <= 1:
        return grid[0] if grid else {}
    rows = rows.dropna(subset=[spec.target_column])
    dates = np.sort(rows["date"].unique())
    cut = int(len(dates) * 0.75)
    if cut <= spec.horizon + 20 or len(dates) - cut < 20:
        return grid[0]
    inner = rows[rows["date"] <= dates[cut - spec.horizon - 1]]
    validation = non_overlapping(rows[rows["date"] >= dates[cut]], spec.horizon)
    best, best_ic = grid[0], -np.inf
    for params in grid:
        pipeline = fit_model(model_name, inner, numeric, categorical, max_rows // 2, spec.target_column, spec.target, params, spec.horizon)
        scored = validation.assign(score=predict_score(pipeline, validation[numeric + categorical]))
        ic = information_coefficients(scored, "score", spec.future_column).mean()
        if np.isfinite(ic) and ic > best_ic:
            best, best_ic = params, ic
    return best


def train_final_model(
    features: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
    validation: dict | None = None,
    max_rows: int = 200_000,
    model_path: Path | None = MODEL_PATH,
    spec: ResearchSpec = PRODUCTION_SPEC,
    params: dict | None = None,
    tune: bool = True,
) -> dict:
    """Fit on every labelled row so current predictions use the most recent data, and save the bundle.

    `validation` holds the walk-forward metrics that justify this model; they are stored with it.
    """
    if model_name not in models_for(spec):
        model_name = linear_model_for(spec)
    features = investable(features)
    labelled = features.dropna(subset=[spec.target_column])
    numeric, categorical = available_features(labelled)
    if params is None:
        params = tune_params(model_name, labelled, numeric, categorical, spec, max_rows) if tune else {}
    pipeline = fit_model(model_name, labelled, numeric, categorical, max_rows, spec.target_column, spec.target, params, spec.horizon)
    bundle = {
        "pipeline": pipeline,
        "model_name": model_name,
        "params": params,
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
    """Score each ticker's most recent row, ranked by the model's score (rank 1 = highest).

    Tickers left out because of a suspect recent move are listed in `result.attrs["excluded"]`.
    """
    columns = model_bundle["numeric"] + model_bundle["categorical"]
    members = investable(features)
    latest = members.sort_values("date").groupby("ticker").tail(1)
    # Delisted or stale tickers would otherwise be ranked on old data alongside current ones.
    latest = latest[latest["date"] >= latest["date"].max() - pd.Timedelta(days=max_staleness_days)].copy()
    excluded = suspect_moves(members[members["ticker"].isin(latest["ticker"])])
    latest = latest[~latest["ticker"].isin(excluded)]
    for col in columns:  # a bundle trained on richer data than this frame: missing inputs are imputed
        if col not in latest.columns:
            latest[col] = np.nan
    latest["model_probability"] = predict_score(model_bundle["pipeline"], latest[columns])
    latest = latest.sort_values("model_probability", ascending=False).reset_index(drop=True)
    latest["rank"] = latest.index + 1
    latest.attrs["excluded"] = excluded
    return latest
