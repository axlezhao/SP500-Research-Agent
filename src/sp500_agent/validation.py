"""Walk-forward validation, model comparison and signal diagnostics.

Every number here comes from out-of-sample predictions: each fold trains only on dates before its
test window, with a HORIZON_DAYS gap so no training label overlaps a test label.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from .config import HORIZON_DAYS
from .model import MODEL_FACTORIES, TARGET, available_features, fit_model, investable, sample_rows


@dataclass(frozen=True)
class Fold:
    number: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class WalkForwardResult:
    model_name: str
    predictions: pd.DataFrame  # date, ticker, fold, probability, target and forward returns
    fold_metrics: pd.DataFrame
    columns: list[str] = field(default_factory=list)
    last_pipeline: Pipeline | None = None
    last_test: pd.DataFrame | None = field(default=None, repr=False)


def walk_forward_folds(dates, n_splits: int = 5, min_train_fraction: float = 0.5, embargo_days: int = HORIZON_DAYS) -> list[Fold]:
    """Expanding-window folds: train on everything before the gap, test on the next block of dates."""
    dates = np.sort(pd.unique(pd.Series(dates)))
    first_test = int(len(dates) * min_train_fraction)
    if first_test <= embargo_days or len(dates) - first_test < n_splits:
        raise ValueError(f"Not enough distinct dates ({len(dates)}) for {n_splits} walk-forward folds.")
    blocks = np.array_split(np.arange(first_test, len(dates)), n_splits)
    return [
        Fold(
            number=i + 1,
            train_end=pd.Timestamp(dates[block[0] - embargo_days - 1]),
            test_start=pd.Timestamp(dates[block[0]]),
            test_end=pd.Timestamp(dates[block[-1]]),
        )
        for i, block in enumerate(blocks)
    ]


def information_coefficients(df: pd.DataFrame, score_col: str, return_col: str = "future_return_5d", min_names: int = 5) -> pd.Series:
    """Per-date Spearman rank correlation between a score and the forward return (the quant 'IC').

    Computed with grouped sums rather than a Python call per date, which matters with thousands of dates.
    """
    df = df.dropna(subset=[score_col, return_col])
    df = df[df.groupby("date")["ticker"].transform("count") >= min_names]
    if df.empty:
        return pd.Series(dtype=float)
    ranks = df[[score_col, return_col]].groupby(df["date"]).rank()
    x, y, date = ranks[score_col], ranks[return_col], df["date"]
    stats = pd.DataFrame({"x": x, "y": y, "xy": x * y, "xx": x * x, "yy": y * y, "date": date}).groupby("date").mean()
    cov = stats["xy"] - stats["x"] * stats["y"]
    var_x = stats["xx"] - stats["x"] ** 2
    var_y = stats["yy"] - stats["y"] ** 2
    # A score identical for every stock on a date (e.g. no news anywhere) carries no ranking: leave it out.
    valid = (var_x > 1e-12) & (var_y > 1e-12)
    return (cov[valid] / np.sqrt(var_x[valid] * var_y[valid])).rename(None)


def ic_summary(ic: pd.Series, step: int = HORIZON_DAYS) -> dict:
    """Mean IC over all dates; t-stat from every `step`-th date only.

    Consecutive daily ICs on 5-day forward returns share 4 of their 5 days, so treating them as
    independent would inflate the t-stat by roughly sqrt(step). Non-overlapping dates avoid that.
    """
    independent = ic.iloc[::step]
    tstat = (
        float(independent.mean() / independent.std() * np.sqrt(len(independent)))
        if len(independent) > 2 and independent.std() > 0
        else float("nan")
    )
    return {
        "ic_mean": float(ic.mean()) if len(ic) else float("nan"),
        "ic_tstat": tstat,
        "ic_positive_share": float((ic > 0).mean()) if len(ic) else float("nan"),
        "ic_days": len(ic),
    }


def _fold_metrics(fold: Fold, train: pd.DataFrame, test: pd.DataFrame, probability: np.ndarray) -> dict:
    y = test[TARGET].astype(int)
    majority = int(train[TARGET].mean() >= 0.5)
    scored = test.assign(probability=probability)
    two_classes = y.nunique() == 2
    return {
        "fold": fold.number,
        "train_end": fold.train_end,
        "test_start": fold.test_start,
        "test_end": fold.test_end,
        "train_rows": len(train),
        "test_rows": len(test),
        "auc": roc_auc_score(y, probability) if two_classes else float("nan"),
        "accuracy": accuracy_score(y, probability >= 0.5),
        "baseline_accuracy": accuracy_score(y, np.full(len(y), majority)),
        "brier": brier_score_loss(y, probability),
        "log_loss": log_loss(y, np.clip(probability, 1e-6, 1 - 1e-6), labels=[0, 1]),
        "ic_mean": information_coefficients(scored, "probability").mean(),
    }


def walk_forward(features: pd.DataFrame, model_name: str, n_splits: int = 5, max_rows: int = 150_000) -> WalkForwardResult:
    labelled = investable(features).dropna(subset=[TARGET])
    numeric, categorical = available_features(labelled)
    columns = numeric + categorical
    keep = ["date", "ticker", "sector", TARGET, "future_return_5d", "tradable_return_5d"]
    predictions, metrics = [], []
    pipeline, test = None, None
    for fold in walk_forward_folds(labelled["date"], n_splits=n_splits):
        train = labelled[labelled["date"] <= fold.train_end]
        test = labelled[(labelled["date"] >= fold.test_start) & (labelled["date"] <= fold.test_end)]
        pipeline = fit_model(model_name, train, numeric, categorical, max_rows)
        probability = pipeline.predict_proba(test[columns])[:, 1]
        metrics.append(_fold_metrics(fold, train, test, probability))
        predictions.append(test[[col for col in keep if col in test.columns]].assign(fold=fold.number, probability=probability))
    return WalkForwardResult(
        model_name=model_name,
        predictions=pd.concat(predictions, ignore_index=True),
        fold_metrics=pd.DataFrame(metrics),
        columns=columns,
        last_pipeline=pipeline,
        last_test=test,
    )


def summarise(result: WalkForwardResult) -> dict:
    folds = result.fold_metrics
    preds = result.predictions
    return {
        "model": result.model_name,
        "auc_mean": folds["auc"].mean(),
        "auc_std": folds["auc"].std(),
        "accuracy": accuracy_score(preds[TARGET].astype(int), preds["probability"] >= 0.5),
        "baseline_accuracy": folds["baseline_accuracy"].mean(),
        "brier": brier_score_loss(preds[TARGET].astype(int), preds["probability"]),
        **ic_summary(information_coefficients(preds, "probability")),
        "oos_start": preds["date"].min(),
        "oos_end": preds["date"].max(),
        "folds": len(folds),
    }


def compare_models(
    features: pd.DataFrame, model_names: list[str] | None = None, n_splits: int = 5, max_rows: int = 150_000
) -> tuple[pd.DataFrame, dict[str, WalkForwardResult]]:
    results = {name: walk_forward(features, name, n_splits, max_rows) for name in (model_names or list(MODEL_FACTORIES))}
    comparison = pd.DataFrame([summarise(result) for result in results.values()])
    return comparison.sort_values("auc_mean", ascending=False).reset_index(drop=True), results


def calibration_table(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Do predicted probabilities match how often the predicted event actually happened?"""
    df = predictions.dropna(subset=["probability", TARGET])
    df = df.assign(bucket=pd.qcut(df["probability"].rank(method="first"), q=min(bins, len(df)), labels=False))
    table = df.groupby("bucket").agg(
        mean_predicted=("probability", "mean"), actual_rate=(TARGET, "mean"), rows=(TARGET, "size")
    )
    return table.reset_index(drop=True).rename_axis("bucket").reset_index().assign(bucket=lambda t: t["bucket"] + 1)


def feature_importance(result: WalkForwardResult, max_rows: int = 20_000, n_repeats: int = 5) -> pd.DataFrame:
    """Permutation importance on the last fold's unseen test rows: how much AUC drops when a feature is shuffled."""
    if result.last_pipeline is None or result.last_test is None or result.last_test[TARGET].nunique() < 2:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
    test = sample_rows(result.last_test, max_rows)
    columns = result.columns
    importance = permutation_importance(
        result.last_pipeline, test[columns], test[TARGET].astype(int), scoring="roc_auc", n_repeats=n_repeats, random_state=42, n_jobs=1
    )
    return (
        pd.DataFrame({"feature": columns, "importance_mean": importance.importances_mean, "importance_std": importance.importances_std})
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def signal_report(features: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """IC of each raw feature over the out-of-sample period, next to the model's IC.

    A model that cannot beat its best single input is not adding much.
    """
    features = investable(features)
    window = features[(features["date"] >= predictions["date"].min()) & (features["date"] <= predictions["date"].max())]
    numeric, _ = available_features(window)
    rows = [{"signal": "model_probability", **ic_summary(information_coefficients(predictions, "probability"))}]
    for col in numeric:
        rows.append({"signal": col, **ic_summary(information_coefficients(window, col))})
    return pd.DataFrame(rows).sort_values("ic_mean", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
