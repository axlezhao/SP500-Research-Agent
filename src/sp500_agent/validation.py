"""Walk-forward validation, model comparison and signal diagnostics.

Every number here comes from out-of-sample predictions: each refit trains only on dates before its
test window, with a gap of `spec.horizon` sessions so no training label overlaps a test label. The
model is refit every `retrain_every` sessions (a quarter by default), with its hyperparameters chosen
inside the training window, so the test period never trades a model more than a quarter stale.

Predictions carry generic columns (target, label, future_return, tradable_return) for the spec they
were made under, so the backtest and diagnostics work the same for any horizon, peer group or target.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from .config import DECAY_HORIZONS, PRODUCTION_SPEC, RETRAIN_EVERY, ResearchSpec
from .model import MODEL_FACTORIES, available_features, fit_model, investable, models_for, predict_score, sample_rows, tune_params
from .stats import information_coefficients, newey_west_tstat

__all__ = ["information_coefficients"]  # re-exported: it used to live here

# Columns carried with each prediction for the backtest: exposures for neutral portfolios and costs.
CARRIED = [
    "sector", "industry_group", "beta_252", "market_cap_z", "book_to_market_z", "momentum_12_1_z",
    "volatility_20d", "idio_vol_63", "cs_spread_21d", "market_cap_xs_rank",
]


@dataclass(frozen=True)
class Fold:
    number: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class WalkForwardResult:
    model_name: str
    predictions: pd.DataFrame  # date, ticker, fold, probability, target, label, future_return, tradable_return, ...
    fold_metrics: pd.DataFrame
    spec: ResearchSpec = PRODUCTION_SPEC
    columns: list[str] = field(default_factory=list)
    last_pipeline: Pipeline | None = None
    last_test: pd.DataFrame | None = field(default=None, repr=False)
    params: list[dict] = field(default_factory=list)


def walk_forward_folds(
    dates, n_splits: int = 5, min_train_fraction: float = 0.5, embargo_days: int = PRODUCTION_SPEC.horizon,
    block_size: int | None = None, test_start: pd.Timestamp | None = None,
) -> list[Fold]:
    """Expanding-window folds: train on everything before the gap, test on the next block of dates.

    With `block_size`, the test period is cut into blocks of that many sessions (one refit each);
    otherwise into `n_splits` equal blocks. The test period starts at `test_start` when given, otherwise
    after the first `min_train_fraction` of dates.
    """
    dates = np.sort(pd.unique(pd.Series(dates)))
    first_test = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(test_start)))) if test_start is not None else int(len(dates) * min_train_fraction)
    if first_test <= embargo_days or len(dates) - first_test < n_splits:
        raise ValueError(f"Not enough distinct dates ({len(dates)}) for {n_splits} walk-forward folds.")
    positions = np.arange(first_test, len(dates))
    if block_size:
        blocks = [positions[i : i + block_size] for i in range(0, len(positions), block_size)]
    else:
        blocks = np.array_split(positions, n_splits)
    return [
        Fold(
            number=i + 1,
            train_end=pd.Timestamp(dates[block[0] - embargo_days - 1]),
            test_start=pd.Timestamp(dates[block[0]]),
            test_end=pd.Timestamp(dates[block[-1]]),
        )
        for i, block in enumerate(blocks)
    ]


def ic_summary(ic: pd.Series, step: int = PRODUCTION_SPEC.horizon) -> dict:
    """Mean IC over all dates with a Newey-West t-stat, and the older t-stat from every `step`-th date only.

    Consecutive daily ICs on h-day forward returns share h-1 of their h days. Newey-West with h lags uses
    every date and corrects for that overlap; the non-overlapping version throws away (h-1)/h of the data.
    """
    independent = ic.iloc[::step]
    nonoverlap = (
        float(independent.mean() / independent.std() * np.sqrt(len(independent)))
        if len(independent) > 2 and independent.std() > 0
        else float("nan")
    )
    return {
        "ic_mean": float(ic.mean()) if len(ic) else float("nan"),
        "ic_tstat": newey_west_tstat(ic, lags=step),
        "ic_tstat_nonoverlap": nonoverlap,
        "ic_positive_share": float((ic > 0).mean()) if len(ic) else float("nan"),
        "ic_days": len(ic),
    }


def _fold_metrics(fold: Fold, train: pd.DataFrame, test: pd.DataFrame, probability: np.ndarray, params: dict) -> dict:
    y = test["label"]
    known = y.notna().to_numpy()
    y = y[known].astype(int)
    p = probability[known]
    majority = int(train["label"].mean() >= 0.5) if train["label"].notna().any() else 1
    scored = test.assign(probability=probability)
    two_classes = y.nunique() == 2
    return {
        "fold": fold.number,
        "train_end": fold.train_end,
        "test_start": fold.test_start,
        "test_end": fold.test_end,
        "train_rows": len(train),
        "test_rows": len(test),
        "auc": roc_auc_score(y, p) if two_classes else float("nan"),
        "accuracy": accuracy_score(y, p >= 0.5) if len(y) else float("nan"),
        "baseline_accuracy": accuracy_score(y, np.full(len(y), majority)) if len(y) else float("nan"),
        "brier": brier_score_loss(y, p) if len(y) else float("nan"),
        "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]) if len(y) else float("nan"),
        "ic_mean": information_coefficients(scored, "probability").mean(),
        "params": str(params) if params else "",
    }


def labelled_rows(features: pd.DataFrame, spec: ResearchSpec) -> pd.DataFrame:
    labelled = investable(features).dropna(subset=[spec.target_column])
    return labelled.assign(
        target=labelled[spec.target_column],
        label=labelled[spec.binary_column] if spec.binary_column in labelled.columns else np.nan,
        future_return=labelled[spec.future_column],
        tradable_return=labelled[spec.tradable_column],
    )


def walk_forward(
    features: pd.DataFrame,
    model_name: str,
    n_splits: int = 5,
    max_rows: int = 150_000,
    spec: ResearchSpec = PRODUCTION_SPEC,
    retrain_every: int | None = None,
    tune: bool = False,
    params: dict | None = None,
    test_start: pd.Timestamp | None = None,
) -> WalkForwardResult:
    """Out-of-sample predictions for one model. `params` fixes the hyperparameters (no tuning)."""
    labelled = labelled_rows(features, spec)
    numeric, categorical = available_features(labelled)
    columns = numeric + categorical
    keep = ["date", "ticker", "target", "label", "future_return", "tradable_return", *CARRIED]
    predictions, metrics, chosen = [], [], []
    pipeline, test = None, None
    fixed = params
    for fold in walk_forward_folds(labelled["date"], n_splits=n_splits, embargo_days=spec.horizon, block_size=retrain_every, test_start=test_start):
        train = labelled[labelled["date"] <= fold.train_end]
        test = labelled[(labelled["date"] >= fold.test_start) & (labelled["date"] <= fold.test_end)]
        if fixed is not None:
            params = fixed
        else:
            params = tune_params(model_name, train, numeric, categorical, spec, max_rows) if tune else {}
        pipeline = fit_model(model_name, train, numeric, categorical, max_rows, spec.target_column, spec.target, params, spec.horizon)
        probability = predict_score(pipeline, test[columns])
        metrics.append(_fold_metrics(fold, train, test, probability, params))
        chosen.append(params)
        predictions.append(test[[col for col in dict.fromkeys(keep) if col in test.columns]].assign(fold=fold.number, probability=probability))
    return WalkForwardResult(
        model_name=model_name,
        predictions=pd.concat(predictions, ignore_index=True),
        fold_metrics=pd.DataFrame(metrics),
        spec=spec,
        columns=columns,
        last_pipeline=pipeline,
        last_test=test,
        params=chosen,
    )


def summarise(result: WalkForwardResult) -> dict:
    folds = result.fold_metrics
    preds = result.predictions
    labelled = preds.dropna(subset=["label"])
    return {
        "model": result.model_name,
        "auc_mean": folds["auc"].mean(),
        "auc_std": folds["auc"].std(),
        "accuracy": accuracy_score(labelled["label"].astype(int), labelled["probability"] >= 0.5) if len(labelled) else float("nan"),
        "baseline_accuracy": folds["baseline_accuracy"].mean(),
        "brier": brier_score_loss(labelled["label"].astype(int), labelled["probability"]) if len(labelled) else float("nan"),
        **ic_summary(information_coefficients(preds, "probability"), step=result.spec.horizon),
        "oos_start": preds["date"].min(),
        "oos_end": preds["date"].max(),
        "folds": len(folds),
    }


def compare_models(
    features: pd.DataFrame,
    model_names: list[str] | None = None,
    n_splits: int = 5,
    max_rows: int = 150_000,
    spec: ResearchSpec = PRODUCTION_SPEC,
    retrain_every: int | None = None,
    tune: bool = False,
    log=lambda message: None,
) -> tuple[pd.DataFrame, dict[str, WalkForwardResult]]:
    """Walk every model forward and rank them by the Newey-West t-stat of their rank IC.

    IC, not AUC: the portfolio trades the ranking, and AUC only scores the yes/no split at the median.
    """
    names = [name for name in (model_names or models_for(spec)) if name in MODEL_FACTORIES and spec.target in MODEL_FACTORIES[name]["kinds"]]
    results = {}
    for name in names:
        log(f"    {name}...")
        results[name] = walk_forward(features, name, n_splits, max_rows, spec, retrain_every, tune)
    comparison = pd.DataFrame([summarise(result) for result in results.values()])
    comparison = comparison.sort_values("ic_tstat", ascending=False, na_position="last").reset_index(drop=True)
    return comparison, results


def calibration_table(predictions: pd.DataFrame, bins: int = 10, target: str = "target") -> pd.DataFrame:
    """Do predicted scores match outcomes? For a yes/no target, how often the event happened; for a rank target,
    the average realised percentile."""
    df = predictions.dropna(subset=["probability", target])
    df = df.assign(bucket=pd.qcut(df["probability"].rank(method="first"), q=min(bins, len(df)), labels=False))
    table = df.groupby("bucket").agg(mean_predicted=("probability", "mean"), actual_rate=(target, "mean"), rows=(target, "size"))
    return table.reset_index(drop=True).rename_axis("bucket").reset_index().assign(bucket=lambda t: t["bucket"] + 1)


def feature_importance(result: WalkForwardResult, max_rows: int = 20_000, n_repeats: int = 5) -> pd.DataFrame:
    """Permutation importance on the last fold's unseen test rows: how much the score's rank correlation with
    the target drops when a feature is shuffled (AUC for yes/no targets)."""
    if result.last_pipeline is None or result.last_test is None or result.last_test["target"].nunique() < 2:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
    test = sample_rows(result.last_test, max_rows)
    columns = result.columns
    binary = not result.spec.is_rank

    def scorer(estimator, X, y):
        score = predict_score(estimator, X)
        if binary:
            return roc_auc_score(y, score)
        return pd.Series(score).rank().corr(pd.Series(np.asarray(y)).rank())

    y = test["target"].astype(int) if binary else test["target"].astype(float)
    importance = permutation_importance(result.last_pipeline, test[columns], y, scoring=scorer, n_repeats=n_repeats, random_state=42, n_jobs=1)
    return (
        pd.DataFrame({"feature": columns, "importance_mean": importance.importances_mean, "importance_std": importance.importances_std})
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def signal_report(features: pd.DataFrame, predictions: pd.DataFrame, spec: ResearchSpec = PRODUCTION_SPEC) -> pd.DataFrame:
    """IC of each raw feature over the out-of-sample period, next to the model's IC.

    A model that cannot beat its best single input is not adding much.
    """
    features = investable(features)
    window = features[(features["date"] >= predictions["date"].min()) & (features["date"] <= predictions["date"].max())]
    numeric, _ = available_features(window)
    rows = [{"signal": "model_probability", **ic_summary(information_coefficients(predictions, "probability"), step=spec.horizon)}]
    for col in numeric:
        rows.append({"signal": col, **ic_summary(information_coefficients(window, col, spec.future_column), step=spec.horizon)})
    return pd.DataFrame(rows).sort_values("ic_mean", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def ic_breakdown(predictions: pd.DataFrame, spec: ResearchSpec = PRODUCTION_SPEC, min_names: int = 5) -> pd.DataFrame:
    """Where does the model's ranking power live? Mean IC and t-stat by calendar year, by sector (ranking
    within each sector) and by size tercile (within each third of the universe by market cap)."""
    rows = []
    daily = information_coefficients(predictions, "probability")
    if len(daily):
        for year, ic in daily.groupby(pd.DatetimeIndex(daily.index).year):
            rows.append({"slice": "year", "group": str(year), **ic_summary(ic, step=spec.horizon)})
    if "sector" in predictions.columns:
        for sector, group in predictions.groupby("sector"):
            ic = information_coefficients(group, "probability", min_names=min_names)
            if len(ic) >= 20:
                rows.append({"slice": "sector", "group": str(sector), **ic_summary(ic, step=spec.horizon)})
    if "market_cap_xs_rank" in predictions.columns and predictions["market_cap_xs_rank"].notna().any():
        size = pd.cut(predictions["market_cap_xs_rank"], [0, 1 / 3, 2 / 3, 1], labels=["small third", "middle third", "large third"], include_lowest=True)
        for bucket, group in predictions.groupby(size, observed=True):
            ic = information_coefficients(group, "probability", min_names=min_names)
            if len(ic) >= 20:
                rows.append({"slice": "size", "group": str(bucket), **ic_summary(ic, step=spec.horizon)})
    return pd.DataFrame(rows)


def ic_decay(features: pd.DataFrame, predictions: pd.DataFrame, signals: list[str], horizons=DECAY_HORIZONS, every: int = 5) -> pd.DataFrame:
    """How long does each signal's predictive power last? IC of the model score and the given signals
    against forward returns of 1 to 63 sessions, on every `every`-th out-of-sample date.

    A signal whose IC peaks at a day and vanishes by a week can't pay for weekly trading; one that holds up
    for months can be traded slowly and cheaply.
    """
    members = investable(features)
    window = members[(members["date"] >= predictions["date"].min()) & (members["date"] <= predictions["date"].max())]
    dates = np.sort(window["date"].unique())[::every]
    panel = features[["ticker", "date", "close"]].sort_values(["ticker", "date"])
    by_ticker = panel.groupby("ticker", sort=False)["close"]
    forward = pd.DataFrame({f"fwd_{h}": by_ticker.shift(-h) / panel["close"] - 1 for h in horizons}, index=panel.index)
    panel = pd.concat([panel[["ticker", "date"]], forward], axis=1)
    base = window[window["date"].isin(dates)][["ticker", "date", *[s for s in signals if s in window.columns]]]
    base = base.merge(predictions[["ticker", "date", "probability"]], on=["ticker", "date"], how="left").merge(panel, on=["ticker", "date"], how="left")
    rows = []
    for signal in ["probability", *[s for s in signals if s in base.columns]]:
        for h in horizons:
            ic = information_coefficients(base, signal, f"fwd_{h}")
            lags = int(np.ceil(h / every))
            rows.append({"signal": "model_probability" if signal == "probability" else signal, "horizon": h, "ic_mean": float(ic.mean()) if len(ic) else np.nan, "ic_tstat": newey_west_tstat(ic, lags)})
    return pd.DataFrame(rows)
