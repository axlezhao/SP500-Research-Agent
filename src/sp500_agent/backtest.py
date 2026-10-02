"""Portfolio backtest of the model's out-of-sample signal.

Every HORIZON_DAYS sessions the stocks are ranked by predicted probability. Three portfolios are
tracked: long the top quantile, long-short (top minus bottom quantile), and an equal-weight
benchmark of every stock. Positions are entered EXECUTION_LAG_DAYS after the signal and held for
HORIZON_DAYS, so the next rebalance never overlaps the previous holding period.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import HORIZON_DAYS, TRADING_DAYS_PER_YEAR
from .validation import ic_summary, information_coefficients

STRATEGIES = ["long_only", "long_short", "benchmark"]


@dataclass(frozen=True)
class BacktestConfig:
    holding_days: int = HORIZON_DAYS
    quantile: float = 0.2
    cost_bps: float = 10.0  # charged on every unit of weight traded, each way
    min_names: int = 5
    score_col: str = "probability"
    return_col: str = "tradable_return_5d"


@dataclass
class BacktestResult:
    config: BacktestConfig
    returns: pd.DataFrame  # one row per rebalance: net and gross returns, turnover and IC
    summary: pd.DataFrame  # one row per strategy
    quantile_returns: pd.DataFrame


def _target_weights(cross_section: pd.DataFrame, config: BacktestConfig) -> dict[str, pd.Series]:
    ranked = cross_section.sort_values(config.score_col, ascending=False)
    n = len(ranked)
    k = max(1, int(round(n * config.quantile)))
    top, bottom = ranked["ticker"].iloc[:k], ranked["ticker"].iloc[-k:]
    return {
        "long_only": pd.Series(1.0 / k, index=top),
        "long_short": pd.concat([pd.Series(1.0 / k, index=top), pd.Series(-1.0 / k, index=bottom)]),
        "benchmark": pd.Series(1.0 / n, index=ranked["ticker"]),
    }


def _turnover(new: pd.Series, old: pd.Series) -> float:
    return float(new.sub(old, fill_value=0.0).abs().sum())


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    return float((equity / equity.cummax() - 1).min()) if len(equity) else float("nan")


def performance(returns: pd.Series, periods_per_year: float) -> dict:
    returns = returns.dropna()
    n = len(returns)
    if n == 0:
        return {}
    growth = float((1 + returns).prod())
    std = returns.std()
    return {
        "periods": n,
        "total_return": growth - 1,
        "cagr": growth ** (periods_per_year / n) - 1 if growth > 0 else -1.0,
        "ann_volatility": std * np.sqrt(periods_per_year) if n > 1 else float("nan"),
        "sharpe": returns.mean() / std * np.sqrt(periods_per_year) if n > 1 and std > 0 else float("nan"),
        "max_drawdown": max_drawdown(returns),
        "hit_rate": float((returns > 0).mean()),
    }


def quantile_returns(predictions: pd.DataFrame, rebalance_dates, config: BacktestConfig, buckets: int = 5) -> pd.DataFrame:
    """Average forward return by prediction quintile: a working signal should rise from Q1 to Q5."""
    df = predictions[predictions["date"].isin(rebalance_dates)].dropna(subset=[config.score_col, config.return_col])
    df = df[df.groupby("date")["ticker"].transform("count") >= max(buckets, config.min_names)]
    if df.empty:
        return pd.DataFrame(columns=["quantile", "mean_return", "observations"])
    pct = df.groupby("date")[config.score_col].rank(pct=True, method="first")
    df = df.assign(quantile=np.ceil(pct * buckets).clip(1, buckets).astype(int))
    per_date = df.groupby(["date", "quantile"])[config.return_col].mean().reset_index()
    return (
        per_date.groupby("quantile")[config.return_col]
        .agg(mean_return="mean", observations="size")
        .reset_index()
    )


def run_backtest(predictions: pd.DataFrame, config: BacktestConfig | None = None) -> BacktestResult:
    config = config or BacktestConfig()
    usable = predictions.dropna(subset=[config.score_col, config.return_col])
    dates = np.sort(usable["date"].unique())
    rebalance_dates = dates[:: config.holding_days]

    previous = {name: pd.Series(dtype=float) for name in STRATEGIES}
    rows = []
    for date in rebalance_dates:
        cross_section = usable[usable["date"] == date]
        if len(cross_section) < config.min_names:
            continue
        realised = cross_section.set_index("ticker")[config.return_col]
        row = {"date": pd.Timestamp(date), "names": len(cross_section)}
        for name, weights in _target_weights(cross_section, config).items():
            gross = float((weights * realised.reindex(weights.index)).sum())
            turnover = _turnover(weights, previous[name])
            # Holdings drift with returns during the period; the next rebalance trades back from the drifted weights.
            drifted = weights * (1 + realised.reindex(weights.index))
            previous[name] = drifted / drifted.abs().sum() * weights.abs().sum()
            row[f"{name}_gross"] = gross
            row[f"{name}_turnover"] = turnover
            row[name] = gross - turnover * config.cost_bps / 10_000
        rows.append(row)

    returns = pd.DataFrame(rows)
    if returns.empty:
        raise ValueError("No rebalance dates had enough stocks to backtest.")
    ic = information_coefficients(usable[usable["date"].isin(rebalance_dates)], config.score_col, config.return_col, config.min_names)
    returns["ic"] = returns["date"].map(ic)
    returns["active"] = returns["long_only"] - returns["benchmark"]

    periods_per_year = TRADING_DAYS_PER_YEAR / config.holding_days
    summary_rows = []
    for name in STRATEGIES:
        stats = performance(returns[name], periods_per_year)
        stats.update(strategy=name, avg_turnover=returns[f"{name}_turnover"].iloc[1:].mean(), gross_total_return=float((1 + returns[f"{name}_gross"]).prod() - 1))
        summary_rows.append(stats)
    summary = pd.DataFrame(summary_rows).set_index("strategy")
    active = returns["active"]
    summary.loc["long_only", "information_ratio"] = (
        active.mean() / active.std() * np.sqrt(periods_per_year) if len(active) > 1 and active.std() > 0 else float("nan")
    )
    for key, value in ic_summary(ic, step=1).items():  # rebalance dates are already non-overlapping
        summary.loc["long_short", key] = value
    return BacktestResult(config, returns, summary.reset_index(), quantile_returns(usable, rebalance_dates, config))


def config_dict(config: BacktestConfig) -> dict:
    return asdict(config)
