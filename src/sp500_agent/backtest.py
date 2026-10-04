"""Portfolio backtest of the model's out-of-sample signal.

Every `holding_days` sessions the stocks are ranked by predicted probability. Portfolios tracked:
long the top quantile, long-short (top minus bottom quantile), an equal-weight benchmark of every
member and, when index prices are supplied, the S&P 500 itself (SPY, dividends included). With
`neutralize="sector"` the top and bottom quantiles are picked within each sector, so the portfolio
makes no bet on which sectors will do well. Positions are entered EXECUTION_LAG_DAYS after the signal
and held for `holding_days`, so the next rebalance never overlaps the previous holding period.

With a risk-free rate (annual, decimal, by date), Sharpe ratios of long-only portfolios and the
benchmarks are computed on returns in excess of cash. The long-short portfolio is self-financing, so
its Sharpe uses raw returns.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import EXECUTION_LAG_DAYS, TRADING_DAYS_PER_YEAR
from .validation import ic_summary, information_coefficients

STRATEGIES = ["long_only", "long_short", "benchmark"]
MARKET = "sp500"
MIN_SECTOR_NAMES = 4


@dataclass(frozen=True)
class BacktestConfig:
    holding_days: int = 5
    quantile: float = 0.2
    cost_bps: float = 10.0  # charged on every unit of weight traded, each way
    min_names: int = 5
    score_col: str = "probability"
    return_col: str = "tradable_return"
    neutralize: str = "none"  # "none" or "sector"


@dataclass
class BacktestResult:
    config: BacktestConfig
    returns: pd.DataFrame  # one row per rebalance: net and gross returns, turnover and IC
    summary: pd.DataFrame  # one row per strategy
    quantile_returns: pd.DataFrame


def _pick(ranked: pd.DataFrame, quantile: float) -> tuple[pd.Index, pd.Index]:
    k = max(1, int(round(len(ranked) * quantile)))
    return pd.Index(ranked["ticker"].iloc[:k]), pd.Index(ranked["ticker"].iloc[-k:])


def _target_weights(cross_section: pd.DataFrame, config: BacktestConfig) -> dict[str, pd.Series]:
    ranked = cross_section.sort_values(config.score_col, ascending=False)
    if config.neutralize == "sector" and "sector" in ranked.columns:
        tops, bottoms = [], []
        for _, group in ranked.groupby("sector", sort=False):
            if len(group) >= MIN_SECTOR_NAMES:
                top, bottom = _pick(group, config.quantile)
                tops.append(top)
                bottoms.append(bottom)
        top = tops[0].append(tops[1:]) if tops else pd.Index([])
        bottom = bottoms[0].append(bottoms[1:]) if bottoms else pd.Index([])
    else:
        top, bottom = _pick(ranked, config.quantile)
    n = len(ranked)
    long_weights = pd.Series(1.0 / max(len(top), 1), index=top)
    return {
        "long_only": long_weights,
        "long_short": pd.concat([long_weights, pd.Series(-1.0 / max(len(bottom), 1), index=bottom)]),
        "benchmark": pd.Series(1.0 / n, index=ranked["ticker"]),
    }


def _turnover(new: pd.Series, old: pd.Series) -> float:
    return float(new.sub(old, fill_value=0.0).abs().sum())


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    return float((equity / equity.cummax() - 1).min()) if len(equity) else float("nan")


def performance(returns: pd.Series, periods_per_year: float, cash: pd.Series | None = None) -> dict:
    returns = returns.dropna()
    n = len(returns)
    if n == 0:
        return {}
    growth = float((1 + returns).prod())
    excess = returns - cash.reindex(returns.index).fillna(0.0) if cash is not None else returns
    std = excess.std()
    return {
        "periods": n,
        "total_return": growth - 1,
        "cagr": growth ** (periods_per_year / n) - 1 if growth > 0 else -1.0,
        "ann_volatility": std * np.sqrt(periods_per_year) if n > 1 else float("nan"),
        "sharpe": excess.mean() / std * np.sqrt(periods_per_year) if n > 1 and std > 0 else float("nan"),
        "max_drawdown": max_drawdown(returns),
        "hit_rate": float((returns > 0).mean()),
    }


def index_period_returns(index_prices: pd.Series, dates, holding_days: int, lag: int = EXECUTION_LAG_DAYS) -> pd.Series:
    """Return of the index over the same window as each rebalance: enter `lag` sessions after the date, hold `holding_days`."""
    prices = index_prices.dropna().sort_index()
    prices.index = pd.to_datetime(prices.index).astype("datetime64[ns]")
    values = prices.to_numpy()
    out = {}
    for date in pd.to_datetime(pd.Index(dates)).astype("datetime64[ns]"):
        i = prices.index.searchsorted(date)  # first index session on or after the signal date
        start, end = i + lag, i + lag + holding_days
        out[date] = values[end] / values[start] - 1 if end < len(values) and i < len(values) and prices.index[i] == date else np.nan
    return pd.Series(out, dtype=float)


def quantile_returns(predictions: pd.DataFrame, rebalance_dates, config: BacktestConfig, buckets: int = 5) -> pd.DataFrame:
    """Average forward return by prediction quintile: a working signal should rise from Q1 to Q5."""
    df = predictions[predictions["date"].isin(rebalance_dates)].dropna(subset=[config.score_col, config.return_col])
    df = df[df.groupby("date")["ticker"].transform("count") >= max(buckets, config.min_names)]
    if df.empty:
        return pd.DataFrame(columns=["quantile", "mean_return", "observations"])
    pct = df.groupby("date")[config.score_col].rank(pct=True, method="first")
    df = df.assign(quantile=np.ceil(pct * buckets).clip(1, buckets).astype(int))
    per_date = df.groupby(["date", "quantile"])[config.return_col].mean().reset_index()
    return per_date.groupby("quantile")[config.return_col].agg(mean_return="mean", observations="size").reset_index()


def run_backtest(
    predictions: pd.DataFrame,
    config: BacktestConfig | None = None,
    risk_free: pd.Series | None = None,
    index_prices: pd.Series | None = None,
) -> BacktestResult:
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
    strategies = list(STRATEGIES)
    if index_prices is not None and not index_prices.dropna().empty:
        market = index_period_returns(index_prices, returns["date"], config.holding_days)
        if market.notna().mean() > 0.9:  # only when the index covers the backtest period
            returns[MARKET] = returns["date"].map(market).values
            returns[f"{MARKET}_gross"] = returns[MARKET]
            returns[f"{MARKET}_turnover"] = 0.0
            returns = returns.dropna(subset=[MARKET]).reset_index(drop=True)
            strategies.append(MARKET)
    ic = information_coefficients(usable[usable["date"].isin(rebalance_dates)], config.score_col, config.return_col, config.min_names)
    returns["ic"] = returns["date"].map(ic)
    returns["active"] = returns["long_only"] - returns["benchmark"]
    if risk_free is not None and not risk_free.dropna().empty:
        rate = risk_free.dropna().sort_index()
        rate_frame = rate.rename("rate").rename_axis("date").reset_index().astype({"date": "datetime64[ns]"})
        annual = pd.merge_asof(returns[["date"]].astype({"date": "datetime64[ns]"}), rate_frame, on="date", direction="backward")["rate"]
        returns["cash"] = (annual.fillna(0.0) * config.holding_days / TRADING_DAYS_PER_YEAR).values

    periods_per_year = TRADING_DAYS_PER_YEAR / config.holding_days
    summary_rows = []
    for name in strategies:
        cash = returns["cash"] if "cash" in returns.columns and name != "long_short" else None
        stats = performance(returns[name], periods_per_year, cash)
        stats.update(strategy=name, avg_turnover=returns[f"{name}_turnover"].iloc[1:].mean(), gross_total_return=float((1 + returns[f"{name}_gross"]).prod() - 1))
        summary_rows.append(stats)
    summary = pd.DataFrame(summary_rows).set_index("strategy")
    for benchmark, column in [("benchmark", "information_ratio"), (MARKET, "information_ratio_vs_sp500")]:
        if benchmark in returns.columns:
            active = returns["long_only"] - returns[benchmark]
            summary.loc["long_only", column] = active.mean() / active.std() * np.sqrt(periods_per_year) if len(active) > 1 and active.std() > 0 else float("nan")
    for key, value in ic_summary(ic, step=1).items():  # rebalance dates are already non-overlapping
        summary.loc["long_short", key] = value
    return BacktestResult(config, returns, summary.reset_index(), quantile_returns(usable, rebalance_dates, config))


def config_dict(config: BacktestConfig) -> dict:
    return asdict(config)
