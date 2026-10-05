"""Portfolio backtest of the model's out-of-sample signal.

Every `holding_days` sessions the stocks are ranked by the model's score. Portfolios tracked: long the
top quantile, long-short (top minus bottom quantile), an equal-weight benchmark of every member and, when
index prices are supplied, the S&P 500 itself (SPY, dividends included). Positions are entered
EXECUTION_LAG_DAYS after the signal and held for `holding_days`, so the next rebalance never overlaps the
previous holding period.

Construction options, all aimed at the cost drag that decides short-horizon results:
- `exit_quantile`: a buffer. Buy in the top `quantile`, but only sell once a holding drops out of the top
  `exit_quantile`, so stocks hovering around the cut-off don't churn.
- `smooth_days`: rank on each stock's score averaged over its last few daily predictions.
- `construction="optimizer"`: a cost-aware optimiser (cvxpy) with a Ledoit-Wolf risk model. Long-only
  maximises expected return net of trading costs within a tracking-error budget against the equal-weight
  universe; long-short is also neutral to market beta, size and value and runs at a target volatility.
- `neutralize="sector"`: pick the top and bottom quantiles within each sector.

Costs: a flat `cost_bps` per unit of weight traded, or per-stock costs (`cost_model="spread"`) that vary
with each stock's Corwin-Schultz bid-ask spread estimated from daily highs and lows, scaled so the median
stock pays `cost_bps`. Only the estimator's cross-section is used: its level overstates spreads for large,
liquid stocks (a median near 50 bps for S&P 500 members, against quoted spreads of a few bps). Plus an
annual borrow fee on shorts.

With a risk-free rate (annual, decimal, by date), Sharpe ratios of long-only portfolios and the
benchmarks are computed on returns in excess of cash. The long-short portfolio is self-financing, so
its Sharpe uses raw returns.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
from scipy.special import ndtri

from .config import EXECUTION_LAG_DAYS, TRADING_DAYS_PER_YEAR
from .validation import ic_summary, information_coefficients

STRATEGIES = ["long_only", "long_short", "benchmark"]
MARKET = "sp500"
MIN_SECTOR_NAMES = 4


@dataclass(frozen=True)
class BacktestConfig:
    holding_days: int = 5
    quantile: float = 0.2
    cost_bps: float = 10.0  # flat cost per unit of weight traded, each way
    min_names: int = 5
    score_col: str = "probability"
    return_col: str = "tradable_return"
    neutralize: str = "none"  # "none" or "sector"
    exit_quantile: float | None = None  # buffer: hold until a stock leaves this top share (None = no buffer)
    smooth_days: int = 1  # rank on the average of each stock's last N daily scores
    construction: str = "quantile"  # "quantile" or "optimizer"
    cost_model: str = "flat"  # "flat" (cost_bps) or "spread" (cost_bps scaled by each stock's spread relative to the day's median)
    spread_range: tuple = (0.25, 4.0)  # limits on a stock's cost relative to the median stock
    borrow_bps: float = 0.0  # annual fee on the value of short positions
    offset: int = 0  # which session of each holding period the rebalance falls on
    # Optimiser settings.
    assumed_ic: float = 0.02  # converts scores to expected returns: alpha = IC x volatility x z-score (Grinold)
    # The optimiser weighs one period's expected return against trading costs spread over this many periods,
    # the typical life of a position. Charging the full cost against a single period makes it so averse to
    # trading that it never rebalances, and its backtest becomes a buy-and-hold of its first portfolio.
    # Realised costs are always charged in full.
    cost_amortization: float = 4.0
    max_weight: float = 0.02
    tracking_error: float = 0.04  # long-only active risk budget, annual
    target_vol: float = 0.10  # long-short risk budget, annual
    max_gross: float = 3.0
    neutral_exposures: tuple = ("beta_252", "market_cap_z", "book_to_market_z")
    risk_lookback: int = 126


@dataclass
class BacktestResult:
    config: BacktestConfig
    returns: pd.DataFrame  # one row per rebalance: net and gross returns, turnover, costs and IC
    summary: pd.DataFrame  # one row per strategy
    quantile_returns: pd.DataFrame


def _pick(ranked: pd.DataFrame, quantile: float) -> tuple[pd.Index, pd.Index]:
    k = max(1, int(round(len(ranked) * quantile)))
    return pd.Index(ranked["ticker"].iloc[:k]), pd.Index(ranked["ticker"].iloc[-k:])


def _buffered(order: pd.Index, k: int, k_exit: int, held: set) -> pd.Index:
    """Top k of `order` (best first), except that current holdings still inside the top k_exit are kept."""
    if k_exit <= k or not held:
        return order[:k]
    keep = [t for t in order[:k_exit] if t in held][:k]
    fill = [t for t in order if t not in keep][: k - len(keep)]
    chosen = set(keep) | set(fill)
    return pd.Index([t for t in order if t in chosen])


def _select(ranked: pd.DataFrame, config: BacktestConfig, held_long: set, held_short: set) -> tuple[pd.Index, pd.Index]:
    order = pd.Index(ranked["ticker"])
    k = max(1, int(round(len(ranked) * config.quantile)))
    k_exit = max(k, int(round(len(ranked) * (config.exit_quantile or config.quantile))))
    return _buffered(order, k, k_exit, held_long), _buffered(order[::-1], k, k_exit, held_short)


def _target_weights(cross_section: pd.DataFrame, config: BacktestConfig, previous: dict | None = None) -> dict[str, pd.Series]:
    ranked = cross_section.sort_values(config.score_col, ascending=False)
    previous = previous or {}
    held_long = set(previous.get("long_only", pd.Series(dtype=float)).index)
    short_book = previous.get("long_short", pd.Series(dtype=float))
    held_short = set(short_book[short_book < 0].index)
    if config.neutralize == "sector" and "sector" in ranked.columns:
        tops, bottoms = [], []
        for _, group in ranked.groupby("sector", sort=False):
            if len(group) >= MIN_SECTOR_NAMES:
                top, bottom = _select(group, config, held_long, held_short)
                tops.append(top)
                bottoms.append(bottom)
        top = tops[0].append(tops[1:]) if tops else pd.Index([])
        bottom = bottoms[0].append(bottoms[1:]) if bottoms else pd.Index([])
    else:
        top, bottom = _select(ranked, config, held_long, held_short)
    n = len(ranked)
    long_weights = pd.Series(1.0 / max(len(top), 1), index=top)
    return {
        "long_only": long_weights,
        "long_short": pd.concat([long_weights, pd.Series(-1.0 / max(len(bottom), 1), index=bottom)]),
        "benchmark": pd.Series(1.0 / n, index=ranked["ticker"]),
    }


def _risk_factors(history: pd.DataFrame, tickers: pd.Index, lookback: int, holding_days: int) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf risk model in factored form: risk(w)^2 = ||A w||^2 + d ||w||^2 over the holding period."""
    from sklearn.covariance import ledoit_wolf_shrinkage

    window = history.reindex(columns=tickers).tail(lookback)
    window = window.loc[window.notna().any(axis=1)].fillna(0.0)
    if len(window) < 20:
        vol = np.full(len(tickers), 0.02)
        return np.zeros((1, len(tickers))), float(np.mean(vol**2) * holding_days)
    X = window.to_numpy(dtype=float)
    X = X - X.mean(axis=0)
    t = len(X)
    shrinkage = float(ledoit_wolf_shrinkage(X, assume_centered=True))
    mu = float(np.trace(X.T @ X / t) / X.shape[1])
    return np.sqrt((1 - shrinkage) * holding_days / t) * X, shrinkage * mu * holding_days


def _optimize(cross_section: pd.DataFrame, config: BacktestConfig, previous: dict, history: pd.DataFrame, unit_cost: pd.Series) -> dict[str, pd.Series]:
    """Cost-aware long-only and factor-neutral long-short books (falls back to quantile weights if a solve fails)."""
    import cvxpy as cp

    fallback = _target_weights(cross_section, config, previous)
    cs = cross_section.reset_index(drop=True)
    tickers = pd.Index(cs["ticker"])
    n = len(cs)
    scores = cs[config.score_col].rank(method="average")
    z = ndtri(((scores - 0.5) / n).to_numpy())
    vol = cs["volatility_20d"].fillna(cs["volatility_20d"].median()).fillna(0.3).to_numpy() if "volatility_20d" in cs else np.full(n, 0.3)
    horizon = np.sqrt(config.holding_days / TRADING_DAYS_PER_YEAR)
    alpha = config.assumed_ic * vol * horizon * z
    cost = unit_cost.reindex(tickers).to_numpy()
    A, d = _risk_factors(history, tickers, config.risk_lookback, config.holding_days)
    cap = max(config.max_weight, 1.5 / n)  # a small universe can't be fully invested at 2% per name
    out = dict(fallback)

    def solve(benchmark: np.ndarray, prev: np.ndarray, budget: float, long_only: bool) -> np.ndarray | None:
        w = cp.Variable(n)
        active = w - benchmark
        risk = cp.norm(cp.hstack([A @ active, np.sqrt(d) * active]))
        constraints = [risk <= budget * horizon]
        if long_only:
            constraints += [cp.sum(w) == 1, w >= 0, w <= cap]
        else:
            constraints += [cp.sum(w) == 0, cp.abs(w) <= cap, cp.norm1(w) <= config.max_gross]
            for exposure in config.neutral_exposures:
                if exposure in cs.columns and cs[exposure].notna().any():
                    constraints.append(cs[exposure].fillna(cs[exposure].median()).to_numpy() @ w == 0)
        problem = cp.Problem(cp.Maximize(alpha @ w - cost / config.cost_amortization @ cp.abs(w - prev)), constraints)
        try:
            problem.solve(solver=cp.CLARABEL)
        except cp.error.SolverError:
            return None
        if w.value is None or problem.status not in ("optimal", "optimal_inaccurate"):
            return None
        weights = np.where(np.abs(w.value) < 1e-5, 0.0, w.value)
        return weights / weights.sum() if long_only else weights

    prev_long = previous.get("long_only", pd.Series(dtype=float)).reindex(tickers).fillna(0.0).to_numpy()
    long_only = solve(np.full(n, 1.0 / n), prev_long, config.tracking_error, True)
    if long_only is not None:
        out["long_only"] = pd.Series(long_only, index=tickers).loc[lambda s: s != 0]
    prev_ls = previous.get("long_short", pd.Series(dtype=float)).reindex(tickers).fillna(0.0).to_numpy()
    long_short = solve(np.zeros(n), prev_ls, config.target_vol, False)
    if long_short is not None:
        out["long_short"] = pd.Series(long_short, index=tickers).loc[lambda s: s != 0]
    return out


def _unit_costs(cross_section: pd.DataFrame, config: BacktestConfig) -> pd.Series:
    """Cost per unit of weight traded, per ticker."""
    flat = pd.Series(config.cost_bps / 10_000, index=cross_section["ticker"].to_numpy())
    if config.cost_model != "spread" or "cs_spread_21d" not in cross_section.columns:
        return flat
    spread = cross_section.set_index("ticker")["cs_spread_21d"]
    median = spread.median()
    if not median or not np.isfinite(median):
        return flat
    relative = (spread / median).clip(*config.spread_range).fillna(1.0)
    return config.cost_bps / 10_000 * relative


def _trading_cost(new: pd.Series, old: pd.Series, unit_cost: pd.Series) -> tuple[float, float]:
    """(turnover, cost) of moving from `old` to `new` weights."""
    traded = new.sub(old, fill_value=0.0).abs()
    fallback = float(unit_cost.median()) if len(unit_cost) else 0.0
    per_name = unit_cost.reindex(traded.index).fillna(fallback)
    return float(traded.sum()), float((traded * per_name).sum())


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


def smooth_scores(predictions: pd.DataFrame, score_col: str, days: int) -> pd.DataFrame:
    """Each stock's score averaged over its last `days` daily predictions (using only past and current days)."""
    if days <= 1:
        return predictions
    predictions = predictions.sort_values(["ticker", "date"])
    smoothed = predictions.groupby("ticker", sort=False)[score_col].transform(lambda s: s.rolling(days, min_periods=1).mean())
    return predictions.assign(**{score_col: smoothed}).sort_values(["date", "ticker"])


def daily_return_panel(features: pd.DataFrame) -> pd.DataFrame:
    """Dates x tickers of daily returns, for the optimiser's risk model."""
    return features.pivot_table(index="date", columns="ticker", values="return_1d").sort_index()


def run_backtest(
    predictions: pd.DataFrame,
    config: BacktestConfig | None = None,
    risk_free: pd.Series | None = None,
    index_prices: pd.Series | None = None,
    daily_returns: pd.DataFrame | None = None,
) -> BacktestResult:
    config = config or BacktestConfig()
    usable = predictions.dropna(subset=[config.score_col, config.return_col])
    usable = smooth_scores(usable, config.score_col, config.smooth_days)
    dates = np.sort(usable["date"].unique())
    rebalance_dates = dates[config.offset :: config.holding_days]
    if config.construction == "optimizer" and daily_returns is None:
        raise ValueError("The optimiser needs daily_returns for its risk model (see daily_return_panel).")

    previous = {name: pd.Series(dtype=float) for name in STRATEGIES}
    rows = []
    borrow = config.borrow_bps / 10_000 * config.holding_days / TRADING_DAYS_PER_YEAR
    for date in rebalance_dates:
        cross_section = usable[usable["date"] == date]
        if len(cross_section) < config.min_names:
            continue
        realised = cross_section.set_index("ticker")[config.return_col]
        unit_cost = _unit_costs(cross_section, config)
        if config.construction == "optimizer":
            history = daily_returns.loc[: pd.Timestamp(date)]
            targets = _optimize(cross_section, config, previous, history, unit_cost)
        else:
            targets = _target_weights(cross_section, config, previous)
        row = {"date": pd.Timestamp(date), "names": len(cross_section)}
        for name, weights in targets.items():
            gross = float((weights * realised.reindex(weights.index)).sum())
            turnover, cost = _trading_cost(weights, previous[name], unit_cost)
            if name == "long_short":
                cost += borrow * float(weights[weights < 0].abs().sum())
            # Holdings drift with returns during the period; the next rebalance trades back from the drifted weights.
            drifted = weights * (1 + realised.reindex(weights.index))
            previous[name] = drifted / drifted.abs().sum() * weights.abs().sum() if drifted.abs().sum() > 0 else weights
            row[f"{name}_gross"] = gross
            row[f"{name}_turnover"] = turnover
            row[f"{name}_cost"] = cost
            row[name] = gross - cost
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
            returns[f"{MARKET}_cost"] = 0.0
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
        stats.update(
            strategy=name,
            avg_turnover=returns[f"{name}_turnover"].iloc[1:].mean(),
            cost_drag=float(returns[f"{name}_cost"].mean() * periods_per_year),
            gross_total_return=float((1 + returns[f"{name}_gross"]).prod() - 1),
        )
        summary_rows.append(stats)
    summary = pd.DataFrame(summary_rows).set_index("strategy")
    for benchmark, column in [("benchmark", "information_ratio"), (MARKET, "information_ratio_vs_sp500")]:
        if benchmark in returns.columns:
            active = returns["long_only"] - returns[benchmark]
            summary.loc["long_only", column] = active.mean() / active.std() * np.sqrt(periods_per_year) if len(active) > 1 and active.std() > 0 else float("nan")
            summary.loc["long_only", "excess_cagr_vs_" + ("equal_weight" if benchmark == "benchmark" else "sp500")] = summary.loc["long_only", "cagr"] - summary.loc[benchmark, "cagr"]
    for key, value in ic_summary(ic, step=1).items():  # rebalance dates are already non-overlapping
        summary.loc["long_short", key] = value
    return BacktestResult(config, returns, summary.reset_index(), quantile_returns(usable, rebalance_dates, config))


def run_staggered(
    predictions: pd.DataFrame,
    config: BacktestConfig,
    risk_free: pd.Series | None = None,
    index_prices: pd.Series | None = None,
    daily_returns: pd.DataFrame | None = None,
    offsets=None,
) -> pd.DataFrame:
    """The same backtest started on each session of the holding period (Jegadeesh-Titman staggering).

    A weekly strategy that rebalances on Mondays and one that rebalances on Wednesdays are equally valid;
    the spread of results across start days shows how much of a backtest is the luck of the calendar.
    Returns one row per strategy with the mean, minimum and maximum CAGR and Sharpe across offsets.
    """
    offsets = list(offsets) if offsets is not None else list(range(config.holding_days))
    frames = []
    for offset in offsets:
        result = run_backtest(predictions, replace(config, offset=offset), risk_free, index_prices, daily_returns)
        frames.append(result.summary.assign(offset=offset))
    table = pd.concat(frames, ignore_index=True)
    return (
        table.groupby("strategy", sort=False)
        .agg(offsets=("offset", "nunique"), cagr_mean=("cagr", "mean"), cagr_min=("cagr", "min"), cagr_max=("cagr", "max"),
             sharpe_mean=("sharpe", "mean"), sharpe_min=("sharpe", "min"), sharpe_max=("sharpe", "max"))
        .reset_index()
    )


def cost_sensitivity(result: BacktestResult, levels=(0, 5, 10, 20, 40)) -> tuple[pd.DataFrame, dict]:
    """Net CAGR and Sharpe at flat costs of `levels` bps per unit traded, and the break-even cost.

    Break-even: the cost per unit traded at which the strategy's gross edge is used up. For long-only, the
    edge is its gross return above the equal-weight benchmark; for long-short, its gross return.
    """
    returns = result.returns
    periods_per_year = TRADING_DAYS_PER_YEAR / result.config.holding_days
    rows = []
    for name in ["long_only", "long_short"]:
        for level in levels:
            net = returns[f"{name}_gross"] - returns[f"{name}_turnover"] * level / 10_000
            cash = returns["cash"] if "cash" in returns.columns and name != "long_short" else None
            stats = performance(net, periods_per_year, cash)
            rows.append({"strategy": name, "cost_bps": level, "cagr": stats.get("cagr"), "sharpe": stats.get("sharpe")})
    breakeven = {}
    active_turnover = (returns["long_only_turnover"] - returns["benchmark_turnover"]).mean()
    if active_turnover > 0:
        breakeven["long_only_vs_equal_weight_bps"] = float((returns["long_only_gross"] - returns["benchmark_gross"]).mean() / active_turnover * 10_000)
    if returns["long_short_turnover"].mean() > 0:
        breakeven["long_short_bps"] = float(returns["long_short_gross"].mean() / returns["long_short_turnover"].mean() * 10_000)
    return pd.DataFrame(rows), breakeven


def config_dict(config: BacktestConfig) -> dict:
    return asdict(config)
