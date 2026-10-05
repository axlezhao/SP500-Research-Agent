"""Where do the backtest returns come from?

Each strategy's per-period returns are regressed on factor returns built from the same universe and
rebalance dates (a home-made Carhart four-factor model):

- MKT: the S&P 500 (SPY) in excess of T-bills; the equal-weight universe when SPY isn't available.
- SMB: small minus big (bottom vs. top 30% by market cap).
- HML: value minus growth (top vs. bottom 30% by book-to-market).
- MOM: winners minus losers (top vs. bottom 30% by 12-1 month momentum).

The intercept (alpha) is the return the factors don't explain; a |t| above about 2 is the usual bar.
t-stats use Newey-West (HAC) standard errors, which allow for the volatility clustering and mild
autocorrelation of strategy returns; plain OLS t-stats are kept alongside for comparison.

`french_attribution` repeats the exercise with Kenneth French's published factors (Fama-French five,
momentum and short-term reversal), built from the whole US market by a third party: a check that the
home-made factors aren't flattering or hiding anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EXECUTION_LAG_DAYS, TRADING_DAYS_PER_YEAR
from .model import investable
from .stats import hac_regression

FACTOR_CHARACTERISTICS = {"SMB": ("market_cap", -1), "HML": ("book_to_market", 1), "MOM": ("momentum_12_1", 1)}
FACTOR_LABELS = {"MKT": "Market", "SMB": "Size (small minus big)", "HML": "Value (cheap minus expensive)", "MOM": "Momentum (winners minus losers)"}
SPLIT = 0.3
MIN_NAMES = 30


def factor_returns(features: pd.DataFrame, returns: pd.DataFrame, return_col: str) -> pd.DataFrame:
    """Long-short factor returns for each rebalance date in `returns`, from the investable universe of that date."""
    universe = investable(features)
    universe = universe[universe["date"].isin(returns["date"])]
    rows = []
    for date, group in universe.groupby("date"):
        row = {"date": date}
        for factor, (column, sign) in FACTOR_CHARACTERISTICS.items():
            data = group.dropna(subset=[column, return_col]) if column in group.columns else group.iloc[0:0]
            if len(data) < MIN_NAMES:
                row[factor] = np.nan
                continue
            ranked = data[column].rank(pct=True)
            high = data.loc[ranked >= 1 - SPLIT, return_col].mean()
            low = data.loc[ranked <= SPLIT, return_col].mean()
            row[factor] = sign * (high - low)
        rows.append(row)
    factors = pd.DataFrame(rows, columns=["date", *FACTOR_CHARACTERISTICS])
    cash = returns.set_index("date")["cash"] if "cash" in returns.columns else pd.Series(0.0, index=returns["date"])
    market = returns.set_index("date")["sp500"] if "sp500" in returns.columns else returns.set_index("date")["benchmark"]
    factors = factors.set_index("date").reindex(returns["date"])
    factors.insert(0, "MKT", (market - cash).reindex(factors.index))
    return factors.reset_index()


def regress(y: pd.Series, factors: pd.DataFrame, periods_per_year: float) -> dict:
    # A factor that can't be built from the data (e.g. size and value without SEC fundamentals) is left out
    # rather than emptying the regression.
    usable = [c for c in factors.columns if factors[c].notna().mean() > 0.5]
    data = pd.concat([y.rename("y"), factors[usable]], axis=1).dropna()
    columns = [c for c in usable if data[c].std() > 0]
    if len(data) < len(columns) + 5:
        return {}
    X = np.column_stack([np.ones(len(data)), data[columns].to_numpy()])
    target = data["y"].to_numpy()
    coef, se_ols, se = hac_regression(target, X)
    residuals = target - X @ coef
    ss_total = ((target - target.mean()) ** 2).sum()
    result = {
        "alpha_annual": coef[0] * periods_per_year,
        "alpha_tstat": coef[0] / se[0],
        "alpha_tstat_ols": coef[0] / se_ols[0],
        "r_squared": 1 - (residuals @ residuals) / ss_total if ss_total > 0 else np.nan,
        "periods": len(data),
    }
    for i, column in enumerate(columns, start=1):
        result[f"beta_{column}"] = coef[i]
        result[f"tstat_{column}"] = coef[i] / se[i]
    return result


def attribute(features: pd.DataFrame, backtest_returns: pd.DataFrame, holding_days: int, return_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(attribution table: one row per strategy, factor returns per rebalance)."""
    factors = factor_returns(features, backtest_returns, return_col)
    periods_per_year = TRADING_DAYS_PER_YEAR / holding_days
    cash = backtest_returns["cash"] if "cash" in backtest_returns.columns else 0.0
    factor_frame = factors.drop(columns="date").set_index(backtest_returns.index)
    rows = []
    for strategy in ["long_only", "long_short"]:
        # Long-only is judged on its return above cash; long-short is self-financing.
        y = backtest_returns[strategy] - (cash if strategy == "long_only" else 0.0)
        stats = regress(y, factor_frame, periods_per_year)
        if stats:
            rows.append({"strategy": strategy, **stats})
    return pd.DataFrame(rows), factors


def french_attribution(backtest_returns: pd.DataFrame, daily_factors: pd.DataFrame, holding_days: int, lag: int = EXECUTION_LAG_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(attribution table, factor returns per rebalance) on Kenneth French's factors.

    Daily factors are compounded over each holding window. Long-only is regressed in excess of French's
    T-bill rate; long-short, which is self-financing, as is. Periods after French's last update are left out.
    """
    from .sources.french import period_factor_returns

    period = period_factor_returns(daily_factors, backtest_returns["date"], holding_days, lag)
    factor_columns = [c for c in ["MKT", "SMB", "HML", "RMW", "CMA", "MOM", "STREV"] if c in period.columns]
    period = period.set_axis(backtest_returns.index)
    periods_per_year = TRADING_DAYS_PER_YEAR / holding_days
    rows = []
    for strategy in ["long_only", "long_short"]:
        y = backtest_returns[strategy] - (period["RF"] if strategy == "long_only" and "RF" in period else 0.0)
        stats = regress(y, period[factor_columns], periods_per_year)
        if stats:
            rows.append({"strategy": strategy, **stats})
    return pd.DataFrame(rows), period.assign(date=backtest_returns["date"].to_numpy())
