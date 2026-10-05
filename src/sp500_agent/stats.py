"""Statistics for honest inference on noisy, autocorrelated return series.

- Information coefficients (daily rank correlations between a score and forward returns).
- Newey-West standard errors: overlapping 5-day returns on consecutive days share 4 days, so their
  ICs are autocorrelated and plain t-stats overstate significance.
- HAC regressions for factor attribution.
- The stationary bootstrap (Politis and Romano) for confidence intervals on CAGR, Sharpe and alpha that
  respect volatility clustering.
- The probabilistic and deflated Sharpe ratios (Bailey and Lopez de Prado), which ask whether the best
  of N tried strategies could have scored this well by luck.
- The probability of backtest overfitting via combinatorially symmetric cross-validation (CSCV).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329


def information_coefficients(df: pd.DataFrame, score_col: str, return_col: str = "future_return", min_names: int = 5) -> pd.Series:
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


def newey_west_variance(x: np.ndarray, lags: int) -> float:
    """Long-run variance of a series with Bartlett weights."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    variance = x @ x / n
    for lag in range(1, min(lags, n - 1) + 1):
        variance += 2 * (1 - lag / (lags + 1)) * (x[lag:] @ x[:-lag]) / n
    return float(variance)


def newey_west_tstat(series, lags: int) -> float:
    """t-stat of the mean with Newey-West standard errors (lags = overlap of the observations)."""
    x = pd.Series(series).dropna().to_numpy(dtype=float)
    if len(x) < 3:
        return float("nan")
    variance = newey_west_variance(x, max(int(lags), 0))
    return float(x.mean() / np.sqrt(variance / len(x))) if variance > 0 else float("nan")


def default_lags(n: int) -> int:
    """Newey and West's rule of thumb: floor(4 (n/100)^(2/9))."""
    return int(np.floor(4 * (n / 100) ** (2 / 9))) if n > 0 else 0


def hac_regression(y: np.ndarray, X: np.ndarray, lags: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS coefficients with plain and Newey-West (HAC) standard errors. X must include the constant.

    Returns (coefficients, OLS standard errors, HAC standard errors).
    """
    n, k = X.shape
    lags = default_lags(n) if lags is None else lags
    xtx_inv = np.linalg.inv(X.T @ X)
    coef = xtx_inv @ X.T @ y
    resid = y - X @ coef
    sigma2 = resid @ resid / (n - k)
    se_ols = np.sqrt(np.diag(sigma2 * xtx_inv))
    scores = X * resid[:, None]
    meat = scores.T @ scores
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1 - lag / (lags + 1)
        gamma = scores[lag:].T @ scores[:-lag]
        meat += weight * (gamma + gamma.T)
    cov = xtx_inv @ meat @ xtx_inv * n / (n - k)
    return coef, se_ols, np.sqrt(np.clip(np.diag(cov), 0, None))


def stationary_bootstrap_indices(n: int, reps: int, mean_block: float, seed: int = 42) -> np.ndarray:
    """(reps x n) resampling indices: blocks of random (geometric) length starting at random points, wrapping around."""
    rng = np.random.default_rng(seed)
    p = 1 / max(mean_block, 1.0)
    idx = np.empty((reps, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, reps)
    jumps = rng.random((reps, n)) < p
    starts = rng.integers(0, n, (reps, n))
    for t in range(1, n):
        idx[:, t] = np.where(jumps[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def _cagr(paths: np.ndarray, periods_per_year: float) -> np.ndarray:
    growth = np.prod(1 + paths, axis=1)
    out = np.full(len(paths), -1.0)
    positive = growth > 0
    out[positive] = growth[positive] ** (periods_per_year / paths.shape[1]) - 1
    return out


def _sharpe(paths: np.ndarray, periods_per_year: float) -> np.ndarray:
    std = paths.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(std > 0, paths.mean(axis=1) / std * np.sqrt(periods_per_year), np.nan)


def bootstrap_performance(returns: pd.DataFrame, strategies: list[str], periods_per_year: float, cash: pd.Series | None = None,
                          reps: int = 2000, mean_block: float | None = None, seed: int = 42, no_cash: tuple[str, ...] = ("long_short",)) -> pd.DataFrame:
    """95% stationary-bootstrap intervals for each strategy's CAGR and Sharpe, and the share of resamples with Sharpe <= 0.

    Strategies are resampled on the same dates, so their joint behaviour is kept.
    """
    n = len(returns)
    if n < 10:
        return pd.DataFrame()
    mean_block = mean_block or max(2.0, n ** (1 / 3))
    idx = stationary_bootstrap_indices(n, reps, mean_block, seed)
    rows = []
    for name in strategies:
        if name not in returns.columns:
            continue
        r = returns[name].to_numpy(dtype=float)
        excess = r - (cash.to_numpy(dtype=float) if cash is not None and name not in no_cash else 0.0)
        if np.isnan(r).any():
            continue
        cagr, sharpe = _cagr(r[idx], periods_per_year), _sharpe(excess[idx], periods_per_year)
        rows.append({
            "strategy": name,
            "cagr_low": float(np.nanpercentile(cagr, 2.5)), "cagr_high": float(np.nanpercentile(cagr, 97.5)),
            "sharpe_low": float(np.nanpercentile(sharpe, 2.5)), "sharpe_high": float(np.nanpercentile(sharpe, 97.5)),
            "prob_sharpe_not_positive": float(np.nanmean(sharpe <= 0)),
        })
    return pd.DataFrame(rows)


def bootstrap_alpha(y: pd.Series, factors: pd.DataFrame, periods_per_year: float, reps: int = 2000, mean_block: float | None = None, seed: int = 42) -> dict:
    """95% stationary-bootstrap interval for the annualised regression alpha (returns and factors resampled together)."""
    usable = [c for c in factors.columns if factors[c].notna().mean() > 0.5]  # skip factors the data can't build
    data = pd.concat([y.rename("y"), factors[usable]], axis=1).dropna()
    columns = [c for c in usable if data[c].std() > 0]
    n = len(data)
    if n < len(columns) + 10:
        return {}
    X = np.column_stack([np.ones(n), data[columns].to_numpy()])
    target = data["y"].to_numpy()
    idx = stationary_bootstrap_indices(n, reps, mean_block or max(2.0, n ** (1 / 3)), seed)
    alphas = np.empty(reps)
    for i, rows in enumerate(idx):
        coef, *_ = np.linalg.lstsq(X[rows], target[rows], rcond=None)
        alphas[i] = coef[0] * periods_per_year
    return {"alpha_low": float(np.percentile(alphas, 2.5)), "alpha_high": float(np.percentile(alphas, 97.5)), "prob_alpha_not_positive": float(np.mean(alphas <= 0))}


def probabilistic_sharpe(sr: float, n: int, skew: float, kurtosis: float, benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > benchmark) given the observed per-period Sharpe, sample length and the
    returns' skewness and (non-excess) kurtosis (Bailey and Lopez de Prado, 2012)."""
    denominator = 1 - skew * sr + (kurtosis - 1) / 4 * sr**2
    if n < 3 or denominator <= 0 or not np.isfinite(sr):
        return float("nan")
    return float(norm.cdf((sr - benchmark) * np.sqrt(n - 1) / np.sqrt(denominator)))


def expected_max_sharpe(trial_sharpes, n_trials: int | None = None) -> float:
    """Expected maximum per-period Sharpe among N unskilled trials with the observed dispersion of Sharpes."""
    trial_sharpes = np.asarray([s for s in trial_sharpes if np.isfinite(s)], dtype=float)
    n = n_trials or len(trial_sharpes)
    if n < 2 or len(trial_sharpes) < 2:
        return 0.0
    spread = trial_sharpes.std(ddof=1)
    return float(spread * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n) + EULER_GAMMA * norm.ppf(1 - 1 / (n * np.e))))


def deflated_sharpe(returns: pd.Series, trial_sharpes, n_trials: int | None = None) -> dict:
    """Deflated Sharpe ratio: the probabilistic Sharpe against the best Sharpe that luck alone would produce
    after `n_trials` tries. Inputs are per-period Sharpe ratios. Above 0.95 is the usual bar."""
    r = pd.Series(returns).dropna()
    if len(r) < 10 or r.std() == 0:
        return {}
    sr = float(r.mean() / r.std())
    threshold = expected_max_sharpe(trial_sharpes, n_trials)
    skew, kurt = float(r.skew()), float(r.kurt() + 3)
    return {
        "sharpe_per_period": sr,
        "trials": int(n_trials or len(list(trial_sharpes))),
        "expected_max_sharpe_per_period": threshold,
        "probabilistic_sharpe": probabilistic_sharpe(sr, len(r), skew, kurt, 0.0),
        "deflated_sharpe": probabilistic_sharpe(sr, len(r), skew, kurt, threshold),
    }


def probability_of_backtest_overfitting(performance: pd.DataFrame, splits: int = 16) -> dict:
    """PBO via combinatorially symmetric cross-validation (Bailey, Borwein, Lopez de Prado and Zhu, 2017).

    `performance` holds one column of per-period returns per strategy tried, on common dates. The dates are
    cut into `splits` blocks; for every way of choosing half the blocks as in-sample, the strategy with the
    best in-sample Sharpe is located in the out-of-sample ranking. PBO is the share of splits where it lands
    in the bottom half: the chance that picking the best backtest picks a below-median strategy.
    """
    data = performance.dropna()
    n_periods, n_strategies = data.shape
    splits = min(splits, n_periods // 4) // 2 * 2
    if n_strategies < 2 or splits < 4:
        return {}
    values = data.to_numpy(dtype=float)
    blocks = np.array_split(np.arange(n_periods), splits)
    sums = np.array([values[b].sum(axis=0) for b in blocks])
    squares = np.array([(values[b] ** 2).sum(axis=0) for b in blocks])
    counts = np.array([len(b) for b in blocks], dtype=float)

    def sharpe(selected) -> np.ndarray:
        total, sq, count = sums[selected].sum(axis=0), squares[selected].sum(axis=0), counts[selected].sum()
        mean = total / count
        var = (sq - count * mean**2) / (count - 1)
        return mean / np.sqrt(np.where(var > 0, var, np.nan))

    logits = []
    everything = set(range(splits))
    for chosen in combinations(range(splits), splits // 2):
        in_sample = list(chosen)
        out_sample = sorted(everything - set(chosen))
        best = int(np.nanargmax(sharpe(in_sample)))
        oos = sharpe(out_sample)
        rank = (np.argsort(np.argsort(oos))[best] + 1) / (n_strategies + 1)  # relative rank in (0, 1)
        logits.append(np.log(rank / (1 - rank)))
    logits = np.array(logits)
    return {"pbo": float(np.mean(logits <= 0)), "combinations": len(logits), "strategies": n_strategies, "splits": splits}
