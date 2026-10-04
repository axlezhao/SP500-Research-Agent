import numpy as np
import pandas as pd
import pytest

from sp500_agent.backtest import BacktestConfig, max_drawdown, run_backtest


def _predictions(n_dates=60, n_tickers=20, signal=1.0, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    df = pd.DataFrame({"date": np.repeat(dates, n_tickers), "ticker": [f"T{i}" for i in range(n_tickers)] * n_dates})
    df["tradable_return"] = rng.normal(0.002, 0.03, len(df))
    df["probability"] = 0.5 + signal * df["tradable_return"] + (1 - signal) * rng.normal(0, 0.03, len(df))
    df["future_return_5d"] = df["tradable_return"]
    return df


def test_perfect_signal_beats_benchmark_and_sorts_quintiles():
    result = run_backtest(_predictions(signal=1.0), BacktestConfig(cost_bps=0))
    stats = result.summary.set_index("strategy")
    assert (result.returns["long_short"] > 0).all()
    assert stats.loc["long_only", "total_return"] > stats.loc["benchmark", "total_return"]
    assert result.quantile_returns["mean_return"].is_monotonic_increasing
    assert stats.loc["long_short", "ic_mean"] == pytest.approx(1.0)


def test_rebalances_do_not_overlap():
    result = run_backtest(_predictions(n_dates=60))
    dates = pd.bdate_range("2024-01-01", periods=60)
    assert result.returns["date"].tolist() == list(dates[::5])


def test_costs_reduce_net_but_not_gross_returns():
    free = run_backtest(_predictions(signal=0.0), BacktestConfig(cost_bps=0)).returns
    costly = run_backtest(_predictions(signal=0.0), BacktestConfig(cost_bps=50)).returns
    assert np.allclose(free["long_short_gross"], costly["long_short_gross"])
    assert (costly["long_short"] < free["long_short"]).all()
    # First rebalance builds the whole book: gross exposure 2 for long-short.
    assert costly["long_short_turnover"].iloc[0] == pytest.approx(2.0)


def test_benchmark_is_equal_weight_average():
    preds = _predictions()
    result = run_backtest(preds, BacktestConfig(cost_bps=0))
    first = result.returns.iloc[0]
    expected = preds[preds["date"] == first["date"]]["tradable_return"].mean()
    assert first["benchmark"] == pytest.approx(expected)


def test_max_drawdown():
    assert max_drawdown(pd.Series([0.1, -0.5, 0.2])) == pytest.approx(-0.5)
    assert max_drawdown(pd.Series([0.1, 0.1])) == 0.0
