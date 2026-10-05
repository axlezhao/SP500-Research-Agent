"""Portfolio construction: buffers, smoothing, per-stock costs, borrow fees, the optimiser and staggered starts."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from sp500_agent.backtest import BacktestConfig, _buffered, cost_sensitivity, run_backtest, run_staggered, smooth_scores


def _panel(n_dates=80, n_tickers=40, persistence=0.9, seed=0):
    """Scores that drift slowly (like a real signal), with returns weakly tied to them, plus exposures."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    tickers = [f"S{i:02d}" for i in range(n_tickers)]
    score = rng.normal(size=n_tickers)
    rows = []
    for date in dates:
        score = persistence * score + np.sqrt(1 - persistence**2) * rng.normal(size=n_tickers)
        for i, t in enumerate(tickers):
            rows.append({"date": date, "ticker": t, "probability": 0.5 + 0.1 * score[i], "tradable_return": 0.002 * score[i] + rng.normal(0, 0.02),
                         "volatility_20d": 0.25 + 0.01 * (i % 5), "beta_252": 0.6 + 0.02 * i, "market_cap_z": rng.normal(), "book_to_market_z": rng.normal(),
                         "cs_spread_21d": 0.0005 + 0.0001 * (i % 10), "sector": "AB"[i % 2]})
    preds = pd.DataFrame(rows)
    daily = preds.pivot(index="date", columns="ticker", values="tradable_return") / 5
    return preds, daily


def test_buffer_keeps_holdings_until_they_leave_the_wider_band():
    order = pd.Index(list("ABCDEFGHIJ"))
    assert list(_buffered(order, 2, 4, held={"D", "Z"})) == ["A", "D"]  # D still in top 4 stays, best new name fills
    assert list(_buffered(order, 2, 4, held={"E"})) == ["A", "B"]  # E dropped out of the top 4
    assert list(_buffered(order, 2, 2, held={"D"})) == ["A", "B"]  # no buffer


def test_buffer_and_smoothing_cut_turnover():
    preds, _ = _panel()
    plain = run_backtest(preds, BacktestConfig(cost_bps=10)).summary.set_index("strategy")
    buffered = run_backtest(preds, BacktestConfig(cost_bps=10, exit_quantile=0.4)).summary.set_index("strategy")
    smoothed = run_backtest(preds, BacktestConfig(cost_bps=10, smooth_days=3)).summary.set_index("strategy")
    assert buffered.loc["long_only", "avg_turnover"] < plain.loc["long_only", "avg_turnover"]
    assert smoothed.loc["long_only", "avg_turnover"] < plain.loc["long_only", "avg_turnover"]
    assert buffered.loc["long_only", "cost_drag"] < plain.loc["long_only", "cost_drag"]


def test_smoothing_uses_only_past_scores():
    preds = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=4), "ticker": "A", "probability": [0.1, 0.2, 0.9, 0.4]})
    out = smooth_scores(preds, "probability", 2)
    assert out["probability"].round(6).tolist() == [0.1, 0.15, 0.55, 0.65]


def test_spread_costs_and_borrow_fees():
    preds, _ = _panel(n_dates=30)
    from sp500_agent.backtest import _unit_costs

    flat = run_backtest(preds, BacktestConfig(cost_bps=10)).returns
    day = preds[preds["date"] == preds["date"].iloc[0]]
    costs = _unit_costs(day, BacktestConfig(cost_model="spread", cost_bps=10))
    # The median stock pays the configured cost; wider-spread stocks pay more, in proportion.
    assert costs.median() == pytest.approx(0.001)
    spreads = day.set_index("ticker")["cs_spread_21d"]
    assert costs[spreads.idxmax()] > costs[spreads.idxmin()]
    assert costs[spreads.idxmax()] / costs[spreads.idxmin()] == pytest.approx(spreads.max() / spreads.min())
    borrow = run_backtest(preds, BacktestConfig(cost_bps=10, borrow_bps=100)).returns
    assert np.allclose(borrow["long_short_cost"] - flat["long_short_cost"], 0.01 * 5 / 252 * 1.0)  # 100 bps a year on a 1.0 short book
    assert np.allclose(borrow["long_only_cost"], flat["long_only_cost"])


def test_missing_spreads_fall_back_to_the_flat_cost():
    preds, _ = _panel(n_dates=15)
    without = preds.drop(columns="cs_spread_21d")
    assert np.allclose(run_backtest(without, BacktestConfig(cost_model="spread", cost_bps=10)).returns["long_only_cost"], run_backtest(without, BacktestConfig(cost_bps=10)).returns["long_only_cost"])


def test_optimiser_respects_budget_and_neutrality():
    from sp500_agent.backtest import _optimize, _unit_costs

    preds, daily = _panel(n_dates=140)
    config = BacktestConfig(construction="optimizer", max_weight=0.1)
    date = preds["date"].unique()[-1]
    cross_section = preds[preds["date"] == date]
    weights = _optimize(cross_section, config, {}, daily.loc[:date], _unit_costs(cross_section, config))
    long_only, long_short = weights["long_only"], weights["long_short"]
    assert long_only.sum() == pytest.approx(1.0) and (long_only >= 0).all() and long_only.max() <= 0.1 + 1e-6
    assert long_short.sum() == pytest.approx(0.0, abs=1e-6) and long_short.abs().sum() <= 3.0 + 1e-6
    exposures = cross_section.set_index("ticker").loc[long_short.index]
    for column in ["beta_252", "market_cap_z", "book_to_market_z"]:
        assert (exposures[column] * long_short).sum() == pytest.approx(0.0, abs=1e-5)
    # Higher scores get more weight on average.
    scores = cross_section.set_index("ticker")["probability"]
    assert np.corrcoef(scores.reindex(long_short.index), long_short)[0, 1] > 0.3


def test_optimiser_backtest_trades_less_with_higher_costs():
    preds, daily = _panel(n_dates=90)
    cheap = run_backtest(preds, BacktestConfig(construction="optimizer", cost_bps=1), daily_returns=daily).summary.set_index("strategy")
    dear = run_backtest(preds, BacktestConfig(construction="optimizer", cost_bps=50), daily_returns=daily).summary.set_index("strategy")
    assert dear.loc["long_short", "avg_turnover"] < cheap.loc["long_short", "avg_turnover"]
    # At ordinary costs it keeps rebalancing rather than freezing its first portfolio.
    usual = run_backtest(preds, BacktestConfig(construction="optimizer", cost_bps=10), daily_returns=daily).summary.set_index("strategy")
    assert usual.loc["long_only", "avg_turnover"] > 0.05
    with pytest.raises(ValueError, match="daily_returns"):
        run_backtest(preds, BacktestConfig(construction="optimizer"))


def test_staggered_starts_cover_every_offset():
    preds, _ = _panel(n_dates=60)
    table = run_staggered(preds, BacktestConfig(cost_bps=10)).set_index("strategy")
    assert table.loc["long_only", "offsets"] == 5
    assert table.loc["long_only", "cagr_min"] <= table.loc["long_only", "cagr_mean"] <= table.loc["long_only", "cagr_max"]
    first = run_backtest(preds, replace(BacktestConfig(), offset=2)).returns
    assert first["date"].iloc[0] == preds["date"].unique()[2]


def test_cost_sensitivity_and_breakeven():
    preds, _ = _panel()
    result = run_backtest(preds, BacktestConfig(cost_bps=10))
    table, breakeven = cost_sensitivity(result, levels=(0, 10, 1000))
    long_short = table[table["strategy"] == "long_short"].set_index("cost_bps")
    gross = (1 + result.returns["long_short_gross"]).prod() ** (50.4 / len(result.returns)) - 1
    assert long_short.loc[0, "cagr"] == pytest.approx(gross, rel=1e-3)
    assert long_short.loc[0, "cagr"] > long_short.loc[10, "cagr"] > long_short.loc[1000, "cagr"]
    assert breakeven["long_short_bps"] > 0  # the panel has a real (if small) signal
