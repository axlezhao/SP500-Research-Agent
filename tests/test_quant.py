"""Research setups (horizon, peer group), sector-neutral portfolios, the S&P 500 benchmark and factor attribution."""

import numpy as np
import pandas as pd
import pytest

from sp500_agent.attribution import regress
from sp500_agent.backtest import BacktestConfig, index_period_returns, run_backtest
from sp500_agent.config import ResearchSpec
from sp500_agent.research import backtest_config_for, run_experiments
from sp500_agent.validation import walk_forward


def test_spec_names_columns_and_rejects_unknown_setups():
    spec = ResearchSpec(21, "sector")
    assert (spec.target_column, spec.future_column, spec.tradable_column) == ("target_beat_sector_21d", "future_return_21d", "tradable_return_21d")
    assert "its sector" in spec.description and spec.label == "21-day, vs. own sector"
    with pytest.raises(ValueError):
        ResearchSpec(10, "market")
    with pytest.raises(ValueError):
        ResearchSpec(5, "industry")
    with pytest.raises(ValueError):
        ResearchSpec(5, "absolute", "rank")  # a rank is always relative to peers
    ranked = ResearchSpec(21, "sector", "rank")
    assert (ranked.target_column, ranked.binary_column) == ("target_rank_sector_21d", "target_beat_sector_21d")
    assert backtest_config_for(spec).neutralize == "sector" and backtest_config_for(spec).holding_days == 21
    assert backtest_config_for(ResearchSpec(5, "market"), BacktestConfig(cost_bps=3)).cost_bps == 3


def test_monthly_targets_and_sector_relative_split(live_features):
    assert live_features.groupby("ticker").tail(21)["target_beat_median_21d"].isna().all()  # future month unknown
    members = live_features[live_features["in_index"]]
    sector = members.dropna(subset=["target_beat_sector_21d"])
    share = sector.groupby(["date", "sector"])["target_beat_sector_21d"].mean()
    assert share.between(0.3, 0.7).all()
    # momentum_12_1 skips the most recent month: it is unknown for the first year of each stock.
    assert live_features.groupby("ticker").head(200)["momentum_12_1"].isna().all()


def test_monthly_walk_forward_uses_a_21_session_gap(live_features):
    result = walk_forward(live_features, "ridge", n_splits=3, spec=ResearchSpec(21, "market", "rank"))
    dates = sorted(live_features["date"].unique())
    position = {d: i for i, d in enumerate(dates)}
    for fold in result.fold_metrics.itertuples():
        assert position[fold.test_start] - position[fold.train_end] == 22
    assert {"target", "future_return", "tradable_return"} <= set(result.predictions.columns)


def _cross_section(n_dates=30, sectors=("A", "B", "C"), per_sector=10, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    rows = []
    for date in dates:
        for s in sectors:
            for i in range(per_sector):
                rows.append({"date": date, "ticker": f"{s}{i}", "sector": s})
    df = pd.DataFrame(rows)
    # Sector A is strong and the model loves it: an overall ranking would only buy sector A.
    df["probability"] = rng.uniform(0.4, 0.6, len(df)) + (df["sector"] == "A") * 0.3
    df["tradable_return"] = rng.normal(0, 0.02, len(df))
    return df


def test_sector_neutral_portfolio_buys_in_every_sector():
    preds = _cross_section()
    overall = run_backtest(preds, BacktestConfig(holding_days=5, cost_bps=0))
    neutral = run_backtest(preds, BacktestConfig(holding_days=5, cost_bps=0, neutralize="sector"))
    date = preds["date"].iloc[0]
    from sp500_agent.backtest import _target_weights

    weights = _target_weights(preds[preds["date"] == date], BacktestConfig(neutralize="sector"))
    longs = weights["long_only"].index
    assert {t[0] for t in longs} == {"A", "B", "C"} and len(longs) == 6  # top 2 of 10 in each sector
    assert abs(weights["long_short"].sum()) < 1e-12  # dollar neutral
    overall_longs = _target_weights(preds[preds["date"] == date], BacktestConfig())["long_only"].index
    assert {t[0] for t in overall_longs} == {"A"}
    assert len(neutral.returns) == len(overall.returns)


def test_index_returns_cover_the_same_window_as_the_strategies():
    sessions = pd.bdate_range("2024-01-01", periods=40)
    prices = pd.Series(100 * 1.01 ** np.arange(40), index=sessions)
    returns = index_period_returns(prices, sessions[:3], holding_days=5, lag=1)
    assert returns.iloc[0] == pytest.approx(1.01**5 - 1)  # enter day 1, exit day 6
    assert np.isnan(index_period_returns(prices, sessions[-3:], holding_days=5).iloc[-1])  # window runs past the data

    preds = _cross_section(n_dates=30)  # index prices run 10 sessions past the last signal, as in real data
    result = run_backtest(preds, BacktestConfig(holding_days=5, cost_bps=0), index_prices=prices)
    assert "sp500" in set(result.summary["strategy"])
    assert result.returns["sp500"].iloc[0] == pytest.approx(1.01**5 - 1)
    assert "information_ratio_vs_sp500" in result.summary.columns


def test_attribution_recovers_known_exposures():
    rng = np.random.default_rng(1)
    factors = pd.DataFrame({"MKT": rng.normal(0.004, 0.02, 300), "SMB": rng.normal(0, 0.01, 300), "HML": rng.normal(0, 0.01, 300)})
    y = 0.001 + 1.2 * factors["MKT"] - 0.5 * factors["HML"] + rng.normal(0, 0.002, 300)
    stats = regress(y, factors, periods_per_year=50)
    assert stats["beta_MKT"] == pytest.approx(1.2, abs=0.05)
    assert stats["beta_HML"] == pytest.approx(-0.5, abs=0.05)
    assert stats["beta_SMB"] == pytest.approx(0.0, abs=0.05)
    assert stats["alpha_annual"] == pytest.approx(0.05, abs=0.02) and stats["alpha_tstat"] > 2
    assert stats["r_squared"] > 0.9


def test_experiments_run_each_setup_on_live_like_data(live_features):
    rate = live_features.groupby("date")["tbill_3m"].first() / 100
    specs = (ResearchSpec(5, "market"), ResearchSpec(21, "sector", "rank"))
    table, trials = run_experiments(live_features, specs, 3, 50_000, None, rate, None, production=ResearchSpec(5, "market"))
    assert list(table["setup"]) == ["5-day, vs. all members", "21-day, vs. own sector, ranked"]
    assert table["model"].tolist() == ["logistic_regression", "ridge"]
    assert table["production"].tolist() == [True, False]
    assert table.loc[1, "portfolio"] == "sector-neutral" and table["auc"].notna().all()
    assert len(trials) == 2 and all(len(series) > 0 for series in trials.values())


def test_attribution_drops_factors_the_data_cannot_build():
    rng = np.random.default_rng(2)
    factors = pd.DataFrame({"MKT": rng.normal(0, 0.02, 200), "SMB": np.nan, "MOM": rng.normal(0, 0.01, 200)})
    stats = regress(0.9 * factors["MKT"] + rng.normal(0, 0.002, 200), factors, periods_per_year=50)
    assert stats["beta_MKT"] == pytest.approx(0.9, abs=0.05) and "beta_SMB" not in stats
