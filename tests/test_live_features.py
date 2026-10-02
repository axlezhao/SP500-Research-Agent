"""Point-in-time rules for live data: nothing may be known before it was public."""

import numpy as np
import pandas as pd
import pytest

from sp500_agent.backtest import BacktestConfig, run_backtest
from sp500_agent.features import add_macro_features, add_point_in_time_fundamentals, build_research_features
from sp500_agent.model import MAX_MISSING_SHARE, available_features, score_latest, train_final_model
from sp500_agent.validation import walk_forward


def _frame(dates, ticker="A", close=100.0, raw_close=None):
    return pd.DataFrame({"ticker": ticker, "date": pd.to_datetime(dates), "close": close, "raw_close": raw_close if raw_close is not None else close})


def _filing(filed, **values):
    base = {"ticker": "A", "available_date": pd.Timestamp(filed), "period_end": pd.Timestamp(filed) - pd.Timedelta(days=30),
            "revenue_ttm": 1000.0, "net_income_ttm": 100.0, "revenue_growth_yoy": 0.1, "equity": 500.0, "liabilities": 250.0, "shares_outstanding": 10.0}
    return {**base, **values}


def test_filing_is_usable_only_from_the_next_day():
    features = _frame(["2024-02-14", "2024-02-15", "2024-02-16"])
    pit = pd.DataFrame([_filing("2024-02-15")])
    out = add_point_in_time_fundamentals(features, pit).set_index("date")
    assert pd.isna(out.loc["2024-02-15", "earnings_yield"])  # filing day: not yet known at the close
    assert out.loc["2024-02-16", "earnings_yield"] == pytest.approx(100 / (100 * 10))
    assert out.loc["2024-02-16", "pe_ratio"] == pytest.approx(10.0)
    assert out.loc["2024-02-16", "liabilities_to_equity"] == pytest.approx(0.5)


def test_stale_fundamentals_are_dropped():
    features = _frame(["2024-03-01", "2025-12-01"])
    out = add_point_in_time_fundamentals(features, pd.DataFrame([_filing("2024-02-15")])).set_index("date")
    assert out.loc["2024-03-01", "fundamentals_as_of"] == pd.Timestamp("2024-02-15")
    assert pd.isna(out.loc["2025-12-01", "market_cap"]) and pd.isna(out.loc["2025-12-01", "fundamentals_as_of"])


def test_market_cap_uses_as_traded_price_and_rejects_mismatched_share_counts():
    # Split-adjusted close 25, as-traded 100: market cap must use 100 x shares.
    out = add_point_in_time_fundamentals(_frame(["2024-03-01"], close=25.0, raw_close=100.0), pd.DataFrame([_filing("2024-02-15")]))
    assert out["market_cap"].iloc[0] == 1000.0
    # A share count from a different share class implies an absurd price-to-book: ratios are dropped.
    odd = add_point_in_time_fundamentals(_frame(["2024-03-01"]), pd.DataFrame([_filing("2024-02-15", shares_outstanding=0.0001)]))
    assert pd.isna(odd["market_cap"].iloc[0]) and pd.isna(odd["earnings_yield"].iloc[0])
    assert odd["profit_margin"].iloc[0] == pytest.approx(0.1)  # ratios not involving price are kept


def test_macro_values_are_lagged_one_day():
    features = _frame(["2024-01-03", "2024-01-04"])
    macro = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]), "vix": [10.0, 20.0, 30.0], "tbill_3m": 5.0, "treasury_10y": 4.0, "term_spread": -1.0})
    out = add_macro_features(features, macro).set_index("date")
    assert out.loc["2024-01-03", "vix"] == 10.0 and out.loc["2024-01-04", "vix"] == 20.0


def test_live_features_flag_membership_and_rank_only_members(live_world, live_features):
    joined = live_world.membership.set_index("ticker").loc["T13", "start"]
    left = live_world.membership.set_index("ticker").loc["T12", "end"]
    t13 = live_features[live_features["ticker"] == "T13"]
    t12 = live_features[live_features["ticker"] == "T12"]
    assert not t13.loc[t13["date"] < joined, "in_index"].any() and t13.loc[t13["date"] >= joined, "in_index"].all()
    assert t12.loc[t12["date"] < left, "in_index"].all() and not t12.loc[t12["date"] >= left, "in_index"].any()
    outside = live_features[~live_features["in_index"]]
    assert outside["return_20d_xs_rank"].isna().all() and outside["earnings_yield_xs_rank"].isna().all()
    inside = live_features[live_features["in_index"]]
    assert inside["return_20d_xs_rank"].between(0, 1).all()
    assert {"earnings_yield_xs_rank", "book_to_market_xs_rank", "market_cap_xs_rank", "vix", "term_spread"} <= set(inside.columns)


def test_kaggle_snapshot_fundamentals_are_never_ranked(sample_features):
    assert "market_cap" in sample_features.columns  # the static snapshot is there for display
    assert not any(col.startswith(("market_cap_", "earnings_yield", "profit_margin_")) for col in sample_features.columns)


def test_training_scoring_and_validation_use_index_members_only(live_features):
    bundle = train_final_model(live_features, "logistic_regression", model_path=None)
    assert "earnings_yield_xs_rank" in bundle["numeric"] and "vix" in bundle["numeric"]
    scored = score_latest(live_features, bundle)
    assert "T12" not in set(scored["ticker"]) and "T13" in set(scored["ticker"])  # T12 left the index, T13 joined
    predictions = walk_forward(live_features, "logistic_regression", n_splits=3).predictions
    members = live_features.loc[live_features["in_index"], ["ticker", "date"]]
    assert len(predictions.merge(members, on=["ticker", "date"])) == len(predictions)


def test_available_features_skips_sparse_and_constant_columns():
    df = pd.DataFrame({
        "return_5d": np.arange(10.0),
        "vix": [1.0] * 10,  # constant
        "sentiment_20d": [np.nan] * int(10 * MAX_MISSING_SHARE + 1) + [0.1] * (10 - int(10 * MAX_MISSING_SHARE + 1)),  # mostly missing
    })
    numeric, _ = available_features(df)
    assert numeric == ["return_5d"]


def test_backtest_sharpe_is_in_excess_of_cash():
    dates = pd.bdate_range("2024-01-01", periods=50)
    preds = pd.DataFrame({"date": np.repeat(dates, 10), "ticker": [f"S{i}" for i in range(10)] * 50})
    preds["tradable_return_5d"] = 0.002 + np.random.default_rng(0).normal(0, 0.01, len(preds))
    preds["probability"] = np.random.default_rng(1).uniform(size=len(preds))
    plain = run_backtest(preds, BacktestConfig(cost_bps=0)).summary.set_index("strategy")
    rate = pd.Series(0.05, index=dates)
    with_cash = run_backtest(preds, BacktestConfig(cost_bps=0), risk_free=rate)
    stats = with_cash.summary.set_index("strategy")
    assert with_cash.returns["cash"].iloc[0] == pytest.approx(0.05 * 5 / 252)
    assert stats.loc["benchmark", "sharpe"] < plain.loc["benchmark", "sharpe"]
    assert stats.loc["long_short", "sharpe"] == pytest.approx(plain.loc["long_short", "sharpe"])  # self-financing


def test_relative_target_splits_each_day_at_the_median(live_features):
    members = live_features[live_features["in_index"] & live_features["target_beat_median_5d"].notna()]
    share = members.groupby("date")["target_beat_median_5d"].mean()
    assert share.between(0.4, 0.6).all()  # half the members beat the median (ties aside)
    outside = live_features[~live_features["in_index"]]
    assert outside["target_beat_median_5d"].isna().all()
    assert live_features.groupby("ticker").tail(5)["target_beat_median_5d"].isna().all()  # future unknown


def test_stance_follows_rank():
    from sp500_agent.formatting import stance

    assert [stance(r, 10) for r in [1, 2, 3, 8, 9, 10]] == ["constructive", "constructive", "neutral", "neutral", "cautious", "cautious"]
    assert stance(1, 1) == "constructive" and stance(1, 0) == "neutral"


def test_suspect_recent_moves_are_left_out_of_the_ranking(live_features):
    from sp500_agent.research_tools import ResearchData, ResearchToolkit, ToolInputError

    features = live_features.copy()
    last = features.index[(features["ticker"] == "T05") & (features["date"] == features["date"].max())]
    features.loc[last, "return_1d"] = -0.84
    bundle = train_final_model(features, "logistic_regression", model_path=None)
    scored = score_latest(features, bundle)
    assert "T05" not in set(scored["ticker"]) and "-84%" in scored.attrs["excluded"]["T05"]
    assert scored["rank"].tolist() == list(range(1, len(scored) + 1))
    toolkit = ResearchToolkit(ResearchData(features, bundle))
    with pytest.raises(ToolInputError, match="left out of today's ranking"):
        toolkit.call("stock_snapshot", {"ticker": "T05"})
    assert toolkit.call("price_history", {"ticker": "T05"})["ticker"] == "T05"
    assert "T05" in toolkit.call("dataset_overview", {})["left_out_of_ranking"]
