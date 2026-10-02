import numpy as np
import pandas as pd

from sp500_agent.config import HORIZON_DAYS
from sp500_agent.data_loader import load_news
from sp500_agent.features import build_research_features


def _prices(dates, ticker="AAA"):
    return pd.DataFrame({"ticker": ticker, "date": pd.to_datetime(dates), "close": np.linspace(100, 130, len(dates))})


def test_latest_rows_are_kept_without_a_target(sample_raw_dir, sample_features):
    raw = pd.read_csv(sample_raw_dir / "sample_prices.csv", parse_dates=["date"])
    assert sample_features["date"].max() == raw["date"].max()
    for _, group in sample_features.groupby("ticker"):
        targets = group.sort_values("date")["target_up_5d"]
        assert targets.tail(HORIZON_DAYS).isna().all()
        assert targets.iloc[:-HORIZON_DAYS].notna().all()


def test_momentum_is_not_a_copy_of_the_20_day_return(sample_features):
    both = sample_features.dropna(subset=["momentum_60d"])
    assert not np.allclose(both["momentum_60d"], both["return_20d"])


def test_volatility_is_annualised(sample_features):
    # Sample daily volatility is 1.2%-3%, i.e. roughly 19%-48% a year.
    assert sample_features["volatility_20d"].median() > 0.1


def test_sentiment_averages_only_days_with_news():
    dates = pd.bdate_range("2024-01-01", periods=30)
    news = pd.DataFrame({"ticker": "AAA", "date": [dates[22]], "title": ["x"], "sentiment": ["positive"]})
    features = build_research_features(_prices(dates), pd.DataFrame({"ticker": ["AAA"]}), news).set_index("date")
    assert np.isnan(features.loc[dates[21], "sentiment_20d"])  # no news yet: unknown, not neutral
    assert features.loc[dates[29], "sentiment_20d"] == 1.0  # one positive article, not diluted by empty days
    assert features.loc[dates[29], "news_count_20d"] == 1


def test_weekend_news_rolls_to_next_trading_day():
    dates = pd.bdate_range("2024-01-01", periods=40)
    saturday = pd.Timestamp("2024-02-03")
    news = pd.DataFrame({"ticker": "AAA", "date": [saturday], "title": ["x"], "sentiment": ["negative"]})
    features = build_research_features(_prices(dates), pd.DataFrame({"ticker": ["AAA"]}), news).set_index("date")
    assert features.loc[pd.Timestamp("2024-02-02"), "news_count_20d"] == 0  # Friday
    assert features.loc[pd.Timestamp("2024-02-05"), "news_count_20d"] == 1  # Monday


def test_numeric_sentiment_scores_are_used(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=30)
    news = pd.DataFrame({"ticker": "AAA", "date": [dates[25]] * 2, "title": ["x", "y"], "sentiment": [0.5, -0.1]})
    features = build_research_features(_prices(dates), pd.DataFrame({"ticker": ["AAA"]}), news).set_index("date")
    assert np.isclose(features.loc[dates[25], "sentiment_20d"], 0.2)


def test_plain_dates_are_not_mistaken_for_utc_offsets():
    from sp500_agent.data_loader import _to_market_time

    parsed = _to_market_time(pd.Series(["2024-03-05", "2024-03-06T09:00:00Z"]))
    assert parsed.dt.date.astype(str).tolist() == ["2024-03-05", "2024-03-06"]


def test_news_timestamps_use_market_time_and_after_close_moves_forward(tmp_path):
    pd.DataFrame(
        {
            "ticker": ["aaa", "aaa", "aaa"],
            "date": ["2024-03-05T14:00:00-05:00", "2024-03-05T21:30:00+00:00", "2024-03-06 16:30:00"],
            "title": ["during session", "after close in New York", "after close, naive"],
        }
    ).to_csv(tmp_path / "news.csv", index=False)
    news = load_news(tmp_path)
    assert news["date"].dt.date.astype(str).tolist() == ["2024-03-05", "2024-03-06", "2024-03-07"]
