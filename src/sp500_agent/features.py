from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEATURES_PATH, HORIZON_DAYS, TRADING_DAYS_PER_YEAR


POSITIVE_WORDS = {"beat", "beats", "strong", "growth", "raises", "expand", "expands", "upgrade", "bullish", "record"}
NEGATIVE_WORDS = {"miss", "misses", "weak", "pressure", "contract", "contracts", "downgrade", "bearish", "lawsuit", "decline"}
SENTIMENT_LABELS = {"positive": 1.0, "bullish": 1.0, "neutral": 0.0, "negative": -1.0, "bearish": -1.0}

# Fundamentals in the dataset are a single current snapshot, so they describe each company as of
# the download date, not as of each historical row. They are shown in briefs but not used for training.
SNAPSHOT_FIELDS = ["market_cap", "pe_ratio", "revenue", "profit_margin", "debt_to_equity", "roe"]


def _lexicon_sentiment(text: str) -> float:
    words = str(text).lower().replace(",", " ").replace(".", " ").split()
    pos = sum(word in POSITIVE_WORDS for word in words)
    neg = sum(word in NEGATIVE_WORDS for word in words)
    return 0.0 if pos == neg == 0 else (pos - neg) / (pos + neg)


def _sentiment_scores(news: pd.DataFrame) -> pd.Series:
    if "sentiment" in news.columns:
        numeric = pd.to_numeric(news["sentiment"], errors="coerce")
        if numeric.notna().mean() > 0.5:
            return numeric.clip(-1, 1).fillna(0.0)
        return news["sentiment"].astype(str).str.strip().str.lower().map(SENTIMENT_LABELS).fillna(0.0)
    text = news["title"].astype(str)
    if "summary" in news.columns:
        text = text + " " + news["summary"].astype(str)
    return text.map(_lexicon_sentiment)


def prepare_news_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    """Daily news count and summed sentiment per ticker and calendar date."""
    news = news.copy()
    news["sentiment_score"] = _sentiment_scores(news)
    news["date"] = pd.to_datetime(news["date"])
    return (
        news.groupby(["ticker", "date"], as_index=False)
        .agg(news_count=("title", "count"), sentiment_sum=("sentiment_score", "sum"))
        .sort_values(["ticker", "date"])
    )


def align_news_to_trading_days(daily_news: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Move news from weekends and holidays onto that ticker's next trading day."""
    if daily_news.empty:
        return daily_news.assign(date=pd.Series(dtype="datetime64[ns]"))
    sessions = prices[["ticker", "date"]].drop_duplicates().rename(columns={"date": "session"})
    sessions["session"] = sessions["session"].astype("datetime64[ns]")
    daily_news = daily_news.assign(date=daily_news["date"].astype("datetime64[ns]"))
    aligned = pd.merge_asof(
        daily_news.sort_values("date"),
        sessions.sort_values("session"),
        left_on="date",
        right_on="session",
        by="ticker",
        direction="forward",
    )
    # News after the last price date has no session to attach to yet.
    aligned = aligned.dropna(subset=["session"])
    return (
        aligned.groupby(["ticker", "session"], as_index=False)[["news_count", "sentiment_sum"]]
        .sum()
        .rename(columns={"session": "date"})
    )


def build_research_features(prices: pd.DataFrame, fundamentals: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    """One row per ticker and trading day.

    Rows in the last HORIZON_DAYS of each ticker have no target yet (target_up_5d is NaN):
    they are excluded from training but are exactly the rows used for current predictions.
    """
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"]).astype("datetime64[ns]")
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    by_ticker = prices.groupby("ticker")["close"]

    prices["return_1d"] = prices["close"] / by_ticker.shift(1) - 1
    prices["return_5d"] = prices["close"] / by_ticker.shift(5) - 1
    prices["return_20d"] = prices["close"] / by_ticker.shift(20) - 1
    prices["momentum_60d"] = prices["close"] / by_ticker.shift(60) - 1
    prices["volatility_20d"] = prices.groupby("ticker")["return_1d"].transform(
        lambda returns: returns.rolling(20).std()
    ) * np.sqrt(TRADING_DAYS_PER_YEAR)
    prices["future_return_5d"] = by_ticker.shift(-HORIZON_DAYS) / prices["close"] - 1
    prices["target_up_5d"] = (prices["future_return_5d"] > 0).astype(float).where(prices["future_return_5d"].notna())

    sentiment = align_news_to_trading_days(prepare_news_sentiment(news), prices)
    prices = prices.merge(sentiment, on=["ticker", "date"], how="left")
    prices["news_count"] = prices["news_count"].fillna(0)
    prices["sentiment_sum"] = prices["sentiment_sum"].fillna(0)
    by_ticker_news = prices.groupby("ticker")
    prices["news_count_20d"] = by_ticker_news["news_count"].transform(lambda s: s.rolling(20, min_periods=1).sum())
    sentiment_sum_20d = by_ticker_news["sentiment_sum"].transform(lambda s: s.rolling(20, min_periods=1).sum())
    # Average over articles only; with no news in the window the sentiment is unknown, not neutral.
    prices["sentiment_20d"] = (sentiment_sum_20d / prices["news_count_20d"]).where(prices["news_count_20d"] > 0)

    fundamentals = fundamentals.copy()
    fundamentals["ticker"] = fundamentals["ticker"].astype(str).str.upper()
    fundamentals = fundamentals.drop(columns=[col for col in fundamentals.columns if col in prices.columns and col != "ticker"])
    features = prices.merge(fundamentals, on="ticker", how="left")
    for col in SNAPSHOT_FIELDS:
        if col in features.columns:
            features[col] = pd.to_numeric(features[col], errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    return features.dropna(subset=["return_5d", "return_20d", "volatility_20d"]).reset_index(drop=True)


def save_features(features: pd.DataFrame, path: Path = FEATURES_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(path, index=False)
    return path


def load_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run python run_pipeline.py first.")
    return pd.read_parquet(path)
