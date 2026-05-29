from __future__ import annotations

import numpy as np
import pandas as pd


POSITIVE_WORDS = {"beat", "beats", "strong", "growth", "raises", "expand", "expands", "upgrade", "bullish", "record"}
NEGATIVE_WORDS = {"miss", "misses", "weak", "pressure", "contract", "contracts", "downgrade", "bearish", "lawsuit", "decline"}


def _lexicon_sentiment(text: str) -> float:
    words = str(text).lower().replace(",", " ").replace(".", " ").split()
    pos = sum(word in POSITIVE_WORDS for word in words)
    neg = sum(word in NEGATIVE_WORDS for word in words)
    return 0.0 if pos == neg == 0 else (pos - neg) / (pos + neg)


def prepare_news_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    news = news.copy()
    if "sentiment" in news.columns:
        mapping = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        news["sentiment_score"] = news["sentiment"].astype(str).str.lower().map(mapping).fillna(0.0)
    else:
        text = news["title"].astype(str)
        if "summary" in news.columns:
            text = text + " " + news["summary"].astype(str)
        news["sentiment_score"] = text.map(_lexicon_sentiment)
    news["date"] = pd.to_datetime(news["date"])
    return news.groupby(["ticker", "date"], as_index=False).agg(news_count=("title", "count"), sentiment_score=("sentiment_score", "mean")).sort_values(["ticker", "date"])


def build_research_features(prices: pd.DataFrame, fundamentals: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy().sort_values(["ticker", "date"])
    prices["date"] = pd.to_datetime(prices["date"])
    prices["return_1d"] = prices.groupby("ticker")["close"].pct_change()
    prices["return_5d"] = prices.groupby("ticker")["close"].pct_change(5)
    prices["return_20d"] = prices.groupby("ticker")["close"].pct_change(20)
    prices["volatility_20d"] = prices.groupby("ticker")["return_1d"].rolling(20).std().reset_index(level=0, drop=True)
    prices["momentum_20d"] = prices.groupby("ticker")["close"].pct_change(20)
    prices["future_return_5d"] = prices.groupby("ticker")["close"].shift(-5) / prices["close"] - 1
    prices["target_up_5d"] = (prices["future_return_5d"] > 0).astype(int)
    sentiment = prepare_news_sentiment(news)
    prices = prices.merge(sentiment, on=["ticker", "date"], how="left")
    prices["news_count"] = prices["news_count"].fillna(0)
    prices["sentiment_score"] = prices["sentiment_score"].fillna(0)
    prices["sentiment_20d"] = prices.groupby("ticker")["sentiment_score"].rolling(20, min_periods=1).mean().reset_index(level=0, drop=True)
    prices["news_count_20d"] = prices.groupby("ticker")["news_count"].rolling(20, min_periods=1).sum().reset_index(level=0, drop=True)
    fundamentals = fundamentals.copy()
    fundamentals["ticker"] = fundamentals["ticker"].astype(str).str.upper()
    features = prices.merge(fundamentals, on="ticker", how="left")
    for col in ["market_cap", "pe_ratio", "revenue", "profit_margin", "debt_to_equity", "roe"]:
        if col in features.columns:
            features[col] = pd.to_numeric(features[col], errors="coerce")
    return features.replace([np.inf, -np.inf], np.nan).dropna(subset=["return_5d", "return_20d", "volatility_20d", "future_return_5d"])

