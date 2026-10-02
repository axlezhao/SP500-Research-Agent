from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import EXECUTION_LAG_DAYS, FEATURES_PATH, HORIZON_DAYS, NEWS_PATH, TRADING_DAYS_PER_YEAR


POSITIVE_WORDS = {"beat", "beats", "strong", "growth", "raises", "expand", "expands", "upgrade", "bullish", "record"}
NEGATIVE_WORDS = {"miss", "misses", "weak", "pressure", "contract", "contracts", "downgrade", "bearish", "lawsuit", "decline"}
SENTIMENT_LABELS = {"positive": 1.0, "bullish": 1.0, "neutral": 0.0, "negative": -1.0, "bearish": -1.0}

# Display fields for briefs. From the Kaggle files they are a single current snapshot (as of the download
# date), so they are never model inputs. With live SEC data the same names hold point-in-time values,
# and the model uses their cross-sectional ranks instead (see POINT_IN_TIME_RATIOS).
SNAPSHOT_FIELDS = ["market_cap", "pe_ratio", "revenue", "profit_margin", "debt_to_equity", "liabilities_to_equity", "roe"]
POINT_IN_TIME_RATIOS = ["earnings_yield", "sales_yield", "book_to_market", "profit_margin", "roe", "revenue_growth_yoy", "market_cap"]
# Fundamentals older than this (e.g. a company that stopped filing) are treated as unknown.
MAX_FUNDAMENTALS_AGE_DAYS = 550
# A price-to-book outside this range almost always means a share count that doesn't match the traded
# share class (e.g. Berkshire reports Class A shares only); valuation ratios are dropped for such rows.
PLAUSIBLE_PRICE_TO_BOOK = (0.05, 200.0)


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


def build_research_features(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    news: pd.DataFrame,
    pit_fundamentals: pd.DataFrame | None = None,
    macro: pd.DataFrame | None = None,
    membership: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per ticker and trading day.

    Rows in the last HORIZON_DAYS of each ticker have no target yet (target_up_5d is NaN):
    they are excluded from training but are exactly the rows used for current predictions.

    Optional live inputs: point-in-time SEC fundamentals, FRED macro series, and index membership
    (adds `in_index`, used to train and trade only on stocks that were members at the time).
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
    prices["ma_gap_50d"] = prices["close"] / by_ticker.transform(lambda close: close.rolling(50).mean()) - 1
    if "volume" in prices.columns:
        volume = pd.to_numeric(prices["volume"], errors="coerce")
        prices["volume_ratio_20d"] = volume / volume.groupby(prices["ticker"]).transform(lambda v: v.rolling(20).mean())
    prices["future_return_5d"] = by_ticker.shift(-HORIZON_DAYS) / prices["close"] - 1
    # The return a backtest can actually earn: enter EXECUTION_LAG_DAYS after the signal, hold HORIZON_DAYS.
    prices["tradable_return_5d"] = (
        by_ticker.shift(-(HORIZON_DAYS + EXECUTION_LAG_DAYS)) / by_ticker.shift(-EXECUTION_LAG_DAYS) - 1
    )
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
    if membership is not None:
        from .sources.wikipedia import membership_flags

        features["in_index"] = membership_flags(features, membership)
    if pit_fundamentals is not None and not pit_fundamentals.empty:
        features = add_point_in_time_fundamentals(features, pit_fundamentals)
    if macro is not None and not macro.empty:
        features = add_macro_features(features, macro)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna(subset=["return_5d", "return_20d", "volatility_20d"]).reset_index(drop=True)
    return add_cross_sectional_features(features)


def add_cross_sectional_features(features: pd.DataFrame) -> pd.DataFrame:
    """Compare each stock with the rest of the market on the same date.

    Percentile ranks remove market-wide moves, so the model learns which stocks are relatively
    strong rather than whether the whole market went up. Sector comes from the fundamentals
    snapshot; sector membership rarely changes, so this is a mild approximation.
    """
    features = features.copy()
    # Compare against the investable universe of the day: index members when membership is known.
    universe = features[features["in_index"]] if "in_index" in features.columns else features
    columns = ["return_20d", "momentum_60d", "volatility_20d"]
    if "fundamentals_as_of" in features.columns:  # only point-in-time fundamentals, never the static snapshot
        columns += [col for col in POINT_IN_TIME_RATIOS if col in features.columns]
    # Relative target: beat the median member's forward return that day (unknown for the last HORIZON_DAYS rows).
    median_forward = universe.groupby("date")["future_return_5d"].transform("median").reindex(features.index)
    beat = (features["future_return_5d"] > median_forward).astype(float)
    features["target_beat_median_5d"] = beat.where(features["future_return_5d"].notna() & median_forward.notna())
    by_date = universe.groupby("date")
    for col in columns:
        features[f"{col}_xs_rank"] = by_date[col].rank(pct=True).reindex(features.index)
    if "sector" in features.columns:
        sector_median = universe.groupby(["date", "sector"])["return_20d"].transform("median").reindex(features.index)
        features["return_20d_vs_sector"] = features["return_20d"] - sector_median
    return features


def add_point_in_time_fundamentals(features: pd.DataFrame, pit: pd.DataFrame, max_age_days: int = MAX_FUNDAMENTALS_AGE_DAYS) -> pd.DataFrame:
    """Join the fundamentals known on each date and derive valuation and quality ratios.

    A filing becomes usable the day after its filing date. Market cap uses the as-traded price
    (`raw_close`, splits undone) times the share count reported at the time.
    """
    pit = pit.copy()
    pit["available_date"] = pd.to_datetime(pit["available_date"]).astype("datetime64[ns]")
    features = features.assign(date=pd.to_datetime(features["date"]).astype("datetime64[ns]"))
    keep = ["ticker", "available_date", "revenue_ttm", "net_income_ttm", "revenue_growth_yoy", "equity", "liabilities", "shares_outstanding"]
    merged = pd.merge_asof(
        features.sort_values("date"),
        pit[keep].sort_values("available_date"),
        left_on="date", right_on="available_date", by="ticker",
        direction="backward", allow_exact_matches=False,
    ).sort_values(["ticker", "date"]).reset_index(drop=True)
    stale = (merged["date"] - merged["available_date"]).dt.days > max_age_days
    merged.loc[stale, keep[1:]] = np.nan

    price = merged["raw_close"] if "raw_close" in merged.columns else merged["close"]
    equity = merged["equity"].where(merged["equity"] > 0)
    market_cap = price * merged["shares_outstanding"]
    price_to_book = market_cap / equity
    implausible = (price_to_book < PLAUSIBLE_PRICE_TO_BOOK[0]) | (price_to_book > PLAUSIBLE_PRICE_TO_BOOK[1])
    market_cap = market_cap.mask(implausible)
    net_income, revenue = merged["net_income_ttm"], merged["revenue_ttm"]
    merged = merged.assign(
        fundamentals_as_of=merged["available_date"],
        market_cap=market_cap,
        earnings_yield=net_income / market_cap,
        sales_yield=revenue / market_cap,
        book_to_market=equity / market_cap,
        pe_ratio=(market_cap / net_income).where(net_income > 0),
        revenue=revenue,
        profit_margin=(net_income / revenue).where(revenue > 0),
        roe=net_income / equity,
        liabilities_to_equity=merged["liabilities"] / equity,
    )
    return merged.drop(columns=["available_date", "revenue_ttm", "equity", "liabilities", "shares_outstanding"])


def add_macro_features(features: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Market-wide conditions known before each date: VIX level and 20-day change, rates and the yield curve."""
    macro = macro.sort_values("date").ffill()
    macro = macro.assign(
        date=pd.to_datetime(macro["date"]).astype("datetime64[ns]"),
        vix_change_20d=macro["vix"] / macro["vix"].shift(20) - 1,
    ).rename(columns={"date": "macro_date"})
    features = features.assign(date=pd.to_datetime(features["date"]).astype("datetime64[ns]"))
    merged = pd.merge_asof(
        features.sort_values("date"), macro.sort_values("macro_date"),
        left_on="date", right_on="macro_date", direction="backward", allow_exact_matches=False,
    )
    return merged.drop(columns="macro_date").sort_values(["ticker", "date"]).reset_index(drop=True)


def save_features(features: pd.DataFrame, path: Path = FEATURES_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(path, index=False)
    return path


def load_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run python run_pipeline.py first.")
    return pd.read_parquet(path)


def save_news(news: pd.DataFrame, path: Path = NEWS_PATH) -> Path:
    """Keep headlines so the agent can quote recent news, not just the aggregated sentiment."""
    columns = [col for col in ["ticker", "date", "title", "summary", "sentiment", "source", "url"] if col in news.columns]
    news = news[columns].copy()
    news["sentiment_score"] = _sentiment_scores(news)
    path.parent.mkdir(parents=True, exist_ok=True)
    news.sort_values(["ticker", "date"]).to_parquet(path, index=False)
    return path


def load_news_headlines(path: Path = NEWS_PATH) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["ticker", "date", "title", "sentiment_score"])
