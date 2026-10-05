from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtri

from .config import EXECUTION_LAG_DAYS, FEATURES_PATH, HORIZONS, NEWS_PATH, TRADING_DAYS_PER_YEAR
from .sentiment import lexicon_score, score_texts

SENTIMENT_LABELS = {"positive": 1.0, "bullish": 1.0, "neutral": 0.0, "negative": -1.0, "bearish": -1.0}

# Display fields for briefs. From the Kaggle files they are a single current snapshot (as of the download
# date), so they are never model inputs. With live SEC data the same names hold point-in-time values,
# and the model uses their cross-sectional ranks instead (see POINT_IN_TIME_RATIOS).
SNAPSHOT_FIELDS = ["market_cap", "pe_ratio", "revenue", "profit_margin", "debt_to_equity", "liabilities_to_equity", "roe"]
POINT_IN_TIME_RATIOS = [
    "earnings_yield", "sales_yield", "book_to_market", "profit_margin", "roe", "revenue_growth_yoy", "market_cap",
    "gross_profitability", "accruals", "asset_growth", "net_issuance",
]
PRICE_RANKED = ["return_20d", "momentum_60d", "momentum_12_1", "volatility_20d"]
# Stock-level signals that are normalised within each date (rank, then mapped to a standard normal) and
# become model inputs as `<name>_z`. Fundamentals only qualify when they are point in time.
PRICE_SIGNALS = [
    "return_5d", "return_20d", "momentum_60d", "momentum_12_1", "volatility_20d", "ma_gap_50d", "volume_ratio_20d",
    "sentiment_20d", "news_count_20d",
    "return_5d_vs_industry", "return_20d_vs_industry", "industry_momentum",
    "beta_252", "idio_vol_63", "residual_momentum", "max_return_21", "high_52w",
]
EVENT_SIGNALS = ["sue", "ear", "insider_purchases_90d", "insider_net_value_90d"]
# Smallest group (all members, or one sector) a relative target is computed against.
MIN_PEER_GROUP = 4
# Smallest industry (SIC major group) used for industry-relative signals; smaller ones fall back to the sector.
MIN_INDUSTRY_NAMES = 5
# Fundamentals older than this (e.g. a company that stopped filing) are treated as unknown.
MAX_FUNDAMENTALS_AGE_DAYS = 550
# A price-to-book outside this range almost always means a share count that doesn't match the traded
# share class (e.g. Berkshire reports Class A shares only); valuation ratios are dropped for such rows.
PLAUSIBLE_PRICE_TO_BOOK = (0.05, 200.0)
# A date counts as a trading session when at least this share of the busiest day so far has a price.
MIN_SESSION_SHARE = 0.2
# Missing sessions inside a ticker's history are carried forward at most this many sessions.
MAX_FILL_SESSIONS = 5
# An earnings surprise stays a live signal this long after the release; an announcement return, this many sessions.
SUE_MAX_AGE_DAYS = 120
EAR_MAX_SESSIONS = 63
INSIDER_WINDOW_SESSIONS = 63
# A price history ending at least this many days before the data does has stopped (delisted or acquired);
# one that is merely a day or two stale hasn't, and its recent rows must stay unlabelled.
ENDED_AFTER_DAYS = 10
# Floor on annualised volatility when scaling forward returns, so near-constant prices don't dominate the ranks.
MIN_TARGET_VOL = 0.05


# -- news ----------------------------------------------------------------------------------------------
def _lexicon_sentiment(text: str) -> float:
    return lexicon_score(text)


def _sentiment_scores(news: pd.DataFrame) -> pd.Series:
    if "sentiment" in news.columns:
        numeric = pd.to_numeric(news["sentiment"], errors="coerce")
        if numeric.notna().mean() > 0.5:
            return numeric.clip(-1, 1).fillna(0.0)
        return news["sentiment"].astype(str).str.strip().str.lower().map(SENTIMENT_LABELS).fillna(0.0)
    text = news["title"].astype(str)
    if "summary" in news.columns:
        text = text + " " + news["summary"].fillna("").astype(str)
    return pd.Series(score_texts(text.tolist()), index=news.index, dtype=float)


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
    aligned = _to_next_session(daily_news, prices)
    return (
        aligned.groupby(["ticker", "session"], as_index=False)[["news_count", "sentiment_sum"]]
        .sum()
        .rename(columns={"session": "date"})
    )


def _to_next_session(events: pd.DataFrame, prices: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Attach each event to its ticker's first session on or after the event date (events after the last price are dropped)."""
    sessions = prices[["ticker", "date"]].drop_duplicates().rename(columns={"date": "session"})
    sessions["session"] = sessions["session"].astype("datetime64[ns]")
    events = events.assign(**{date_col: pd.to_datetime(events[date_col]).astype("datetime64[ns]")}).dropna(subset=[date_col])
    aligned = pd.merge_asof(
        events.sort_values(date_col), sessions.sort_values("session"),
        left_on=date_col, right_on="session", by="ticker", direction="forward",
    )
    return aligned.dropna(subset=["session"])


# -- price panel -----------------------------------------------------------------------------------------
def on_session_calendar(prices: pd.DataFrame) -> pd.DataFrame:
    """Give every ticker a row for each market session between its first and last price.

    Row-based shifts (shift(5) = "5 sessions ago") are only right when no sessions are missing. Missing
    days inside a ticker's history are inserted with the last close carried forward (at most
    MAX_FILL_SESSIONS) and flagged `_filled`; they are dropped again once the time-series features are built.
    """
    counts = prices.groupby("date")["ticker"].nunique().sort_index()
    # Compared with the busiest day *so far*, so the calendar on any date doesn't depend on later data.
    calendar = pd.DatetimeIndex(counts.index[counts >= MIN_SESSION_SHARE * counts.cummax()])
    span = prices.groupby("ticker")["date"].agg(["min", "max"])
    frames = []
    for ticker, (first, last) in span.iterrows():
        sessions = calendar[(calendar >= first) & (calendar <= last)]
        frames.append(pd.DataFrame({"ticker": ticker, "date": sessions}))
    grid = pd.concat(frames, ignore_index=True) if frames else prices[["ticker", "date"]].iloc[0:0]
    grid["date"] = grid["date"].astype("datetime64[ns]")
    merged = grid.merge(prices.assign(_filled=False), on=["ticker", "date"], how="left").sort_values(["ticker", "date"])
    merged["_filled"] = merged["_filled"].isna()
    merged["close"] = merged.groupby("ticker", sort=False)["close"].ffill(limit=MAX_FILL_SESSIONS)
    if "raw_close" in merged.columns:
        merged["raw_close"] = merged.groupby("ticker", sort=False)["raw_close"].ffill(limit=MAX_FILL_SESSIONS)
    if "split_factor" in merged.columns:
        merged["split_factor"] = merged.groupby("ticker", sort=False)["split_factor"].ffill()
    return merged.reset_index(drop=True)


def _grouped_rolling(frame: pd.DataFrame, columns, window: int, min_periods: int, how: str) -> pd.DataFrame | pd.Series:
    rolled = getattr(frame.groupby("ticker", sort=False)[columns].rolling(window, min_periods=min_periods), how)()
    return rolled.droplevel(0).reindex(frame.index)


def corwin_schultz_spread(prices: pd.DataFrame) -> pd.Series:
    """Daily bid-ask spread estimate from two days of highs and lows (Corwin and Schultz, 2012).

    The high-low range of a day reflects both volatility and the spread; volatility grows with the
    window but the spread doesn't, so comparing one-day and two-day ranges separates them. Highs and lows
    are first adjusted for overnight gaps, as the paper recommends. Negative estimates are set to zero.
    """
    by_ticker = prices.groupby("ticker", sort=False)
    high, low = prices["high"].astype(float), prices["low"].astype(float)
    prev_close = by_ticker["close"].shift(1)
    gap_up = (low - prev_close).clip(lower=0).fillna(0)
    gap_down = (prev_close - high).clip(lower=0).fillna(0)
    high_adj, low_adj = high - gap_up + gap_down, low - gap_up + gap_down
    high_prev, low_prev = by_ticker["high"].shift(1), by_ticker["low"].shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.log(high_prev / low_prev) ** 2 + np.log(high_adj / low_adj) ** 2
        gamma = np.log(np.maximum(high_prev, high_adj) / np.minimum(low_prev, low_adj)) ** 2
        k = 3 - 2 * np.sqrt(2)
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread.clip(lower=0).where(np.isfinite(spread))


def forward_returns(prices: pd.DataFrame, delisting_returns: dict | None = None) -> pd.DataFrame:
    """Forward and tradable returns for every horizon, keeping stocks that stop trading.

    A stock whose price history ends inside the holding window (acquired, delisted) is not dropped,
    which would require knowing on the signal date that it was about to disappear. Its return runs to the
    last available price, compounded with a delisting return: 0 for mergers (the last price is roughly
    the deal price) and about -30% for delistings for poor performance (Shumway, 1997), per
    `delisting_returns` ({ticker: return}). Rows near the end of the data, where the future is simply not
    known yet, stay NaN. A position that can't be entered (no price on the entry day) stays NaN too.
    """
    by_ticker = prices.groupby("ticker", sort=False)
    close = prices["close"]
    position = by_ticker.cumcount()
    length = by_ticker["close"].transform("size")
    ended = by_ticker["date"].transform("max") < prices["date"].max() - pd.Timedelta(days=ENDED_AFTER_DAYS)
    delisting = prices["ticker"].map(delisting_returns or {}).fillna(0.0).astype(float)
    exit_value = by_ticker["close"].transform("last") * (1 + delisting)

    def price_ahead(sessions: int) -> pd.Series:
        ahead = by_ticker["close"].shift(-sessions)
        beyond = ended & (position + sessions >= length)
        return ahead.where(~beyond, exit_value)

    out = {}
    for horizon in HORIZONS:
        future = price_ahead(horizon) / close - 1
        out[f"future_return_{horizon}d"] = future
        # The return a backtest can actually earn: enter EXECUTION_LAG_DAYS after the signal, hold `horizon` sessions.
        out[f"tradable_return_{horizon}d"] = price_ahead(horizon + EXECUTION_LAG_DAYS) / by_ticker["close"].shift(-EXECUTION_LAG_DAYS) - 1
        out[f"target_up_{horizon}d"] = (future > 0).astype(float).where(future.notna())
    return pd.DataFrame(out, index=prices.index)


def add_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Return, momentum, risk and liquidity signals from each ticker's own price history."""
    prices = prices.copy()
    by_ticker = prices.groupby("ticker", sort=False)["close"]
    prices["return_1d"] = prices["close"] / by_ticker.shift(1) - 1
    prices["return_5d"] = prices["close"] / by_ticker.shift(5) - 1
    prices["return_20d"] = prices["close"] / by_ticker.shift(20) - 1
    prices["momentum_60d"] = prices["close"] / by_ticker.shift(60) - 1
    # Classic 12-1 momentum: the return from 12 months ago to 1 month ago, skipping the reversal-prone last month.
    prices["momentum_12_1"] = by_ticker.shift(21) / by_ticker.shift(252) - 1
    prices["volatility_20d"] = _grouped_rolling(prices, "return_1d", 20, 20, "std") * np.sqrt(TRADING_DAYS_PER_YEAR)
    prices["ma_gap_50d"] = prices["close"] / _grouped_rolling(prices, "close", 50, 50, "mean") - 1
    if "volume" in prices.columns:
        prices["volume"] = pd.to_numeric(prices["volume"], errors="coerce")
        prices["volume_ratio_20d"] = prices["volume"] / _grouped_rolling(prices, "volume", 20, 20, "mean")

    # Market: the equal-weighted average daily return of all stocks with a real price that day.
    real = prices["return_1d"].where(~prices["_filled"]).clip(-0.5, 0.5)
    prices["market_return_1d"] = real.groupby(prices["date"]).transform("mean")
    pair = pd.DataFrame({"ticker": prices["ticker"], "r": real, "m": prices["market_return_1d"]})
    pair.loc[pair["r"].isna() | pair["m"].isna(), ["r", "m"]] = np.nan
    pair["rm"], pair["mm"] = pair["r"] * pair["m"], pair["m"] ** 2
    means = _grouped_rolling(pair, ["r", "m", "rm", "mm"], 252, 126, "mean")
    prices["beta_252"] = (means["rm"] - means["r"] * means["m"]) / (means["mm"] - means["m"] ** 2)
    # Residual return: what the market doesn't explain, using the beta known the day before.
    residual = pair["r"] - prices.groupby("ticker", sort=False)["beta_252"].shift(1) * pair["m"]
    residuals = pd.DataFrame({"ticker": prices["ticker"], "e": residual})
    prices["idio_vol_63"] = _grouped_rolling(residuals, "e", 63, 42, "std") * np.sqrt(TRADING_DAYS_PER_YEAR)
    # Residual momentum (Blitz, Huij and Martens): 12-1 month momentum in residual returns, per unit of residual risk.
    sums = _grouped_rolling(residuals, "e", 231, 126, "sum")
    stds = _grouped_rolling(residuals, "e", 231, 126, "std")
    stats = pd.DataFrame({"ticker": prices["ticker"], "s": sums, "sd": stds})
    shifted = stats.groupby("ticker", sort=False)[["s", "sd"]].shift(21)
    prices["residual_momentum"] = shifted["s"] / (shifted["sd"] * np.sqrt(231))
    # Lottery-like stocks (Bali, Cakici and Whitelaw): the largest daily return of the past month.
    prices["max_return_21"] = _grouped_rolling(prices.assign(r=real), "r", 21, 15, "max")
    # Anchoring (George and Hwang): price relative to its 52-week high.
    prices["high_52w"] = prices["close"] / _grouped_rolling(prices, "close", 252, 126, "max")
    if {"high", "low"} <= set(prices.columns):
        prices["cs_spread"] = corwin_schultz_spread(prices)
        prices["cs_spread_21d"] = _grouped_rolling(prices, "cs_spread", 21, 10, "mean")
        prices = prices.drop(columns="cs_spread")
    return prices


# -- assembly -------------------------------------------------------------------------------------------
def build_research_features(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    news: pd.DataFrame,
    pit_fundamentals: pd.DataFrame | None = None,
    macro: pd.DataFrame | None = None,
    membership: pd.DataFrame | None = None,
    eps: pd.DataFrame | None = None,
    earnings_dates: pd.DataFrame | None = None,
    insiders: pd.DataFrame | None = None,
    delisting_returns: dict | None = None,
) -> pd.DataFrame:
    """One row per ticker and trading day.

    Rows in the last `horizon` sessions of each ticker have no target for that horizon yet (NaN):
    they are excluded from training but are exactly the rows used for current predictions.

    Optional live inputs: point-in-time SEC fundamentals and quarterly EPS, earnings-release dates,
    insider trades, FRED macro series, index membership (adds `in_index`, used to train and trade only on
    stocks that were members at the time) and delisting returns for stocks whose prices end early.
    """
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"]).astype("datetime64[ns]")
    prices = prices.drop_duplicates(["ticker", "date"], keep="last")
    prices = add_price_features(on_session_calendar(prices))
    prices = pd.concat([prices, forward_returns(prices, delisting_returns)], axis=1)

    sentiment = align_news_to_trading_days(prepare_news_sentiment(news), prices)
    prices = prices.merge(sentiment, on=["ticker", "date"], how="left")
    prices["news_count"] = prices["news_count"].fillna(0)
    prices["sentiment_sum"] = prices["sentiment_sum"].fillna(0)
    prices["news_count_20d"] = _grouped_rolling(prices, "news_count", 20, 1, "sum")
    sentiment_sum_20d = _grouped_rolling(prices, "sentiment_sum", 20, 1, "sum")
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
    if (eps is not None and not eps.empty) or (earnings_dates is not None and not earnings_dates.empty):
        features = add_earnings_features(features, eps, earnings_dates)
    if insiders is not None and not insiders.empty:
        features = add_insider_features(features, insiders)
    if macro is not None and not macro.empty:
        features = add_macro_features(features, macro)
    features = features[~features["_filled"]].drop(columns="_filled")
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna(subset=["return_5d", "return_20d", "volatility_20d"]).reset_index(drop=True)
    return add_cross_sectional_features(features)


def _gaussian_ranks(frame: pd.DataFrame, columns: list[str], keys) -> pd.DataFrame:
    """Within each group (date): rank, scale to (0, 1), map to a standard normal. Robust to outliers and to
    regime changes in a signal's scale, so every input means "where does this stock sit today"."""
    grouped = frame.groupby(keys)[columns]
    ranks = grouped.rank(method="average")
    counts = grouped.transform("count")
    return pd.DataFrame(ndtri(((ranks - 0.5) / counts).to_numpy(dtype=float)), index=frame.index, columns=columns)


def add_cross_sectional_features(features: pd.DataFrame) -> pd.DataFrame:
    """Compare each stock with the rest of the market on the same date.

    Percentile ranks remove market-wide moves, so the model learns which stocks are relatively
    strong rather than whether the whole market went up. Every comparison is among the investable
    universe of the day (index members when membership is known).
    """
    features = features.copy()
    universe = features[features["in_index"]] if "in_index" in features.columns else features
    point_in_time = "fundamentals_as_of" in features.columns  # never rank the static Kaggle snapshot

    # Industry: the SIC major group when it has enough members that day, otherwise the sector.
    if "industry" in features.columns:
        size = universe.groupby(["date", "industry"])["ticker"].transform("count").reindex(features.index)
        group = features["industry"].where(size >= MIN_INDUSTRY_NAMES)
        features["industry_group"] = group.fillna(features["sector"]) if "sector" in features.columns else group
    elif "sector" in features.columns:
        features["industry_group"] = features["sector"]
    if "industry_group" in features.columns:
        universe = features[features["in_index"]] if "in_index" in features.columns else features
        by_industry = universe.groupby(["date", "industry_group"])
        # Short-term reversal within industry (Da, Liu and Schaumburg): raw reversal is diluted by industry momentum.
        for col in ["return_5d", "return_20d"]:
            features[f"{col}_vs_industry"] = features[col] - by_industry[col].transform("median").reindex(features.index)
        # Industry momentum (Moskowitz and Grinblatt).
        features["industry_momentum"] = by_industry["momentum_12_1"].transform("mean").reindex(features.index)
    if "sector" in features.columns:
        sector_median = universe.groupby(["date", "sector"])["return_20d"].transform("median").reindex(features.index)
        features["return_20d_vs_sector"] = features["return_20d"] - sector_median

    # Targets. Binary: beat the median member (or the sector median). Rank: percentile of the
    # volatility-scaled forward return among the same peers, centred so its mean is 0.5 each day.
    groups = {"market": ["date"], "sector": ["date", "sector"]} if "sector" in features.columns else {"market": ["date"]}
    vol = features["volatility_20d"].clip(lower=MIN_TARGET_VOL)
    for horizon in HORIZONS:
        future_col = f"future_return_{horizon}d"
        known = features[future_col].notna()
        scaled = (features[future_col] / vol).rename("scaled")
        for name, keys in groups.items():
            binary_name = "median" if name == "market" else "sector"
            grouped = universe.groupby(keys)[future_col]
            median_forward = grouped.transform("median").reindex(features.index)
            # A median of one or two stocks isn't a meaningful peer group (a stock can't beat itself).
            enough = grouped.transform("count").reindex(features.index) >= MIN_PEER_GROUP
            beat = (features[future_col] > median_forward).astype(float)
            features[f"target_beat_{binary_name}_{horizon}d"] = beat.where(known & median_forward.notna() & enough)
            ranked = pd.concat([universe[keys], scaled.reindex(universe.index)], axis=1)
            by_group = ranked.groupby(keys)["scaled"]
            percentile = ((by_group.rank(method="average") - 0.5) / by_group.transform("count")).reindex(features.index)
            features[f"target_rank_{name}_{horizon}d"] = percentile.where(known & enough)

    # Display ranks (0-1) for the dashboard, and the model's normalised inputs.
    universe = features[features["in_index"]] if "in_index" in features.columns else features
    by_date = universe.groupby("date")
    display = list(PRICE_RANKED) + ([col for col in POINT_IN_TIME_RATIOS if col in features.columns] if point_in_time else [])
    for col in display:
        features[f"{col}_xs_rank"] = by_date[col].rank(pct=True).reindex(features.index).astype("float32")
    signals = [col for col in PRICE_SIGNALS + EVENT_SIGNALS if col in features.columns]
    if point_in_time:
        signals += [col for col in POINT_IN_TIME_RATIOS if col in features.columns]
    signals = [col for col in signals if universe[col].notna().any()]
    if signals:
        z = _gaussian_ranks(universe, signals, "date").reindex(features.index).astype("float32")
        features[[f"{col}_z" for col in signals]] = z.to_numpy()
    # Market conditions can't rank stocks by themselves (they are the same for all on a date); as
    # interactions they let a linear model shift weight between signals across regimes.
    if "vix_regime" in features.columns:
        reversal = "return_5d_vs_industry_z" if "return_5d_vs_industry_z" in features.columns else "return_5d_z"
        if reversal in features.columns:
            features["reversal_x_vix"] = features[reversal] * features["vix_regime"]
        if "momentum_12_1_z" in features.columns:
            features["momentum_x_vix"] = features["momentum_12_1_z"] * features["vix_regime"]
    return features


def add_point_in_time_fundamentals(features: pd.DataFrame, pit: pd.DataFrame, max_age_days: int = MAX_FUNDAMENTALS_AGE_DAYS) -> pd.DataFrame:
    """Join the fundamentals known on each date and derive valuation, quality and investment ratios.

    A filing becomes usable the day after its filing date. Market cap uses the as-traded price
    (`raw_close`, splits undone) times the share count reported at the time.
    """
    pit = pit.copy()
    pit["available_date"] = pd.to_datetime(pit["available_date"]).astype("datetime64[ns]")
    features = features.assign(date=pd.to_datetime(features["date"]).astype("datetime64[ns]"))
    keep = ["ticker", "available_date", "revenue_ttm", "net_income_ttm", "revenue_growth_yoy", "equity", "liabilities", "shares_outstanding"]
    extra = [col for col in ["gross_profit_ttm", "operating_cash_flow_ttm", "assets"] if col in pit.columns]
    keep += extra
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
    if "assets" in merged.columns:
        assets = merged["assets"].where(merged["assets"] > 0)
        if "gross_profit_ttm" in merged.columns:
            # Gross profitability (Novy-Marx): gross profit over total assets.
            merged["gross_profitability"] = merged["gross_profit_ttm"] / assets
        if "operating_cash_flow_ttm" in merged.columns:
            # Accruals (Sloan): earnings not backed by operating cash flow, over total assets.
            merged["accruals"] = (net_income - merged["operating_cash_flow_ttm"]) / assets
        # Asset growth (Cooper, Gulen and Schill): total assets against a year (252 sessions) earlier.
        merged["asset_growth"] = assets / assets.groupby(merged["ticker"]).shift(TRADING_DAYS_PER_YEAR) - 1
    # Net share issuance (Pontiff and Woodgate): growth in split-adjusted shares outstanding over a year.
    split_factor = merged["split_factor"] if "split_factor" in merged.columns else 1.0
    adjusted_shares = merged["shares_outstanding"] * split_factor
    merged["net_issuance"] = adjusted_shares / adjusted_shares.groupby(merged["ticker"]).shift(TRADING_DAYS_PER_YEAR) - 1
    return merged.drop(columns=["available_date", "revenue_ttm", "equity", "liabilities", "shares_outstanding", *extra])


def standardized_surprises(eps: pd.DataFrame, earnings_dates: pd.DataFrame | None = None) -> pd.DataFrame:
    """Standardized unexpected earnings (SUE) per quarter, with the date the market learned it.

    SUE = (EPS - EPS four quarters earlier) / standard deviation of that seasonal change over the previous
    eight quarters (Foster, Olsen and Shevlin; the basis of post-earnings-announcement drift). The quarter
    becomes known at its earnings release (8-K item 2.02) when one falls between the quarter end and the
    10-Q/10-K filing; otherwise the day after the filing. Returns columns: ticker, available, sue.
    """
    q = eps.copy()
    q["period_end"] = pd.to_datetime(q["period_end"]).astype("datetime64[ns]")
    q["filed"] = pd.to_datetime(q["filed"]).astype("datetime64[ns]")
    q = q.dropna(subset=["period_end", "eps"]).sort_values(["ticker", "period_end"]).drop_duplicates(["ticker", "period_end"])
    previous = q[["ticker", "period_end", "eps"]].rename(columns={"period_end": "prev_end", "eps": "prev_eps"})
    q["target"] = q["period_end"] - pd.Timedelta(days=365)
    q = pd.merge_asof(
        q.sort_values("target"), previous.sort_values("prev_end"),
        left_on="target", right_on="prev_end", by="ticker", direction="nearest", tolerance=pd.Timedelta(days=20),
    ).sort_values(["ticker", "period_end"]).reset_index(drop=True)
    q["surprise"] = q["eps"] - q["prev_eps"]
    q["scale"] = q.groupby("ticker")["surprise"].transform(lambda s: s.shift(1).rolling(8, min_periods=4).std())
    q["sue"] = (q["surprise"] / q["scale"].where(q["scale"] > 0)).clip(-10, 10)
    q["available"] = q["filed"] + pd.Timedelta(days=1)
    if earnings_dates is not None and not earnings_dates.empty:
        releases = earnings_dates[["ticker", "date"]].assign(date=lambda d: pd.to_datetime(d["date"]).astype("datetime64[ns]")).rename(columns={"date": "release"})
        q["after"] = q["period_end"] + pd.Timedelta(days=1)
        q = pd.merge_asof(q.sort_values("after"), releases.sort_values("release"), left_on="after", right_on="release", by="ticker", direction="forward")
        valid = q["release"].notna() & (q["release"] <= q["filed"] + pd.Timedelta(days=3))
        # The release date is the session the news first reached the market, so it is usable that day.
        q.loc[valid, "available"] = q.loc[valid, "release"]
    return q[["ticker", "available", "sue"]].dropna().sort_values(["ticker", "available"]).reset_index(drop=True)


def add_earnings_features(features: pd.DataFrame, eps: pd.DataFrame | None, earnings_dates: pd.DataFrame | None) -> pd.DataFrame:
    """Earnings surprise (SUE) and the stock's abnormal return around the release (EAR).

    EAR is the stock's return minus the market's over the session before, of and after the release
    (Chan, Jegadeesh and Lakonishok; Brandt et al.). It is known at the close of the session after the
    release and carried for EAR_MAX_SESSIONS sessions. Both signals drift: prices underreact to earnings news.
    """
    features = features.assign(date=pd.to_datetime(features["date"]).astype("datetime64[ns]")).sort_values(["ticker", "date"]).reset_index(drop=True)
    if eps is not None and not eps.empty:
        surprises = standardized_surprises(eps, earnings_dates)
        merged = pd.merge_asof(
            features[["ticker", "date"]].reset_index().sort_values("date"), surprises.sort_values("available"),
            left_on="date", right_on="available", by="ticker", direction="backward",
        ).set_index("index").sort_index()
        fresh = (merged["date"] - merged["available"]).dt.days <= SUE_MAX_AGE_DAYS
        features["sue"] = merged["sue"].where(fresh)
    if earnings_dates is not None and not earnings_dates.empty and "market_return_1d" in features.columns:
        events = _to_next_session(earnings_dates[["ticker", "date"]], features)[["ticker", "session"]].drop_duplicates()
        lookup = pd.Series(features.index, index=pd.MultiIndex.from_frame(features[["ticker", "date"]]))
        event_rows = lookup.reindex(pd.MultiIndex.from_frame(events.rename(columns={"session": "date"}))).dropna().astype(int).to_numpy()
        abnormal = features["return_1d"] - features["market_return_1d"]
        three_day = abnormal.groupby(features["ticker"]).transform(lambda s: s.rolling(3, min_periods=3).sum())
        # Value known at the close of the session after the release: row event + 1, same ticker.
        known_rows = event_rows + 1
        same_ticker = (known_rows < len(features)) & (features["ticker"].to_numpy()[np.minimum(known_rows, len(features) - 1)] == features["ticker"].to_numpy()[event_rows])
        known_rows = known_rows[same_ticker]
        ear = pd.Series(np.nan, index=features.index)
        ear.iloc[known_rows] = three_day.iloc[known_rows].to_numpy()
        features["ear"] = ear.groupby(features["ticker"]).ffill(limit=EAR_MAX_SESSIONS)
    return features


def add_insider_features(features: pd.DataFrame, insiders: pd.DataFrame) -> pd.DataFrame:
    """Insider open-market purchases over the last INSIDER_WINDOW_SESSIONS sessions, and net buying scaled by market cap.

    A Form 4 is usable from the session after its filing date (filings often arrive after the close).
    """
    trades = insiders.assign(date=pd.to_datetime(insiders["filed"]).astype("datetime64[ns]") + pd.Timedelta(days=1))
    aligned = _to_next_session(trades, features)
    if aligned.empty:
        return features
    buys = aligned["code"] == "P"
    daily = aligned.assign(
        purchase=np.where(buys, aligned["accession"], None),
        buy_value=aligned["value"].where(buys, 0.0),
        sell_value=aligned["value"].where(~buys, 0.0),
    ).groupby(["ticker", "session"]).agg(purchases=("purchase", "nunique"), buy_value=("buy_value", "sum"), sell_value=("sell_value", "sum"))
    features = features.merge(daily.reset_index().rename(columns={"session": "date"}), on=["ticker", "date"], how="left")
    features[["purchases", "buy_value", "sell_value"]] = features[["purchases", "buy_value", "sell_value"]].fillna(0.0)
    features = features.sort_values(["ticker", "date"]).reset_index(drop=True)
    window = _grouped_rolling(features, ["purchases", "buy_value", "sell_value"], INSIDER_WINDOW_SESSIONS, 1, "sum")
    features["insider_purchases_90d"] = window["purchases"]
    if "market_cap" in features.columns:
        features["insider_net_value_90d"] = (window["buy_value"] - window["sell_value"]) / features["market_cap"]
    return features.drop(columns=["purchases", "buy_value", "sell_value"])


def add_macro_features(features: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Market-wide conditions known before each date: VIX level, change and regime, rates and the yield curve.

    vix_regime is the VIX against its own trailing year (a z-score): high when fear is unusually high.
    """
    macro = macro.sort_values("date").ffill()
    vix_mean = macro["vix"].rolling(252, min_periods=126).mean()
    vix_std = macro["vix"].rolling(252, min_periods=126).std()
    macro = macro.assign(
        date=pd.to_datetime(macro["date"]).astype("datetime64[ns]"),
        vix_change_20d=macro["vix"] / macro["vix"].shift(20) - 1,
        vix_regime=(macro["vix"] - vix_mean) / vix_std,
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
