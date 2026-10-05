"""Daily prices from Yahoo Finance via the yfinance package (no key needed).

yfinance is unofficial: it can break when Yahoo changes its site, and Yahoo's terms don't allow
redistributing the data, so keep downloads local. Delisted tickers are usually unavailable, which
limits how far the survivorship-bias fix can go; the pipeline reports the coverage.
"""

from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume", "splits"]


def yahoo_symbol(ticker: str) -> str:
    return ticker.replace(".", "-")


def _import_yfinance():
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Live prices need yfinance: pip install yfinance") from exc
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # silence per-ticker "possibly delisted" noise
    return yf


def tidy_download(raw: pd.DataFrame, symbols: dict[str, str]) -> pd.DataFrame:
    """Turn yfinance's wide (field x ticker) frame into long rows: ticker, date, prices, volume."""
    frames = []
    for symbol, ticker in symbols.items():
        if isinstance(raw.columns, pd.MultiIndex):
            if symbol not in raw.columns.get_level_values(0):
                continue
            block = raw[symbol]
        else:
            block = raw
        block = block.rename(columns=lambda c: str(c).strip().lower().replace(" ", "_"))
        block = block.rename(columns={"stock_splits": "splits"})
        if "close" not in block or block["close"].notna().sum() == 0:
            continue
        block = block.reindex(columns=PRICE_COLUMNS)
        block = block.dropna(subset=["close"]).assign(ticker=ticker)
        block.index = pd.to_datetime(block.index).tz_localize(None).normalize()
        frames.append(block.rename_axis("date").reset_index())
    if not frames:
        return pd.DataFrame(columns=["ticker", "date", *PRICE_COLUMNS])
    return pd.concat(frames, ignore_index=True)


def add_raw_close(prices: pd.DataFrame) -> pd.DataFrame:
    """Undo split adjustment so price x reported share count gives the market cap of the day.

    Yahoo's `close` is split-adjusted. A share count filed before a 4:1 split must be paired with the
    pre-split price, so multiply by the product of all split ratios after each date.
    """
    prices = prices.sort_values(["ticker", "date"]).copy()
    ratio = prices["splits"].fillna(0).replace(0, 1.0).astype(float)
    # Product of ratios strictly after each row = reverse cumulative product, shifted by one row.
    after = ratio.groupby(prices["ticker"]).transform(lambda r: r[::-1].cumprod()[::-1].shift(-1).fillna(1.0))
    prices["raw_close"] = prices["close"] * after
    # Shares reported at the time x split_factor = shares in today's split-adjusted units (for net issuance).
    prices["split_factor"] = after
    return prices


def fetch_prices(tickers: list[str], start: str, end: str | None = None, batch_size: int = 100, downloader=None) -> pd.DataFrame:
    """Long table: ticker, date, open, high, low, close (split-adjusted), adj_close (dividend-adjusted), volume, splits, raw_close, split_factor."""
    download = downloader or _import_yfinance().download
    frames = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        symbols = {yahoo_symbol(t): t for t in batch}
        raw = download(
            list(symbols), start=start, end=end, auto_adjust=False, actions=True,
            group_by="ticker", threads=True, progress=False,
        )
        frames.append(tidy_download(raw, symbols))
    prices = pd.concat(frames, ignore_index=True) if frames else tidy_download(pd.DataFrame(), {})
    prices = prices[(prices["close"] > 0) & np.isfinite(prices["close"])]
    return add_raw_close(drop_unfinished_session(prices)).reset_index(drop=True)


def drop_unfinished_session(prices: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Yahoo returns today's bar while the market is still open; its 'close' is just the latest trade."""
    now = now or datetime.now(ZoneInfo("America/New_York"))
    if (now.hour, now.minute) >= (16, 30):
        return prices
    return prices[prices["date"] < pd.Timestamp(now.date())]
