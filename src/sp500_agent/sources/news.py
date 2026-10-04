"""Recent company headlines, fetched when the agent asks for them.

Default: Yahoo Finance's public RSS feed (no key). With FINNHUB_API_KEY set: Finnhub's company-news
endpoint (free tier, about a year of history). Headlines get the same lexicon sentiment score as
the dataset's news so the numbers are comparable.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from datetime import date, timedelta

import pandas as pd

from .http import HttpClient

YAHOO_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"
FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"
NEWS_COLUMNS = ["ticker", "date", "title", "summary", "source", "url"]


def parse_yahoo_rss(xml_bytes: bytes, ticker: str) -> pd.DataFrame:
    rows = []
    for item in ET.fromstring(xml_bytes).iter("item"):
        title = (item.findtext("title") or "").strip()
        if title:
            rows.append(
                {
                    "ticker": ticker,
                    "date": pd.to_datetime(item.findtext("pubDate"), utc=True, errors="coerce"),
                    "title": title,
                    "summary": (item.findtext("description") or "").strip() or None,
                    "source": "Yahoo Finance",
                    "url": item.findtext("link"),
                }
            )
    return pd.DataFrame(rows, columns=NEWS_COLUMNS)


def parse_finnhub(items: list[dict], ticker: str) -> pd.DataFrame:
    rows = [
        {
            "ticker": ticker,
            "date": pd.to_datetime(item.get("datetime"), unit="s", utc=True, errors="coerce"),
            "title": item.get("headline"),
            "summary": item.get("summary") or None,
            "source": item.get("source"),
            "url": item.get("url"),
        }
        for item in items
        if item.get("headline")
    ]
    return pd.DataFrame(rows, columns=NEWS_COLUMNS)


class NewsFetcher:
    """Fetches and caches headlines per ticker for the lifetime of the object (one app session)."""

    def __init__(self, client: HttpClient | None = None, finnhub_key: str | None = None, days: int = 14):
        self.client = client or HttpClient(max_per_second=2)
        self.finnhub_key = finnhub_key if finnhub_key is not None else os.environ.get("FINNHUB_API_KEY")
        self.days = days
        self._cache: dict[str, pd.DataFrame] = {}

    @property
    def source(self) -> str:
        return "Finnhub" if self.finnhub_key else "Yahoo Finance RSS"

    def fetch(self, ticker: str) -> pd.DataFrame:
        if ticker not in self._cache:
            symbol = ticker.replace(".", "-")
            if self.finnhub_key:
                today = date.today()
                items = self.client.get_json(
                    FINNHUB_NEWS_URL,
                    {"symbol": symbol, "from": (today - timedelta(days=self.days)).isoformat(), "to": today.isoformat(), "token": self.finnhub_key},
                )
                news = parse_finnhub(items, ticker)
            else:
                response = self.client.get(YAHOO_RSS_URL, {"s": symbol, "region": "US", "lang": "en-US"})
                news = parse_yahoo_rss(response.content, ticker)
            self._cache[ticker] = news.sort_values("date", ascending=False).reset_index(drop=True)
        return self._cache[ticker]
