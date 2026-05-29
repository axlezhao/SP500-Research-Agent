from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RAW_DIR


def make_sample_data(seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tickers = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "UNH", "AMZN", "GOOGL"]
    sectors = {"AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "JPM": "Financials", "XOM": "Energy", "UNH": "Health Care", "AMZN": "Consumer Discretionary", "GOOGL": "Communication Services"}
    dates = pd.bdate_range("2024-01-02", periods=260)
    price_rows = []
    for ticker in tickers:
        returns = rng.normal(rng.normal(0.0005, 0.0003), rng.uniform(0.012, 0.03), len(dates))
        close = rng.uniform(80, 350) * np.cumprod(1 + returns)
        for date, price in zip(dates, close):
            price_rows.append({"ticker": ticker, "date": date.date().isoformat(), "close": round(float(price), 2), "volume": int(rng.integers(2_000_000, 120_000_000))})
    fundamentals = [{"ticker": ticker, "company_name": f"{ticker} Corporation", "sector": sectors[ticker], "market_cap": int(rng.uniform(80, 3000) * 1_000_000_000), "pe_ratio": round(float(rng.uniform(12, 55)), 2), "revenue": int(rng.uniform(20, 500) * 1_000_000_000), "profit_margin": round(float(rng.uniform(0.05, 0.42)), 3), "debt_to_equity": round(float(rng.uniform(0.1, 2.8)), 2), "roe": round(float(rng.uniform(0.04, 0.55)), 3)} for ticker in tickers]
    phrases = {"positive": ["beats expectations", "raises guidance", "margin expands", "strong demand"], "negative": ["misses estimates", "regulatory pressure", "weak demand", "margin contracts"], "neutral": ["announces product update", "hosts investor day", "files quarterly update"]}
    news_rows = []
    for ticker in tickers:
        for date in dates[::10]:
            tone = rng.choice(["positive", "negative", "neutral"], p=[0.42, 0.28, 0.30])
            news_rows.append({"ticker": ticker, "date": date.date().isoformat(), "title": f"{ticker} {rng.choice(phrases[tone])}", "summary": f"{ticker} news item with {tone} implications.", "sentiment": tone})
    pd.DataFrame(price_rows).to_csv(RAW_DIR / "sample_prices.csv", index=False)
    pd.DataFrame(fundamentals).to_csv(RAW_DIR / "sample_fundamentals.csv", index=False)
    pd.DataFrame(news_rows).to_csv(RAW_DIR / "sample_news.csv", index=False)

