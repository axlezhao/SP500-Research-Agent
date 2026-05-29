from __future__ import annotations

import argparse

import pandas as pd

from .config import PROCESSED_DIR, REPORT_DIR
from .model import load_model, score_latest


def _fmt_pct(value) -> str:
    return "n/a" if pd.isna(value) else f"{float(value) * 100:.1f}%"


def _fmt_num(value) -> str:
    if pd.isna(value):
        return "n/a"
    value = float(value)
    return f"${value / 1_000_000_000:.1f}B" if abs(value) >= 1_000_000_000 else f"{value:.2f}"


def make_research_brief(ticker: str) -> str:
    ticker = ticker.upper()
    features = pd.read_csv(PROCESSED_DIR / "research_features.csv", parse_dates=["date"])
    scored = score_latest(features, load_model())
    match = scored[scored["ticker"] == ticker]
    if match.empty:
        raise ValueError(f"{ticker} not found. Try one of: {', '.join(scored['ticker'].head(20).tolist())}")
    row = match.iloc[0]
    rank = int(scored.index.get_loc(row.name)) + 1
    stance = "constructive" if row["up_probability_5d"] >= 0.60 else "cautious" if row["up_probability_5d"] <= 0.40 else "neutral"
    brief = f"""# {ticker} Research Brief

## Agent View

The model view is **{stance}**. {ticker} ranks **{rank} of {len(scored)}** by predicted 5-day upward-move probability.

This is educational model output, not financial advice.

## Key Signals

- Predicted 5-day upward probability: **{row['up_probability_5d']:.1%}**
- Latest close: **{_fmt_num(row.get('close'))}**
- 5-day return: **{_fmt_pct(row.get('return_5d'))}**
- 20-day return: **{_fmt_pct(row.get('return_20d'))}**
- 20-day volatility: **{_fmt_pct(row.get('volatility_20d'))}**
- 20-day news sentiment: **{row.get('sentiment_20d', 0):.2f}**
- 20-day news count: **{row.get('news_count_20d', 0):.0f}**

## Fundamentals Snapshot

- Sector: **{row.get('sector', 'n/a')}**
- Market cap: **{_fmt_num(row.get('market_cap'))}**
- P/E ratio: **{_fmt_num(row.get('pe_ratio'))}**
- Revenue: **{_fmt_num(row.get('revenue'))}**
- Profit margin: **{_fmt_pct(row.get('profit_margin'))}**
- Debt to equity: **{_fmt_num(row.get('debt_to_equity'))}**
- ROE: **{_fmt_pct(row.get('roe'))}**
"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{ticker}_research_brief.md").write_text(brief, encoding="utf-8")
    return brief


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()
    print(make_research_brief(args.ticker))


if __name__ == "__main__":
    main()

