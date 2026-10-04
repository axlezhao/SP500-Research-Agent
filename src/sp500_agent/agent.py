from __future__ import annotations

import argparse

import pandas as pd

from .config import REPORT_DIR, TARGET_DESCRIPTION
from .features import load_features
from .formatting import company_label, fmt_money, fmt_num, fmt_pct, fmt_price, fmt_prob, stance
from .model import load_model, score_latest


def brief_markdown(row: pd.Series, total: int, heading: str = "#") -> str:
    """Research brief for one scored row (from score_latest)."""
    ticker = row["ticker"]
    name = company_label(row)
    title = f"{ticker}: {name}" if name != ticker else ticker
    news_count = row.get("news_count_20d")
    if pd.notna(row.get("fundamentals_as_of")):
        fundamentals_note = f"Trailing-twelve-month figures from SEC filings public on {pd.Timestamp(row['fundamentals_as_of']).date()}. Their ranks against other index members are model inputs."
    else:
        fundamentals_note = "Latest values from the dataset. They are shown for context and are not model inputs, because no history is available for them."
    leverage_col = "liabilities_to_equity" if pd.notna(row.get("liabilities_to_equity")) else "debt_to_equity"
    leverage_label = "Liabilities to equity" if leverage_col == "liabilities_to_equity" else "Debt to equity"
    return f"""{heading} {title} Research Brief

{heading}# Agent View

The model view is **{stance(int(row['rank']), total)}**. {ticker} ranks **{int(row['rank'])} of {total}** by the model's probability that it will {TARGET_DESCRIPTION}, as of **{pd.Timestamp(row['date']).date()}**.

This is educational model output, not financial advice.

{heading}# Key Signals

- Model probability: **{fmt_prob(row['model_probability'])}**
- Latest close: **{fmt_price(row.get('close'))}**
- 5-day return: **{fmt_pct(row.get('return_5d'))}**
- 20-day return: **{fmt_pct(row.get('return_20d'))}**
- 60-day momentum: **{fmt_pct(row.get('momentum_60d'))}**
- 20-day volatility (annualised): **{fmt_pct(row.get('volatility_20d'))}**
- 20-day news sentiment: **{fmt_num(row.get('sentiment_20d'))}** (−1 to +1, average over articles)
- 20-day news count: **{fmt_num(news_count, 0)}**

{heading}# Fundamentals

{fundamentals_note}

- Sector: **{row.get('sector') if pd.notna(row.get('sector')) else 'n/a'}**
- Market cap: **{fmt_money(row.get('market_cap'))}**
- P/E ratio: **{fmt_num(row.get('pe_ratio'), 1)}**
- Revenue: **{fmt_money(row.get('revenue'))}**
- Profit margin: **{fmt_pct(row.get('profit_margin'))}**
- {leverage_label}: **{fmt_num(row.get(leverage_col))}**
- ROE: **{fmt_pct(row.get('roe'))}**
"""


def make_research_brief(ticker: str) -> str:
    ticker = ticker.upper()
    scored = score_latest(load_features(), load_model())
    match = scored[scored["ticker"] == ticker]
    if match.empty:
        raise ValueError(f"{ticker} not found. Try one of: {', '.join(scored['ticker'].head(20).tolist())}")
    brief = brief_markdown(match.iloc[0], len(scored))
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
