"""Rule-based question handling for the Streamlit chatbot, kept free of Streamlit so it can be tested."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .agent import brief_markdown
from .formatting import COMPANY_NAME_COLUMNS, company_label, fmt_pct, fmt_prob

POPULAR_ALIASES = {
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "NVIDIA": "NVDA",
    "TESLA": "TSLA",
    "AMAZON": "AMZN",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "META": "META",
    "FACEBOOK": "META",
    "JPMORGAN": "JPM",
    "JP MORGAN": "JPM",
    "EXXON": "XOM",
}

# Everyday words, several of which are also S&P 500 tickers (A, IT, ON, ALL, NOW, KEY, ARE, ...).
# They only count as tickers when typed in capitals inside an otherwise mixed-case message, or with a $.
COMMON_WORDS = {
    "A", "I", "AN", "AND", "ANY", "ARE", "AS", "AT", "BE", "BEST", "BIG", "BUY", "BY", "CAN", "CASH",
    "COST", "DO", "FAST", "FOR", "GO", "GOOD", "HAS", "HE", "HOW", "IF", "IN", "IS", "IT", "KEY", "LOW",
    "ME", "MY", "NEXT", "NO", "NOT", "NOW", "OF", "OK", "ON", "ONE", "OR", "OUT", "PEAK", "RANK", "REAL",
    "SEE", "SELL", "SHE", "SO", "THE", "TO", "TOP", "UP", "US", "VIEW", "WAY", "WE", "WELL", "WHAT",
    "WHY", "YOU", "ALL", "ALLY", "HOLD", "LOVE", "TECH",
}

TOKEN_RE = re.compile(r"\$?\b[A-Za-z]{1,5}(?:[.\-][A-Za-z]{1,2})?\b")
RANK_UP_RE = re.compile(r"\b(top|best|strongest|highest|rank(?:ing|ings|ed)?)\b", re.IGNORECASE)
RANK_DOWN_RE = re.compile(r"\b(worst|weakest|lowest|bottom)\b", re.IGNORECASE)
COUNT_RE = re.compile(r"\b(\d{1,3})\b")
MAX_RANKING_ROWS = 25

FALLBACK = (
    "Ask me about a ticker, for example `NVDA`, `AAPL`, or `MSFT`, "
    "or ask for `top 10 stocks` / `weakest 10 stocks`."
)


@dataclass
class ChatAnswer:
    text: str
    table: pd.DataFrame | None = None
    ticker: str | None = None


def _normalise(symbol: str) -> str:
    return symbol.upper().replace("-", ".")


def find_ticker(message: str, scored: pd.DataFrame) -> str | None:
    tickers = scored["ticker"].astype(str).str.upper()
    lookup = {_normalise(ticker): ticker for ticker in tickers}
    # A message that is only a ticker (e.g. from the sidebar's Analyze button) is always that ticker.
    whole = lookup.get(_normalise(message.strip().lstrip("$")))
    if whole:
        return whole

    mixed_case = message != message.upper() and message != message.lower()

    best, best_score = None, 0
    for match in TOKEN_RE.finditer(message):
        raw = match.group(0)
        word = raw.lstrip("$")
        ticker = lookup.get(_normalise(word))
        if ticker is None:
            continue
        typed_upper = mixed_case and word == word.upper()
        common = word.upper() in COMMON_WORDS
        if raw.startswith("$"):
            score = 4
        elif typed_upper and not common:
            score = 3
        elif typed_upper:
            score = 2  # e.g. "Analyze A": still a ticker, but loses to a less ambiguous one
        elif not common and len(word) >= 2:
            score = 1
        else:
            continue
        if score > best_score:
            best, best_score = ticker, score
    if best:
        return best

    upper = message.upper()
    valid = set(tickers)
    for name, ticker in POPULAR_ALIASES.items():
        if re.search(rf"\b{re.escape(name)}\b", upper) and ticker in valid:
            return ticker

    name_columns = [col for col in COMPANY_NAME_COLUMNS if col in scored.columns]
    for col in name_columns:
        for ticker, company_name in zip(tickers, scored[col]):
            if pd.isna(company_name):
                continue
            company_name = str(company_name).upper().strip()
            if len(company_name) >= 3 and re.search(rf"\b{re.escape(company_name)}\b", upper):
                return ticker
    return None


def ranking_answer(scored: pd.DataFrame, n: int = 10, ascending: bool = False) -> ChatAnswer:
    ranked = scored.sort_values("up_probability_5d", ascending=ascending).head(n).copy()
    ranked["company"] = ranked.apply(company_label, axis=1)
    ranked["model_probability"] = ranked["up_probability_5d"].map(fmt_prob)
    ranked["5d_return"] = ranked["return_5d"].map(fmt_pct)
    ranked["20d_return"] = ranked["return_20d"].map(fmt_pct)
    ranked["volatility_ann"] = ranked["volatility_20d"].map(fmt_pct)
    if "sector" not in ranked.columns:
        ranked["sector"] = "n/a"
    table = ranked[["rank", "ticker", "company", "sector", "model_probability", "5d_return", "20d_return", "volatility_ann"]]
    direction = "lowest" if ascending else "highest"
    text = (
        f"Here are the {len(table)} stocks with the {direction} model-ranked 5-day upward-move probability. "
        "This is a research signal, not a buy or sell recommendation."
    )
    return ChatAnswer(text, table=table)


def ticker_answer(ticker: str, scored: pd.DataFrame) -> ChatAnswer:
    row = scored[scored["ticker"] == ticker].iloc[0]
    return ChatAnswer(brief_markdown(row, len(scored), heading="###"), ticker=ticker)


def answer_message(message: str, scored: pd.DataFrame) -> ChatAnswer:
    # A named ticker wins over ranking words: "what's the rank of AAPL?" is about AAPL,
    # and its brief includes the rank.
    ticker = find_ticker(message, scored)
    if ticker:
        return ticker_answer(ticker, scored)

    down = RANK_DOWN_RE.search(message)
    if down or RANK_UP_RE.search(message):
        count = COUNT_RE.search(message)
        n = min(max(int(count.group(1)), 1), MAX_RANKING_ROWS) if count else 10
        return ranking_answer(scored, n=n, ascending=bool(down))

    return ChatAnswer(FALLBACK)
