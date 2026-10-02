import pandas as pd
import pytest

from sp500_agent.chat import answer_message, find_ticker

TICKERS = ["AAPL", "NVDA", "MSFT", "A", "IT", "ON", "ALL", "NOW", "KEY", "ARE", "BRK.B"]


@pytest.fixture
def scored():
    n = len(TICKERS)
    return pd.DataFrame(
        {
            "ticker": TICKERS,
            "company_name": [f"{t} Corp" for t in TICKERS],
            "model_probability": [0.9 - i * 0.05 for i in range(n)],
            "rank": range(1, n + 1),
            "date": pd.Timestamp("2024-12-30"),
            "close": 100.0,
            "return_5d": 0.01,
            "return_20d": 0.02,
            "volatility_20d": 0.3,
        }
    )


@pytest.mark.parametrize(
    "message, expected",
    [
        ("Give me a view of AAPL", "AAPL"),
        ("Is it worth looking at NVDA?", "NVDA"),
        ("How is MSFT doing now", "MSFT"),
        ("A quick look at NVDA", "NVDA"),
        ("analyze nvda", "NVDA"),
        ("ANALYZE NVDA", "NVDA"),
        ("Tell me about NOW", "NOW"),
        ("what about $on", "ON"),
        ("A", "A"),
        ("IT", "IT"),
        ("Thoughts on brk-b?", "BRK.B"),
        ("How is Apple doing?", "AAPL"),
        ("is it all good?", None),
    ],
)
def test_find_ticker(message, expected, scored):
    assert find_ticker(message, scored) == expected


def test_ticker_beats_ranking_words(scored):
    assert answer_message("What's the rank of AAPL?", scored).ticker == "AAPL"


def test_ranking_words_match_whole_words_only(scored):
    answer = answer_message("I can't stop thinking about the laptop market", scored)
    assert answer.table is None and answer.ticker is None


@pytest.mark.parametrize("message, first, rows", [("top 3 stocks", "AAPL", 3), ("Weakest 4 stocks", "BRK.B", 4), ("best stocks", "AAPL", 10)])
def test_rankings(message, first, rows, scored):
    answer = answer_message(message, scored)
    assert answer.table.iloc[0]["ticker"] == first
    assert len(answer.table) == rows
