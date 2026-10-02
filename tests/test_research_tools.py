import json

import pytest

from sp500_agent.research_tools import TOOL_DEFINITIONS, ResearchToolkit, ToolInputError


@pytest.fixture(scope="module")
def toolkit(research_data):
    return ResearchToolkit(research_data)


def _json(result):
    # Results go to the model as JSON; NaN would make the payload invalid.
    return json.loads(json.dumps(result, allow_nan=False))


@pytest.mark.parametrize("tool", [t["name"] for t in TOOL_DEFINITIONS])
def test_every_tool_returns_strict_json(toolkit, tool):
    arguments = {
        "search_companies": {"query": "corp"},
        "stock_snapshot": {"ticker": "NVDA"},
        "compare_stocks": {"tickers": ["AAPL", "MSFT"]},
        "price_history": {"ticker": "AAPL", "days": 120},
        "recent_news": {"ticker": "JPM"},
    }.get(tool, {})
    assert _json(toolkit.call(tool, arguments))


def test_snapshot_has_signals_and_fundamentals(toolkit):
    snap = toolkit.call("stock_snapshot", {"ticker": "nvda"})
    assert snap["ticker"] == "NVDA" and 1 <= snap["rank"] <= snap["out_of"] == 8
    assert {"return_20d", "volatility_20d", "momentum_60d"} <= set(snap["signals"])
    assert "market_cap" in snap["fundamentals_snapshot"]


def test_unknown_ticker_suggests_alternatives(toolkit):
    with pytest.raises(ToolInputError, match="Did you mean: AAPL"):
        toolkit.call("stock_snapshot", {"ticker": "AAP"})


def test_rank_by_sector_and_direction(toolkit):
    tech = toolkit.call("rank_stocks", {"n": 10, "sector": "technology"})["stocks"]
    assert {s["sector"] for s in tech} == {"Technology"} and len(tech) == 3
    bottom = toolkit.call("rank_stocks", {"n": 2, "direction": "bottom"})["stocks"]
    assert [s["rank"] for s in bottom] == [8, 7]


def test_unknown_sector_lists_valid_ones(toolkit):
    with pytest.raises(ToolInputError, match="Valid sectors: .*Energy"):
        toolkit.call("rank_stocks", {"sector": "Crypto"})


def test_screen_applies_filters(toolkit, research_data):
    limit = float(research_data.scored["volatility_20d"].median())
    result = toolkit.call("screen_stocks", {"max_volatility": limit, "sort_by": "volatility_20d", "ascending": True})
    vols = [s["volatility_20d"] for s in result["stocks"]]
    assert vols and max(vols) <= limit and vols == sorted(vols)


def test_price_history_is_compact(toolkit):
    history = toolkit.call("price_history", {"ticker": "AAPL", "days": 252})
    assert len(history["series"]) <= 61
    assert history["series"][-1]["date"] == history["end_date"]
    assert history["max_drawdown"] <= 0


def test_recent_news_is_newest_first(toolkit):
    dates = [h["date"] for h in toolkit.call("recent_news", {"ticker": "JPM", "limit": 5})["headlines"]]
    assert len(dates) == 5 and dates == sorted(dates, reverse=True)


def test_call_validates_arguments(toolkit):
    with pytest.raises(ToolInputError, match="Unexpected argument"):
        toolkit.call("rank_stocks", {"count": 5})
    with pytest.raises(ToolInputError, match="Missing required"):
        toolkit.call("stock_snapshot", {})
    with pytest.raises(ToolInputError, match="Unknown tool"):
        toolkit.call("buy_stock", {"ticker": "AAPL"})
    with pytest.raises(ToolInputError, match="2 to 8"):
        toolkit.call("compare_stocks", {"tickers": ["AAPL"]})


def test_model_and_backtest_tools_report_the_evidence(toolkit):
    perf = toolkit.call("model_performance", {})
    assert len(perf["model_comparison"]) == 3 and perf["calibration"]
    backtest = toolkit.call("backtest_results", {})
    assert {row["strategy"] for row in backtest["summary"]} == {"long_only", "long_short", "benchmark"}
    assert any("Survivorship" in c for c in backtest["caveats"])
