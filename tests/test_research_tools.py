import json

import pytest

from sp500_agent.research_tools import TOOL_DEFINITIONS, ResearchToolkit, ToolInputError


@pytest.fixture(scope="module")
def toolkit(research_data):
    return ResearchToolkit(research_data)


def _json(result):
    # Results go to the model as JSON; NaN would make the payload invalid.
    return json.loads(json.dumps(result, allow_nan=False))


SAMPLE_ARGS = {
    "search_companies": {"query": "corp"},
    "stock_snapshot": {"ticker": "NVDA"},
    "compare_stocks": {"tickers": ["AAPL", "MSFT"]},
    "price_history": {"ticker": "AAPL", "days": 120},
    "recent_news": {"ticker": "JPM"},
    "fundamentals_history": {"ticker": "AAPL"},
    "recent_filings": {"ticker": "AAPL"},
}
LIVE_ARGS = {key: {k: ("T01" if k == "ticker" else ["T01", "T02"] if k == "tickers" else v) for k, v in args.items()} for key, args in SAMPLE_ARGS.items()}
LIVE_ARGS["search_companies"] = {"query": "T0"}


@pytest.fixture(scope="module")
def live_toolkit(live_research_data):
    return ResearchToolkit(live_research_data)


@pytest.mark.parametrize("tool", [t["name"] for t in TOOL_DEFINITIONS])
def test_every_tool_returns_strict_json(toolkit, live_toolkit, tool):
    assert _json(toolkit.call(tool, SAMPLE_ARGS.get(tool, {})))
    assert _json(live_toolkit.call(tool, LIVE_ARGS.get(tool, {})))


def test_live_only_tools_explain_when_data_is_missing(toolkit):
    for tool, args in [("fundamentals_history", {"ticker": "AAPL"}), ("macro_snapshot", {}), ("index_changes", {}), ("recent_filings", {"ticker": "AAPL"})]:
        assert "live data" in _json(toolkit.call(tool, args)).get("note", "")


def test_fundamentals_history_is_newest_first_with_filing_dates(live_toolkit):
    history = live_toolkit.call("fundamentals_history", {"ticker": "T03", "quarters": 3})["history"]
    assert len(history) == 3 and history[0]["available_date"] > history[-1]["available_date"]
    assert history[0]["profit_margin"] == pytest.approx(history[0]["net_income_ttm"] / history[0]["revenue_ttm"], rel=1e-4)  # tool output is rounded


def test_macro_snapshot(live_toolkit):
    snapshot = live_toolkit.call("macro_snapshot", {})
    vix = snapshot["series"]["vix"]
    assert 0 <= vix["percentile_10y"] <= 1 and vix["one_year_ago"] is not None
    assert snapshot["series"]["term_spread"]["latest"] == pytest.approx(3.0 - 4.5)


def test_index_changes(live_toolkit):
    assert len(live_toolkit.call("index_changes", {})["recent_changes"]) == 2
    t12 = live_toolkit.call("index_changes", {"ticker": "t12"})
    assert t12["currently_in_index"] is False and t12["changes"][0]["removed"] == "T12"
    assert live_toolkit.call("index_changes", {"ticker": "T13"})["membership_periods"][0]["to"] is None


def test_live_news_and_filings(live_toolkit):
    news = live_toolkit.call("recent_news", {"ticker": "T01", "limit": 5})
    assert news["fetched_from"] == "Fake RSS"
    assert [h["sentiment_score"] for h in news["headlines"]] == [1.0, -1.0]
    filings = live_toolkit.call("recent_filings", {"ticker": "T01", "form": "8-K"})["filings"]
    assert [f["form"] for f in filings] == ["8-K"]
    with pytest.raises(ToolInputError, match="form must be"):
        live_toolkit.call("recent_filings", {"ticker": "T01", "form": "S-1"})


def test_snapshot_reports_point_in_time_fundamentals_and_data_warnings(live_toolkit, live_research_data):
    snap = live_toolkit.call("stock_snapshot", {"ticker": "T01"})
    assert snap["fundamentals_as_of"] is not None and "SEC filings" in snap["notes"]
    assert snap["data_warning"] is None
    # A 60% one-day drop in the last 20 sessions is flagged as a possible corporate action.
    features = live_research_data.features
    last = features.index[(features["ticker"] == "T02") & (features["date"] == features["date"].max())]
    original = features.loc[last, "return_1d"].copy()
    features.loc[last, "return_1d"] = -0.6
    try:
        warning = live_toolkit.call("stock_snapshot", {"ticker": "T02"})["data_warning"]
    finally:
        features.loc[last, "return_1d"] = original
    assert "-60%" in warning and "corporate action" in warning


def test_snapshot_has_signals_and_fundamentals(toolkit):
    snap = toolkit.call("stock_snapshot", {"ticker": "nvda"})
    assert snap["ticker"] == "NVDA" and 1 <= snap["rank"] <= snap["out_of"] == 8
    assert {"return_20d", "volatility_20d", "momentum_60d"} <= set(snap["signals"])
    assert "market_cap" in snap["fundamentals"] and "not model inputs" in snap["notes"]


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
    assert len(perf["model_comparison"]) == 4 and perf["calibration"]
    assert perf["overfitting_checks"]["trials_counted"] >= 4
    backtest = toolkit.call("backtest_results", {})
    assert {row["strategy"] for row in backtest["summary"]} == {"long_only", "long_short", "benchmark"}
    assert any("Survivorship" in c for c in backtest["caveats"])
