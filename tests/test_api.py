"""Web API tests on the synthetic live dataset, with a scripted fake LLM for the chat stream."""

import json

import pytest
from fastapi.testclient import TestClient

from sp500_agent.api.server import create_app
from sp500_agent.llm_agent import DeepSeekResearchAgent
from test_llm_agent import ScriptedChatClient, chat_response


@pytest.fixture(scope="module")
def client(live_research_data, tmp_path_factory):
    static = tmp_path_factory.mktemp("dist")
    (static / "assets").mkdir()
    (static / "index.html").write_text("<!doctype html><title>app</title>")
    (static / "assets" / "app.js").write_text("console.log('hi')")

    def agent_factory(toolkit, provider):
        script = [chat_response("tool_calls", "Checking.", [("stock_snapshot", '{"ticker": "T01"}')]), chat_response("stop", "T01 looks average.")] * 5
        return DeepSeekResearchAgent(toolkit, model="fake", client=ScriptedChatClient(script))

    app = create_app(loader=lambda: live_research_data, agent_factory=agent_factory, providers=lambda: ["deepseek"], static_dir=static)
    return TestClient(app)


def _events(response) -> list[dict]:
    return [json.loads(line[len("data: "):]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_overview(client):
    body = client.get("/api/overview").json()
    assert body["data_source"] == "live" and len(body["top"]) == 10
    assert body["model_metrics"]["auc_mean"] is not None
    assert set(body["backtest"]) == {"long_only", "long_short", "benchmark"}
    assert body["macro"]["series"]["vix"]["latest"] is not None
    assert "beat the median" in body["target"]


def test_stocks_list_and_search(client):
    body = client.get("/api/stocks").json()
    stocks = body["stocks"]
    assert len(stocks) == 13  # T12 left the index
    assert stocks[0]["rank"] == 1 and stocks[0]["stance"] == "constructive"
    assert {"ticker", "company", "model_probability", "market_cap", "pe_ratio"} <= set(stocks[0])
    assert client.get("/api/search", params={"q": "T0"}).json()["matches"]


def test_stock_detail_endpoints(client):
    detail = client.get("/api/stocks/t01").json()
    assert detail["ticker"] == "T01" and detail["snapshot"]["rank"] >= 1
    assert all(0 <= p["percentile"] <= 1 for p in detail["percentiles"]) and len(detail["percentiles"]) >= 5
    prices = client.get("/api/stocks/T01/prices", params={"days": 100}).json()["prices"]
    assert len(prices) == 100 and prices[-1]["date"] > prices[0]["date"]
    assert client.get("/api/stocks/T01/fundamentals").json()["history"]
    assert client.get("/api/stocks/T01/news").json()["headlines"]
    assert client.get("/api/stocks/T01/filings").json()["filings"]


def test_unknown_ticker_is_404_and_former_member_has_history(client):
    response = client.get("/api/stocks/ZZZZ")
    assert response.status_code == 404 and "not among" in response.json()["detail"]
    assert client.get("/api/stocks/T12/prices").status_code == 200  # left the index, history still available


def test_backtest_model_and_data(client):
    backtest = client.get("/api/backtest").json()
    assert backtest["growth"][0].keys() >= {"date", "long_only", "long_short", "benchmark"}
    assert min(row["benchmark"] for row in backtest["drawdown"]) <= 0
    model = client.get("/api/model").json()
    assert len(model["comparison"]) == 3 and model["calibration"] and model["features"]
    assert client.get("/api/model/report").text.startswith("# S&P 500 Research Report")
    data = client.get("/api/data").json()
    assert data["macro"] and data["index_changes"]


def test_chat_streams_tool_progress_then_answer(client):
    response = client.post("/api/agent/chat", json={"session_id": "s1", "message": "How is T01?"})
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _events(response)
    assert [e["type"] for e in events] == ["tool_start", "tool_end", "answer"]
    assert events[1]["result"]["ticker"] == "T01"
    assert events[-1]["text"] == "T01 looks average." and events[-1]["provider"] == "deepseek"
    # Same session continues the same conversation; reset starts a new one.
    client.post("/api/agent/chat", json={"session_id": "s1", "message": "And now?"})
    agent, _ = client.app.state.service.session_agent("s1", "deepseek")
    assert sum(1 for m in agent.history if m["role"] == "user") == 2
    client.post("/api/agent/reset", json={"session_id": "s1"})
    agent, _ = client.app.state.service.session_agent("s1", "deepseek")
    assert agent.history == []


def test_chat_falls_back_to_rule_based_without_a_key(live_research_data):
    app = create_app(loader=lambda: live_research_data, providers=lambda: [], static_dir=None)
    with TestClient(app) as rule_client:
        assert rule_client.get("/api/agent/status").json()["mode"] == "rule-based"
        events = _events(rule_client.post("/api/agent/chat", json={"session_id": "x", "message": "top 3 stocks"}))
        assert events[-1]["provider"] == "rule-based" and "| rank |" in events[-1]["text"]


def test_chat_validates_input(client):
    assert client.post("/api/agent/chat", json={"session_id": "", "message": "hi"}).status_code == 422


def test_frontend_is_served_with_spa_fallback(client):
    assert "<title>app</title>" in client.get("/").text
    assert "<title>app</title>" in client.get("/stock/AAPL").text  # client-side route
    assert client.get("/assets/app.js").text == "console.log('hi')"
    assert client.get("/api/nope").status_code == 404


def test_chat_limiter_counts_per_client_and_per_day():
    from sp500_agent.api.server import ChatLimiter

    now = [0.0]
    limiter = ChatLimiter(per_hour=2, per_day=3, clock=lambda: now[0])
    assert limiter.check("a") is None and limiter.check("a") is None
    assert "2 questions per hour" in limiter.check("a")
    assert limiter.check("b") is None  # another visitor
    assert "daily" in limiter.check("c")  # 3 accepted today
    now[0] = 3601
    assert "daily" in limiter.check("a")  # hourly window reset, daily cap still reached
    now[0] = 86401
    assert limiter.check("a") is None


def test_demo_mode_limits_chat_and_locks_reload(live_research_data):
    from sp500_agent.api.server import ChatLimiter

    app = create_app(loader=lambda: live_research_data, providers=lambda: [], static_dir=None, demo=True, limiter=ChatLimiter(per_hour=1))
    with TestClient(app) as demo_client:
        status = demo_client.get("/api/agent/status").json()
        assert status["demo"] is True and status["limits"]["per_hour"] == 1
        ok = demo_client.post("/api/agent/chat", json={"session_id": "s", "message": "top 3"}, headers={"X-Forwarded-For": "1.2.3.4"})
        assert ok.status_code == 200
        limited = demo_client.post("/api/agent/chat", json={"session_id": "s", "message": "top 3"}, headers={"X-Forwarded-For": "1.2.3.4, 10.0.0.1"})
        assert limited.status_code == 429 and "per hour" in limited.json()["detail"]
        assert demo_client.post("/api/agent/chat", json={"session_id": "t", "message": "top 3"}, headers={"X-Forwarded-For": "5.6.7.8"}).status_code == 200
        assert demo_client.post("/api/reload").status_code == 403


def test_admin_token_guards_reload(live_research_data):
    app = create_app(loader=lambda: live_research_data, providers=lambda: [], static_dir=None, admin_token="secret")
    with TestClient(app) as admin_client:
        assert admin_client.post("/api/reload").status_code == 403
        assert admin_client.post("/api/reload", headers={"X-Admin-Token": "secret"}).status_code == 200
