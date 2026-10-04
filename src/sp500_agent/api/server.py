"""FastAPI backend for the web dashboard.

Read endpoints serve the research outputs (rankings, stock detail, backtest, model diagnostics, data
quality). POST /api/agent/chat streams the agent's progress as server-sent events: one event per tool
call as it starts and finishes, then the answer. Without an LLM key the chat falls back to the
rule-based assistant. In production the built React app (web/dist) is served from the same origin.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from collections import OrderedDict, defaultdict, deque
from pathlib import Path
from typing import Callable

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import PRODUCTION_SPEC, PROJECT_ROOT, TRADING_DAYS_PER_YEAR
from ..formatting import company_label, stance
from ..research_tools import ResearchData, ResearchToolkit, ToolInputError, to_json_safe

WEB_DIST = PROJECT_ROOT / "web" / "dist"
MAX_SESSIONS = 50
SCREENER_FIELDS = [
    "rank", "ticker", "company", "sector", "model_probability", "stance", "close", "return_5d", "return_20d",
    "momentum_60d", "volatility_20d", "market_cap", "pe_ratio", "profit_margin", "revenue_growth_yoy", "sentiment_20d",
]
# Signals shown as "where does this stock sit among index members today" (cross-sectional percentile).
PERCENTILE_SIGNALS = {
    "return_20d_xs_rank": "20-day return",
    "momentum_60d_xs_rank": "60-day momentum",
    "volatility_20d_xs_rank": "Volatility",
    "earnings_yield_xs_rank": "Earnings yield",
    "book_to_market_xs_rank": "Book-to-market",
    "sales_yield_xs_rank": "Sales yield",
    "profit_margin_xs_rank": "Profit margin",
    "roe_xs_rank": "Return on equity",
    "revenue_growth_yoy_xs_rank": "Revenue growth",
    "market_cap_xs_rank": "Size",
}
STRATEGY_LABELS = {"long_only": "Long top 20%", "long_short": "Long-short", "benchmark": "Equal-weight benchmark", "sp500": "S&P 500 (SPY)"}


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=4000)
    provider: str | None = None


class ResetRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)


class ChatLimiter:
    """Caps chat messages per client per hour and in total per day (a cost guard for public demos).

    Limits of 0 or None mean unlimited. Counts live in memory, so they reset when the server restarts.
    """

    def __init__(self, per_hour: int | None = None, per_day: int | None = None, clock: Callable[[], float] = time.monotonic):
        self.per_hour = per_hour or None
        self.per_day = per_day or None
        self._clock = clock
        self._lock = threading.Lock()
        self._clients: dict[str, deque] = defaultdict(deque)
        self._all: deque = deque()

    def check(self, client: str) -> str | None:
        """Record one message for `client`, or return why it is refused."""
        now = self._clock()
        with self._lock:
            mine = self._clients[client]
            while mine and now - mine[0] > 3600:
                mine.popleft()
            while self._all and now - self._all[0] > 86400:
                self._all.popleft()
            if self.per_day and len(self._all) >= self.per_day:
                return "This demo has reached its daily question limit. Please try again tomorrow."
            if self.per_hour and len(mine) >= self.per_hour:
                return f"You've reached the demo limit of {self.per_hour} questions per hour. Please try again later."
            mine.append(now)
            self._all.append(now)
            return None


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _spec(bundle: dict) -> dict:
    """What the served model predicts (stored with it), falling back to the configured production setup."""
    return bundle.get("spec") or PRODUCTION_SPEC.as_dict()


class ResearchService:
    """Loads the research data once (thread-safe, reloadable) and keeps one agent per chat session."""

    def __init__(self, loader: Callable[[], ResearchData], agent_factory: Callable | None = None, providers: Callable[[], list[str]] | None = None):
        self._loader = loader
        self._agent_factory = agent_factory
        self._providers = providers
        self._lock = threading.Lock()
        self._data: ResearchData | None = None
        self._toolkit: ResearchToolkit | None = None
        self._sessions: OrderedDict[str, tuple[object, threading.Lock]] = OrderedDict()

    @property
    def data(self) -> ResearchData:
        with self._lock:
            if self._data is None:
                try:
                    self._data = self._loader()
                except FileNotFoundError as exc:
                    raise HTTPException(503, f"Research data not found. Run `python run_pipeline.py` first. ({exc})") from exc
                self._toolkit = ResearchToolkit(self._data)
            return self._data

    @property
    def toolkit(self) -> ResearchToolkit:
        self.data  # noqa: B018 - make sure it's loaded
        return self._toolkit

    def reload(self) -> None:
        with self._lock:
            self._data, self._toolkit = None, None
            self._sessions.clear()

    def providers(self) -> list[str]:
        if self._providers is not None:
            return self._providers()
        from ..llm_agent import configured_providers

        return configured_providers()

    def session_agent(self, session_id: str, provider: str | None):
        """(agent, lock) for a chat session, or (None, lock) when no LLM is configured."""
        available = self.providers()
        provider = provider if provider in available else (available[0] if available else None)
        key = f"{session_id}:{provider}"
        with self._lock:
            if key in self._sessions:
                self._sessions.move_to_end(key)
                return self._sessions[key]
        agent = None
        if provider is not None:
            factory = self._agent_factory
            if factory is None:
                from ..llm_agent import create_agent as factory
            agent = factory(self.toolkit, provider=provider)
        entry = (agent, threading.Lock())
        with self._lock:
            self._sessions[key] = entry
            while len(self._sessions) > MAX_SESSIONS:
                self._sessions.popitem(last=False)
        return entry

    def reset_session(self, session_id: str) -> None:
        with self._lock:
            for key in [k for k in self._sessions if k.startswith(f"{session_id}:")]:
                del self._sessions[key]


def _records(df: pd.DataFrame | None, columns: list[str] | None = None) -> list[dict]:
    if df is None:
        return []
    if columns:
        df = df[[c for c in columns if c in df.columns]]
    return to_json_safe(df)


def _strategy_series(returns: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    growth = (1 + returns[[name for name in STRATEGY_LABELS if name in returns.columns]]).cumprod()
    drawdown = growth / growth.cummax() - 1
    dates = returns["date"]
    return (
        to_json_safe(pd.concat([dates, growth], axis=1)),
        to_json_safe(pd.concat([dates, drawdown], axis=1)),
    )


def _markdown_table(table: pd.DataFrame) -> str:
    header = "| " + " | ".join(map(str, table.columns)) + " |"
    divider = "| " + " | ".join("---" for _ in table.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def create_app(
    loader: Callable[[], ResearchData] = ResearchData.load,
    agent_factory: Callable | None = None,
    providers: Callable[[], list[str]] | None = None,
    static_dir: Path | None = WEB_DIST,
    demo: bool | None = None,
    limiter: ChatLimiter | None = None,
    admin_token: str | None = None,
) -> FastAPI:
    """Build the app. Demo mode (DEMO_MODE=1) rate-limits the chat and locks admin endpoints.

    Settings default to environment variables: DEMO_MODE, CHAT_LIMIT_PER_HOUR, CHAT_LIMIT_PER_DAY, ADMIN_TOKEN.
    """
    demo = _env_flag("DEMO_MODE") if demo is None else demo
    limiter = limiter or ChatLimiter(
        _env_int("CHAT_LIMIT_PER_HOUR", 20 if demo else 0), _env_int("CHAT_LIMIT_PER_DAY", 300 if demo else 0)
    )
    admin_token = admin_token if admin_token is not None else os.environ.get("ADMIN_TOKEN") or None
    app = FastAPI(title="S&P 500 Research Agent", version="0.6.0")
    service = ResearchService(loader, agent_factory, providers)
    app.state.service = service
    app.state.limiter = limiter

    @app.exception_handler(ToolInputError)
    def tool_input_error(request, exc: ToolInputError) -> JSONResponse:
        not_found = "not among" in str(exc) or "left out" in str(exc) or "No SEC CIK" in str(exc)
        return JSONResponse({"detail": str(exc)}, status_code=404 if not_found else 400)

    def tool(name: str, **arguments) -> dict:
        try:
            return service.toolkit.call(name, arguments)
        except ToolInputError as exc:
            raise HTTPException(404 if "not among" in str(exc) or "left out" in str(exc) else 400, str(exc)) from exc

    # -- overview ---------------------------------------------------------------------------------
    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/api/reload")
    def reload(x_admin_token: str | None = Header(default=None)) -> dict:
        if admin_token is not None and x_admin_token != admin_token:
            raise HTTPException(403, "Admin token required.")
        if admin_token is None and demo:
            raise HTTPException(403, "Reloading is disabled in demo mode.")
        service.reload()
        return {"status": "reloaded", "as_of": to_json_safe(service.data.scored["date"].max())}

    @app.get("/api/overview")
    def overview() -> dict:
        data = service.data
        artifacts = data.artifacts
        summary = artifacts.get("backtest_summary")
        backtest = to_json_safe(summary.set_index("strategy").to_dict(orient="index")) if summary is not None else {}
        sectors = tool("sector_summary")["sectors"] if "sector" in data.scored.columns else []
        return {
            **tool("dataset_overview"),
            "target": _spec(data.bundle)["description"],
            "horizon_days": _spec(data.bundle)["horizon"],
            "spec": _spec(data.bundle),
            "model_metrics": to_json_safe({k: v for k, v in data.bundle.get("metrics", {}).items() if k in ("auc_mean", "auc_std", "ic_mean", "ic_tstat", "accuracy", "baseline_accuracy", "oos_start", "oos_end", "folds")}),
            "backtest": backtest,
            "top": tool("rank_stocks", n=10, direction="top")["stocks"],
            "bottom": tool("rank_stocks", n=10, direction="bottom")["stocks"],
            "sectors": sectors,
            "macro": tool("macro_snapshot") if data.macro is not None else None,
        }

    # -- stocks ------------------------------------------------------------------------------------
    @app.get("/api/stocks")
    def stocks() -> dict:
        scored = service.data.scored.copy()
        total = len(scored)
        scored["company"] = scored.apply(company_label, axis=1)
        scored["stance"] = [stance(int(r), total) for r in scored["rank"]]
        return {"as_of": to_json_safe(scored["date"].max()), "target": _spec(service.data.bundle)["description"], "excluded": service.data.excluded, "stocks": _records(scored, SCREENER_FIELDS)}

    @app.get("/api/search")
    def search(q: str = Query(min_length=1, max_length=50), limit: int = 8) -> dict:
        return tool("search_companies", query=q, limit=limit)

    @app.get("/api/stocks/{ticker}")
    def stock(ticker: str) -> dict:
        toolkit = service.toolkit
        symbol = toolkit.resolve(ticker)  # raises ToolInputError (-> 404) for unknown tickers
        excluded_reason = service.data.excluded.get(symbol)
        snapshot = None if excluded_reason else tool("stock_snapshot", ticker=symbol)
        latest = service.data.features[service.data.features["ticker"] == symbol].sort_values("date").iloc[-1]
        percentiles = [
            {"signal": col, "label": label, "percentile": latest.get(col)}
            for col, label in PERCENTILE_SIGNALS.items()
            if col in latest.index and pd.notna(latest.get(col))
        ]
        company = service.data.companies
        info = company[company["ticker"] == symbol].iloc[0].to_dict() if company is not None and (company["ticker"] == symbol).any() else {}
        return to_json_safe({
            "ticker": symbol,
            "company": snapshot["company"] if snapshot else info.get("company_name", symbol),
            "sector": snapshot["sector"] if snapshot else info.get("sector"),
            "sub_industry": info.get("sub_industry"),
            "excluded_reason": excluded_reason,
            "snapshot": snapshot,
            "latest": {k: latest.get(k) for k in ["date", "close", "return_1d", "return_5d", "return_20d", "momentum_60d", "volatility_20d"]},
            "percentiles": percentiles,
            "membership": tool("index_changes", ticker=symbol) if service.data.index_changes is not None else None,
            "live": {"news": service.data.news_fetcher is not None, "filings": service.data.filings_fetcher is not None},
        })

    @app.get("/api/stocks/{ticker}/prices")
    def prices(ticker: str, days: int = Query(1260, ge=20, le=5000)) -> dict:
        symbol = service.toolkit.resolve(ticker)
        rows = service.data.features.loc[service.data.features["ticker"] == symbol, ["date", "close", "volume"] if "volume" in service.data.features else ["date", "close"]]
        return {"ticker": symbol, "prices": _records(rows.sort_values("date").tail(days))}

    @app.get("/api/stocks/{ticker}/fundamentals")
    def fundamentals(ticker: str) -> dict:
        return tool("fundamentals_history", ticker=ticker, quarters=20)

    @app.get("/api/stocks/{ticker}/news")
    def news(ticker: str) -> dict:
        try:
            return tool("recent_news", ticker=ticker, limit=15)
        except HTTPException:
            raise
        except Exception as exc:  # network trouble with a live source
            raise HTTPException(502, f"News source unavailable: {exc}") from exc

    @app.get("/api/stocks/{ticker}/filings")
    def filings(ticker: str) -> dict:
        try:
            return tool("recent_filings", ticker=ticker, limit=15)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(502, f"SEC EDGAR unavailable: {exc}") from exc

    # -- research ------------------------------------------------------------------------------------
    @app.get("/api/backtest")
    def backtest() -> dict:
        artifacts = service.data.artifacts
        returns = artifacts.get("backtest_returns")
        if returns is None:
            raise HTTPException(404, "No backtest yet. Run python run_pipeline.py.")
        growth, drawdown = _strategy_series(returns)
        periods_per_year = TRADING_DAYS_PER_YEAR / (artifacts.get("backtest_config") or {}).get("holding_days", _spec(service.data.bundle)["horizon"])
        rolling_ic = returns.set_index("date")["ic"].rolling(12, min_periods=6).mean().reset_index().rename(columns={"ic": "ic_rolling"})
        return {
            "config": artifacts.get("backtest_config"),
            "labels": {k: v for k, v in STRATEGY_LABELS.items() if k in returns.columns},
            "attribution": _records(artifacts.get("attribution")),
            "periods_per_year": periods_per_year,
            "summary": _records(artifacts.get("backtest_summary")),
            "growth": growth,
            "drawdown": drawdown,
            "quintiles": _records(artifacts.get("quantile_returns")),
            "ic": to_json_safe(pd.concat([returns[["date", "ic"]], rolling_ic["ic_rolling"]], axis=1)),
        }

    @app.get("/api/model")
    def model() -> dict:
        artifacts = service.data.artifacts
        if artifacts.get("model_comparison") is None:
            raise HTTPException(404, "No model research yet. Run python run_pipeline.py.")
        eda = artifacts.get("eda") or {}
        return {
            "selected": service.data.bundle.get("model_name"),
            "target": _spec(service.data.bundle)["description"],
            "spec": _spec(service.data.bundle),
            "experiments": _records(artifacts.get("experiments")),
            "features": eda.get("model_features", service.data.bundle.get("numeric", [])),
            "comparison": _records(artifacts.get("model_comparison")),
            "folds": _records(artifacts.get("fold_metrics")),
            "calibration": _records(artifacts.get("calibration")),
            "importance": _records(artifacts.get("feature_importance")),
            "signals": _records(artifacts.get("signal_ic")),
            "report_available": bool(artifacts.get("report")),
        }

    @app.get("/api/model/report", response_model=None)
    def report():
        text = service.data.artifacts.get("report")
        if not text:
            raise HTTPException(404, "No report yet.")
        return StreamingResponse(iter([text]), media_type="text/markdown", headers={"Content-Disposition": "attachment; filename=research_report.md"})

    @app.get("/api/data")
    def data_quality() -> dict:
        data = service.data
        macro = data.macro
        macro_series = []
        if macro is not None and not macro.empty:
            recent = macro[macro["date"] >= macro["date"].max() - pd.Timedelta(days=3653)].copy()
            macro_series = _records(recent.iloc[::2])  # every other day keeps 10 years light
        changes = data.index_changes.sort_values("date", ascending=False).head(25) if data.index_changes is not None else None
        return {
            "source": data.source,
            "quality": data.quality,
            "eda": data.artifacts.get("eda"),
            "macro": macro_series,
            "index_changes": _records(changes, ["date", "added", "added_name", "removed", "removed_name", "reason"]),
            "excluded": data.excluded,
        }

    # -- agent -----------------------------------------------------------------------------------------
    @app.get("/api/agent/status")
    def agent_status() -> dict:
        available = service.providers()
        return {
            "providers": available,
            "mode": "llm" if available else "rule-based",
            "demo": demo,
            "limits": {"per_hour": limiter.per_hour, "per_day": limiter.per_day},
        }

    @app.post("/api/agent/reset")
    def agent_reset(request: ResetRequest) -> dict:
        service.reset_session(request.session_id)
        return {"status": "reset"}

    @app.post("/api/agent/chat")
    def agent_chat(request: ChatRequest, http: Request) -> StreamingResponse:
        # Behind a hosting proxy the visitor's address is the first X-Forwarded-For entry.
        client = (http.headers.get("x-forwarded-for", "").split(",")[0].strip()) or (http.client.host if http.client else "unknown")
        refusal = limiter.check(client)
        if refusal:
            raise HTTPException(429, refusal)
        events: queue.Queue = queue.Queue()

        def work() -> None:
            try:
                agent, lock = service.session_agent(request.session_id, request.provider)
                with lock:
                    if agent is None:
                        from ..chat import answer_message

                        answer = answer_message(request.message, service.data.scored)
                        text = answer.text + (("\n\n" + _markdown_table(answer.table)) if answer.table is not None else "")
                        events.put({"type": "answer", "text": text, "provider": "rule-based", "model": None, "usage": {}, "ticker": answer.ticker})
                        return
                    agent.on_event = events.put
                    try:
                        reply = agent.ask(request.message)
                    finally:
                        agent.on_event = None
                    events.put({"type": "answer", "text": reply.text, "provider": reply.provider, "model": reply.model, "usage": reply.usage, "stop_reason": reply.stop_reason})
            except Exception as exc:
                events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            finally:
                events.put(None)

        threading.Thread(target=work, daemon=True).start()

        def stream():
            yield ": connected\n\n"
            while (event := events.get()) is not None:
                yield f"data: {json.dumps(to_json_safe(event), allow_nan=False)}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # -- frontend --------------------------------------------------------------------------------------
    if static_dir is not None and (static_dir / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404, "Not found")
            candidate = (static_dir / path).resolve()
            if path and candidate.is_file() and static_dir.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(static_dir / "index.html")  # client-side routes

    return app


def _app_factory() -> FastAPI:
    from ..config import load_environment

    load_environment()
    return create_app()


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve the S&P 500 research dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Restart on code changes (development).")
    args = parser.parse_args()
    if not (WEB_DIST / "index.html").exists():
        print("Note: web/dist not found, serving the API only. Build the dashboard with: cd web && npm install && npm run build")
    print(f"Dashboard: http://{args.host}:{args.port}")
    uvicorn.run("sp500_agent.api.server:_app_factory", factory=True, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
