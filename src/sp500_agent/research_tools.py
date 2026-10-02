"""Research tools the LLM agent can call.

Each tool is a method returning a JSON-serialisable dict. TOOL_DEFINITIONS describes them to the model;
ResearchToolkit.call validates the arguments and runs them. Nothing here talks to the API, so the tools
are tested directly.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, TARGET_DESCRIPTION, TRADING_DAYS_PER_YEAR, load_environment
from .features import SNAPSHOT_FIELDS, _lexicon_sentiment, load_features, load_news_headlines
from .formatting import COMPANY_NAME_COLUMNS, company_label, stance
from .model import load_model, score_latest
from .research import load_artifacts

SIGNAL_FIELDS = [
    "close",
    "return_5d",
    "return_20d",
    "momentum_60d",
    "volatility_20d",
    "ma_gap_50d",
    "sentiment_20d",
    "news_count_20d",
    "return_20d_vs_sector",
]
SORTABLE_FIELDS = ["model_probability", "return_5d", "return_20d", "momentum_60d", "volatility_20d", "ma_gap_50d", "sentiment_20d", "market_cap", "pe_ratio"]
MAX_ROWS = 25
SUSPECT_DAILY_MOVE = 0.4
MAX_COMPARE = 8
CHART_POINTS = 60


class ToolInputError(ValueError):
    """Raised for bad tool arguments; the message is returned to the model so it can correct itself."""


def to_json_safe(value):
    """Convert pandas/numpy values to plain JSON types; NaN and infinities become None."""
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return [to_json_safe(row) for row in value.to_dict(orient="records")]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return None if pd.isna(value) else value.date().isoformat() if isinstance(value, (pd.Timestamp, datetime)) else value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else round(float(value), 6)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None or value is pd.NaT or (not isinstance(value, str) and pd.isna(value)):
        return None
    return value


@dataclass
class ResearchData:
    features: pd.DataFrame
    bundle: dict
    news: pd.DataFrame = field(default_factory=pd.DataFrame)
    artifacts: dict = field(default_factory=dict)
    fundamentals: pd.DataFrame | None = None  # point-in-time SEC data (live source only)
    macro: pd.DataFrame | None = None
    membership: pd.DataFrame | None = None
    index_changes: pd.DataFrame | None = None
    companies: pd.DataFrame | None = None
    quality: dict = field(default_factory=dict)
    news_fetcher: object | None = None  # fetches current headlines on demand (sources.news.NewsFetcher)
    filings_fetcher: object | None = None  # fetches recent SEC filings on demand (sources.sec.FilingsFetcher)
    scored: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        self.scored = score_latest(self.features, self.bundle)
        self.excluded: dict[str, str] = dict(self.scored.attrs.get("excluded", {}))

    @property
    def source(self) -> str:
        return self.quality.get("source", "unknown")

    @classmethod
    def load(cls, directory=PROCESSED_DIR, live_fetchers: bool = True) -> "ResearchData":
        def optional(name):
            path = directory / f"{name}.parquet"
            return pd.read_parquet(path) if path.exists() else None

        quality_path = directory / "data_quality.json"
        data = cls(
            load_features(),
            load_model(),
            load_news_headlines(),
            load_artifacts(),
            fundamentals=optional("fundamentals"),
            macro=optional("macro"),
            membership=optional("membership"),
            index_changes=optional("index_changes"),
            companies=optional("companies"),
            quality=json.loads(quality_path.read_text()) if quality_path.exists() else {},
        )
        if live_fetchers and data.source == "live":
            from .sources.news import NewsFetcher
            from .sources.sec import FilingsFetcher

            load_environment()
            data.news_fetcher = NewsFetcher()
            if data.companies is not None and "cik" in data.companies:
                ciks = {t: int(c) for t, c in zip(data.companies["ticker"], data.companies["cik"]) if pd.notna(c)}
                data.filings_fetcher = FilingsFetcher.from_env(ciks)
        return data


TOOL_DEFINITIONS = [
    {
        "name": "dataset_overview",
        "description": "Start here. Returns the as-of date of the data, how many stocks are covered, the date range, which model is in use, and its out-of-sample quality. Call this before quoting dates or model reliability.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "search_companies",
        "description": "Find tickers by company name or partial ticker, e.g. 'apple' or 'bank'. Use when the user names a company instead of giving a ticker.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Company name or ticker fragment."},
                "limit": {"type": "integer", "description": "Maximum matches (1-25). Default 5."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "stock_snapshot",
        "description": "Everything known about one stock on the latest date: the model's probability and rank, price and momentum signals, volatility, news sentiment, and a fundamentals snapshot (market cap, P/E, margins).",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Ticker symbol, e.g. AAPL."}},
            "required": ["ticker"],
            "additionalProperties": False,
        },
    },
    {
        "name": "rank_stocks",
        "description": f"Stocks with the highest ('top') or lowest ('bottom') model probability to {TARGET_DESCRIPTION}, optionally within one sector.",
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "How many stocks (1-25). Default 10."},
                "direction": {"type": "string", "enum": ["top", "bottom"], "description": "Default 'top'."},
                "sector": {"type": "string", "description": "Optional sector name, e.g. 'Technology'. Use sector_summary to see valid names."},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "screen_stocks",
        "description": "Filter stocks on the latest date by sector, model probability, volatility, momentum and P/E, then sort. All filters are optional. Volatility is annualised (0.3 = 30%). Returns and momentum are decimals (0.05 = 5%).",
        "input_schema": {
            "type": "object",
            "properties": {
                "sector": {"type": "string"},
                "min_probability": {"type": "number", "description": "Minimum model probability, 0-1."},
                "max_volatility": {"type": "number", "description": "Maximum annualised volatility, e.g. 0.35."},
                "min_momentum_60d": {"type": "number", "description": "Minimum 60-day return, e.g. 0.1 for +10%."},
                "max_pe_ratio": {"type": "number"},
                "sort_by": {"type": "string", "enum": SORTABLE_FIELDS, "description": "Default model_probability."},
                "ascending": {"type": "boolean", "description": "Sort ascending. Default false."},
                "limit": {"type": "integer", "description": "Maximum rows (1-25). Default 15."},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "compare_stocks",
        "description": "Side-by-side signals and fundamentals for 2-8 tickers.",
        "input_schema": {
            "type": "object",
            "properties": {"tickers": {"type": "array", "items": {"type": "string"}, "description": "2 to 8 ticker symbols."}},
            "required": ["tickers"],
            "additionalProperties": False,
        },
    },
    {
        "name": "price_history",
        "description": "Price path and risk statistics for one stock over the last N trading days: total return, annualised volatility, maximum drawdown, distance from the high, and a down-sampled series of closes. The app draws a chart from this.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "days": {"type": "integer", "description": "Trading days to look back (20-1260). Default 252, about one year."},
            },
            "required": ["ticker"],
            "additionalProperties": False,
        },
    },
    {
        "name": "recent_news",
        "description": "Most recent news headlines for a stock with a simple word-list sentiment score (-1 negative to +1 positive). With live data the headlines are fetched now; otherwise they come from the dataset.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}, "limit": {"type": "integer", "description": "1-20. Default 8."}},
            "required": ["ticker"],
            "additionalProperties": False,
        },
    },
    {
        "name": "sector_summary",
        "description": "Per-sector averages on the latest date: number of stocks, mean model probability, mean 20-day return and volatility, and the highest-ranked stock in each sector.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "fundamentals_history",
        "description": "Point-in-time fundamentals from SEC filings for one stock: trailing-12-month revenue and net income, revenue growth, equity, liabilities and shares, each with the date it became public. Use for questions about growth, profitability or balance sheets over time.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}, "quarters": {"type": "integer", "description": "How many recent filings (1-20). Default 8."}},
            "required": ["ticker"],
            "additionalProperties": False,
        },
    },
    {
        "name": "macro_snapshot",
        "description": "Current market conditions from FRED: VIX (expected volatility), 3-month T-bill and 10-year Treasury yields, and the yield-curve spread, compared with a year ago and with their 10-year history.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "index_changes",
        "description": "S&P 500 additions and removals. With a ticker: when it joined or left the index. Without: the most recent changes.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}, "limit": {"type": "integer", "description": "1-25. Default 10."}},
            "additionalProperties": False,
        },
    },
    {
        "name": "recent_filings",
        "description": "A company's most recent SEC filings (10-K annual reports, 10-Q quarterly reports, 8-K current reports such as earnings releases or leadership changes) with dates and links. Fetched live from SEC EDGAR.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "form": {"type": "string", "enum": ["10-K", "10-Q", "8-K"], "description": "Only this form type. Default: all three."},
                "limit": {"type": "integer", "description": "1-20. Default 8."},
            },
            "required": ["ticker"],
            "additionalProperties": False,
        },
    },
    {
        "name": "model_performance",
        "description": "How good the model really is: walk-forward comparison of candidate models (AUC, accuracy vs. baseline, information coefficient), calibration, the most important features, and how single signals compare. Use whenever the user asks whether to trust the predictions.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "backtest_results",
        "description": "Out-of-sample portfolio backtest of the model's rankings: long-only top quintile, long-short, and an equal-weight benchmark, with return, Sharpe, drawdown, turnover, returns by prediction quintile, and the backtest's assumptions and caveats.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]
TOOL_NAMES = {tool["name"] for tool in TOOL_DEFINITIONS}


def _int_arg(args: dict, key: str, default: int, low: int, high: int) -> int:
    value = args.get(key, default)
    if value is None:
        return default
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ToolInputError(f"{key} must be an integer.") from exc
    return min(max(value, low), high)


def _float_arg(args: dict, key: str) -> float | None:
    value = args.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ToolInputError(f"{key} must be a number.") from exc


class ResearchToolkit:
    def __init__(self, data: ResearchData):
        self.data = data
        self.scored = data.scored
        self._by_ticker = self.scored.set_index("ticker", drop=False)

    # -- dispatch -------------------------------------------------------------------------------
    def call(self, name: str, arguments: dict | None) -> dict:
        if name not in TOOL_NAMES:
            raise ToolInputError(f"Unknown tool {name!r}. Available: {', '.join(sorted(TOOL_NAMES))}.")
        arguments = dict(arguments or {})
        schema = next(tool["input_schema"] for tool in TOOL_DEFINITIONS if tool["name"] == name)
        unknown = set(arguments) - set(schema["properties"])
        if unknown:
            raise ToolInputError(f"Unexpected argument(s) for {name}: {', '.join(sorted(unknown))}.")
        missing = [key for key in schema.get("required", []) if arguments.get(key) in (None, "", [])]
        if missing:
            raise ToolInputError(f"Missing required argument(s) for {name}: {', '.join(missing)}.")
        return to_json_safe(getattr(self, name)(**arguments))

    # -- helpers --------------------------------------------------------------------------------
    def _symbol(self, ticker: str) -> str:
        """Any ticker with data (ranked or not), for history, news, filings and fundamentals lookups."""
        symbol = str(ticker).strip().upper().lstrip("$").replace("-", ".")
        if symbol in self._by_ticker.index or symbol in self.data.excluded:
            return symbol
        known = self._known_tickers()
        for candidate in (symbol, symbol.replace(".", "-")):
            if candidate in known:
                return candidate
        return self._row(ticker)["ticker"]  # raises with suggestions

    def resolve(self, ticker: str) -> str:
        """Public form of _symbol: the canonical ticker for any stock with data, or ToolInputError."""
        return self._symbol(ticker)

    def _known_tickers(self) -> set:
        if not hasattr(self, "_known"):
            self._known = set(self.data.features["ticker"].unique())
        return self._known

    def _row(self, ticker: str) -> pd.Series:
        symbol = str(ticker).strip().upper().lstrip("$").replace("-", ".")
        if symbol in self._by_ticker.index:
            return self._by_ticker.loc[symbol]
        alternative = symbol.replace(".", "-")
        if alternative in self._by_ticker.index:
            return self._by_ticker.loc[alternative]
        if symbol in self.data.excluded:
            raise ToolInputError(
                f"{symbol} is left out of today's ranking because of {self.data.excluded[symbol]}. "
                "Its price history, news, filings and fundamentals are still available."
            )
        suggestions = [m["ticker"] for m in self.search_companies(ticker, limit=5)["matches"]]
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else " Use search_companies to find the ticker."
        raise ToolInputError(f"{ticker!r} is not among the {len(self.scored)} ranked stocks.{hint}")

    def _sector_rows(self, sector: str | None) -> pd.DataFrame:
        if not sector:
            return self.scored
        if "sector" not in self.scored.columns:
            raise ToolInputError("This dataset has no sector information.")
        mask = self.scored["sector"].astype(str).str.lower() == sector.strip().lower()
        if not mask.any():
            mask = self.scored["sector"].astype(str).str.lower().str.contains(sector.strip().lower(), regex=False)
        if not mask.any():
            sectors = sorted(self.scored["sector"].dropna().astype(str).unique())
            raise ToolInputError(f"Unknown sector {sector!r}. Valid sectors: {', '.join(sectors)}.")
        return self.scored[mask]

    def _summary_row(self, row: pd.Series) -> dict:
        return {
            "ticker": row["ticker"],
            "company": company_label(row),
            "sector": row.get("sector"),
            "rank": row["rank"],
            "model_probability": row["model_probability"],
            "return_5d": row.get("return_5d"),
            "return_20d": row.get("return_20d"),
            "momentum_60d": row.get("momentum_60d"),
            "volatility_20d": row.get("volatility_20d"),
        }

    # -- tools ----------------------------------------------------------------------------------
    def dataset_overview(self) -> dict:
        features, bundle = self.data.features, self.data.bundle
        metrics = bundle.get("metrics", {})
        return {
            "as_of_date": self.scored["date"].max(),
            "stocks_ranked": len(self.scored),
            "history_start": features["date"].min(),
            "history_end": features["date"].max(),
            "sectors": sorted(self.scored["sector"].dropna().astype(str).unique()) if "sector" in self.scored else [],
            "model": bundle.get("model_name"),
            "model_trained_through": bundle.get("trained_through"),
            "prediction_target": f"probability that the stock will {TARGET_DESCRIPTION}",
            "walk_forward_auc": metrics.get("auc_mean"),
            "walk_forward_ic_mean": metrics.get("ic_mean"),
            "accuracy": metrics.get("accuracy"),
            "baseline_accuracy": metrics.get("baseline_accuracy"),
            "data_source": self.data.source,
            "left_out_of_ranking": self.data.excluded,
            "data_quality": self._quality_summary(),
            "live_lookups": {
                "news": getattr(self.data.news_fetcher, "source", None) or "dataset headlines only",
                "sec_filings": "available" if self.data.filings_fetcher else "unavailable (set SEC_USER_AGENT and use live data)",
            },
            "reading_guide": "AUC 0.5 = no skill. Daily stock direction is close to a coin flip; small edges are normal and fragile.",
        }

    def _quality_summary(self) -> dict:
        q = self.data.quality
        out = {"price_tickers": q.get("prices", {}).get("tickers"), "price_end": q.get("prices", {}).get("end")}
        if "survivorship" in q:
            out["former_members_with_prices"] = f"{q['survivorship']['former_members_with_prices']} of {q['survivorship']['former_members']}"
        if "fundamentals" in q:
            out["tickers_with_sec_fundamentals"] = q["fundamentals"]["tickers_with_data"]
        return out

    def search_companies(self, query: str, limit: int = 5) -> dict:
        limit = _int_arg({"limit": limit}, "limit", 5, 1, MAX_ROWS)
        q = str(query).strip().upper()
        if not q:
            raise ToolInputError("query must not be empty.")
        names = self.scored.apply(company_label, axis=1).str.upper()
        tickers = self.scored["ticker"].astype(str).str.upper()
        score = (
            (tickers == q) * 4
            + tickers.str.startswith(q) * 2
            + names.str.contains(q, regex=False) * 3
            + names.str.split().map(lambda words: any(w.startswith(q) for w in words)) * 1
        )
        hits = self.scored[score > 0].assign(_score=score[score > 0]).sort_values(["_score", "rank"], ascending=[False, True])
        return {"query": query, "matches": [{"ticker": r["ticker"], "company": company_label(r), "sector": r.get("sector")} for _, r in hits.head(limit).iterrows()]}

    def _data_warning(self, ticker: str) -> str | None:
        """Flag a recent daily move so large it may be an unadjusted corporate action rather than a real return."""
        recent = self.data.features[self.data.features["ticker"] == ticker].sort_values("date").tail(20)
        moves = recent["return_1d"].abs() if "return_1d" in recent else pd.Series(dtype=float)
        if moves.isna().all() or moves.max() <= SUSPECT_DAILY_MOVE:
            return None
        worst = recent.loc[moves.idxmax()]
        return (
            f"A daily move of {worst['return_1d']:+.0%} on {worst['date'].date()} may be a corporate action (spin-off, split, merger) "
            "that the price source hasn't adjusted for yet. Treat recent returns and the model's view with caution."
        )

    def stock_snapshot(self, ticker: str) -> dict:
        row = self._row(ticker)
        return {
            "data_warning": self._data_warning(row["ticker"]),
            "ticker": row["ticker"],
            "company": company_label(row),
            "sector": row.get("sector"),
            "as_of_date": row["date"],
            "model_probability": row["model_probability"],
            "model_view": stance(int(row["rank"]), len(self.scored)),
            "rank": row["rank"],
            "out_of": len(self.scored),
            "signals": {col: row.get(col) for col in SIGNAL_FIELDS if col in row.index},
            "fundamentals": {col: row.get(col) for col in SNAPSHOT_FIELDS if col in row.index and pd.notna(row.get(col))},
            "fundamentals_as_of": row.get("fundamentals_as_of"),
            "notes": "Volatility is annualised. Returns are decimals. "
            + (
                "Fundamentals are trailing-12-month figures from SEC filings public on fundamentals_as_of; their cross-sectional ranks are model inputs."
                if "fundamentals_as_of" in row.index
                else "Fundamentals are a single current snapshot and are not model inputs."
            ),
        }

    def rank_stocks(self, n: int = 10, direction: str = "top", sector: str | None = None) -> dict:
        n = _int_arg({"n": n}, "n", 10, 1, MAX_ROWS)
        if direction not in ("top", "bottom"):
            raise ToolInputError("direction must be 'top' or 'bottom'.")
        rows = self._sector_rows(sector).sort_values("model_probability", ascending=direction == "bottom").head(n)
        return {"direction": direction, "sector": sector, "as_of_date": self.scored["date"].max(), "stocks": [self._summary_row(r) for _, r in rows.iterrows()]}

    def screen_stocks(
        self,
        sector: str | None = None,
        min_probability: float | None = None,
        max_volatility: float | None = None,
        min_momentum_60d: float | None = None,
        max_pe_ratio: float | None = None,
        sort_by: str = "model_probability",
        ascending: bool = False,
        limit: int = 15,
    ) -> dict:
        args = locals()
        rows = self._sector_rows(sector)
        for key, column, keep_if in [
            ("min_probability", "model_probability", lambda s, v: s >= v),
            ("max_volatility", "volatility_20d", lambda s, v: s <= v),
            ("min_momentum_60d", "momentum_60d", lambda s, v: s >= v),
            ("max_pe_ratio", "pe_ratio", lambda s, v: (s <= v) & (s > 0)),
        ]:
            value = _float_arg(args, key)
            if value is not None:
                if column not in rows.columns:
                    raise ToolInputError(f"{column} is not available in this dataset.")
                rows = rows[keep_if(rows[column], value).fillna(False)]
        if sort_by not in SORTABLE_FIELDS or sort_by not in rows.columns:
            raise ToolInputError(f"sort_by must be one of: {', '.join(c for c in SORTABLE_FIELDS if c in self.scored.columns)}.")
        limit = _int_arg({"limit": limit}, "limit", 15, 1, MAX_ROWS)
        rows = rows.sort_values(sort_by, ascending=bool(ascending), na_position="last")
        out = []
        for _, row in rows.head(limit).iterrows():
            item = self._summary_row(row)
            item["pe_ratio"] = row.get("pe_ratio")
            out.append(item)
        return {"filters": {k: v for k, v in args.items() if k not in ("self",) and v is not None}, "matches": len(rows), "stocks": out}

    def compare_stocks(self, tickers: list[str]) -> dict:
        if not isinstance(tickers, list) or not 2 <= len(tickers) <= MAX_COMPARE:
            raise ToolInputError(f"tickers must be a list of 2 to {MAX_COMPARE} symbols.")
        return {"stocks": [self.stock_snapshot(t) for t in dict.fromkeys(tickers)]}

    def price_history(self, ticker: str, days: int = 252) -> dict:
        days = _int_arg({"days": days}, "days", 252, 20, 1260)
        symbol = self._symbol(ticker)
        history = self.data.features.loc[self.data.features["ticker"] == symbol, ["date", "close"]].sort_values("date").tail(days)
        close = history["close"].reset_index(drop=True)
        daily = close.pct_change().dropna()
        step = max(1, math.ceil(len(history) / CHART_POINTS))
        sampled = pd.concat([history.iloc[::step], history.iloc[[-1]]]).drop_duplicates("date")
        return {
            "ticker": symbol,
            "start_date": history["date"].iloc[0],
            "end_date": history["date"].iloc[-1],
            "trading_days": len(history),
            "start_close": close.iloc[0],
            "end_close": close.iloc[-1],
            "total_return": close.iloc[-1] / close.iloc[0] - 1,
            "annualised_volatility": daily.std() * math.sqrt(TRADING_DAYS_PER_YEAR) if len(daily) > 1 else None,
            "max_drawdown": float((close / close.cummax() - 1).min()),
            "high": close.max(),
            "low": close.min(),
            "distance_from_high": close.iloc[-1] / close.max() - 1,
            "series": [{"date": d, "close": c} for d, c in zip(sampled["date"], sampled["close"])],
        }

    def recent_news(self, ticker: str, limit: int = 8) -> dict:
        limit = _int_arg({"limit": limit}, "limit", 8, 1, 20)
        symbol = self._symbol(ticker)
        if self.data.news_fetcher is not None:
            live = self.data.news_fetcher.fetch(symbol).head(limit)
            live = live.assign(sentiment_score=(live["title"].fillna("") + " " + live["summary"].fillna("")).map(_lexicon_sentiment))
            return {"ticker": symbol, "fetched_from": self.data.news_fetcher.source, "headlines": live[["date", "title", "sentiment_score", "source", "url"]]}
        news = self.data.news
        if news is None or news.empty:
            return {"ticker": symbol, "headlines": [], "note": "No headlines in this dataset. Live news is available when the pipeline uses live data."}
        rows = news[news["ticker"] == symbol].sort_values("date", ascending=False).head(limit)
        keep = [c for c in ["date", "title", "sentiment_score", "source"] if c in rows.columns]
        return {"ticker": symbol, "headlines": rows[keep]}

    def fundamentals_history(self, ticker: str, quarters: int = 8) -> dict:
        quarters = _int_arg({"quarters": quarters}, "quarters", 8, 1, 20)
        symbol = self._symbol(ticker)
        table = self.data.fundamentals
        if table is None or table.empty:
            return {"ticker": symbol, "history": [], "note": "No point-in-time fundamentals in this dataset; they come from SEC EDGAR with the live data source."}
        rows = table[table["ticker"] == symbol].sort_values("available_date").tail(quarters)
        rows = rows.assign(profit_margin=(rows["net_income_ttm"] / rows["revenue_ttm"]).where(rows["revenue_ttm"] > 0))
        columns = ["available_date", "period_end", "revenue_ttm", "net_income_ttm", "profit_margin", "revenue_growth_yoy", "equity", "liabilities", "shares_outstanding"]
        return {
            "ticker": symbol,
            "history": rows[columns].iloc[::-1],
            "notes": "available_date is when the filing became public; values are as first reported (no later restatements). _ttm = trailing twelve months.",
        }

    def macro_snapshot(self) -> dict:
        macro = self.data.macro
        if macro is None or macro.empty:
            return {"note": "No macro data in this dataset; FRED series come with the live data source."}
        macro = macro.sort_values("date").ffill()
        latest = macro.iloc[-1]
        year_ago = macro[macro["date"] <= latest["date"] - pd.Timedelta(days=365)].iloc[-1] if (macro["date"] <= latest["date"] - pd.Timedelta(days=365)).any() else None
        decade = macro[macro["date"] >= latest["date"] - pd.Timedelta(days=3652)]
        out = {"as_of": latest["date"], "series": {}}
        for col, label in [("vix", "VIX"), ("tbill_3m", "3-month T-bill yield, %"), ("treasury_10y", "10-year Treasury yield, %"), ("term_spread", "10y minus 3m, percentage points")]:
            if col in macro.columns:
                out["series"][col] = {
                    "label": label,
                    "latest": latest[col],
                    "one_year_ago": year_ago[col] if year_ago is not None else None,
                    "percentile_10y": float((decade[col] <= latest[col]).mean()),
                }
        out["notes"] = "A negative term spread (inverted yield curve) has often preceded recessions. VIX above ~30 signals stressed markets."
        return out

    def index_changes(self, ticker: str | None = None, limit: int = 10) -> dict:
        limit = _int_arg({"limit": limit}, "limit", 10, 1, MAX_ROWS)
        changes = self.data.index_changes
        if changes is None or changes.empty:
            return {"note": "No index-membership data in this dataset; it comes from Wikipedia with the live data source."}
        columns = ["date", "added", "added_name", "removed", "removed_name", "reason"]
        if not ticker:
            return {"recent_changes": changes.sort_values("date", ascending=False).head(limit)[columns]}
        symbol = str(ticker).strip().upper().replace("-", ".")
        involved = changes[(changes["added"] == symbol) | (changes["removed"] == symbol)].sort_values("date", ascending=False)
        intervals = self.data.membership[self.data.membership["ticker"] == symbol] if self.data.membership is not None else pd.DataFrame()
        return {
            "ticker": symbol,
            "currently_in_index": bool((intervals["end"].isna()).any()) if not intervals.empty else False,
            "membership_periods": [{"from": r.start, "to": r.end} for r in intervals.itertuples()],
            "changes": involved.head(limit)[columns],
            "note": "A missing 'from' date means the stock was already a member when the reconstructed history begins.",
        }

    def recent_filings(self, ticker: str, form: str | None = None, limit: int = 8) -> dict:
        limit = _int_arg({"limit": limit}, "limit", 8, 1, 20)
        symbol = self._symbol(ticker)
        if self.data.filings_fetcher is None:
            return {"ticker": symbol, "filings": [], "note": "SEC filings lookups need the live data source and SEC_USER_AGENT in .env."}
        if form is not None and form not in ("10-K", "10-Q", "8-K"):
            raise ToolInputError("form must be one of 10-K, 10-Q, 8-K.")
        try:
            filings = self.data.filings_fetcher.fetch(symbol, forms=(form,) if form else ("10-K", "10-Q", "8-K"))
        except KeyError as exc:
            raise ToolInputError(f"No SEC CIK is known for {symbol}.") from exc
        return {
            "ticker": symbol,
            "filings": filings.head(limit)[["form", "filed", "report_date", "items", "url"]],
            "notes": "8-K 'items' codes: 2.02 results of operations (earnings), 5.02 executive or director changes, 1.01 material agreements, 8.01 other events.",
        }

    def sector_summary(self) -> dict:
        if "sector" not in self.scored.columns:
            raise ToolInputError("This dataset has no sector information.")
        grouped = self.scored.groupby("sector")
        table = grouped.agg(
            stocks=("ticker", "count"),
            mean_up_probability=("model_probability", "mean"),
            mean_return_20d=("return_20d", "mean"),
            mean_volatility=("volatility_20d", "mean"),
        )
        table["top_ranked"] = grouped.apply(lambda g: g.sort_values("rank")["ticker"].iloc[0])
        return {"as_of_date": self.scored["date"].max(), "sectors": table.sort_values("mean_up_probability", ascending=False).reset_index()}

    def model_performance(self) -> dict:
        artifacts = self.data.artifacts
        comparison = artifacts.get("model_comparison")
        if comparison is None:
            return {"note": "No research artifacts found. Run python run_pipeline.py to produce them.", "metrics": self.data.bundle.get("metrics", {})}
        importance = artifacts.get("feature_importance")
        signals = artifacts.get("signal_ic")
        return {
            "selected_model": self.data.bundle.get("model_name"),
            "model_comparison": comparison[["model", "auc_mean", "auc_std", "accuracy", "baseline_accuracy", "brier", "ic_mean", "ic_tstat"]],
            "calibration": artifacts.get("calibration"),
            "top_features": importance.head(8) if importance is not None else [],
            "single_signal_ic": signals.head(8) if signals is not None else [],
            "how_to_read": "AUC above ~0.52-0.53 and an IC t-stat above 2 (computed on non-overlapping dates) would suggest a real but small edge. Accuracy should be compared with baseline_accuracy.",
        }

    def backtest_results(self) -> dict:
        artifacts = self.data.artifacts
        summary = artifacts.get("backtest_summary")
        if summary is None:
            return {"note": "No backtest found. Run python run_pipeline.py to produce it."}
        returns = artifacts.get("backtest_returns")
        return {
            "config": artifacts.get("backtest_config"),
            "period": {"start": returns["date"].min(), "end": returns["date"].max(), "rebalances": len(returns)} if returns is not None else None,
            "summary": summary,
            "returns_by_prediction_quintile": artifacts.get("quantile_returns"),
            "caveats": [
                "Survivorship bias: only current index members are in the data, which flatters long strategies.",
                "Costs are a flat charge per unit of weight traded; no market impact or short-borrow fees.",
                "Positions are entered one session after the signal and held for 5 sessions.",
            ],
        }
