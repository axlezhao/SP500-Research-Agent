"""Research tools the LLM agent can call.

Each tool is a method returning a JSON-serialisable dict. TOOL_DEFINITIONS describes them to the model;
ResearchToolkit.call validates the arguments and runs them. Nothing here talks to the API, so the tools
are tested directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

from .config import TRADING_DAYS_PER_YEAR
from .features import SNAPSHOT_FIELDS, load_features, load_news_headlines
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
SORTABLE_FIELDS = ["up_probability_5d", "return_5d", "return_20d", "momentum_60d", "volatility_20d", "ma_gap_50d", "sentiment_20d", "market_cap", "pe_ratio"]
MAX_ROWS = 25
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
    scored: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        self.scored = score_latest(self.features, self.bundle)

    @classmethod
    def load(cls) -> "ResearchData":
        return cls(load_features(), load_model(), load_news_headlines(), load_artifacts())


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
        "description": "Everything known about one stock on the latest date: model probability of a 5-day rise, its rank, price and momentum signals, volatility, news sentiment, and a fundamentals snapshot (market cap, P/E, margins).",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Ticker symbol, e.g. AAPL."}},
            "required": ["ticker"],
            "additionalProperties": False,
        },
    },
    {
        "name": "rank_stocks",
        "description": "Stocks with the highest ('top') or lowest ('bottom') model probability of rising over the next 5 trading days, optionally within one sector.",
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
                "sort_by": {"type": "string", "enum": SORTABLE_FIELDS, "description": "Default up_probability_5d."},
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
        "description": "Most recent news headlines for a stock with their sentiment scores (-1 negative to +1 positive).",
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
    def _row(self, ticker: str) -> pd.Series:
        symbol = str(ticker).strip().upper().lstrip("$").replace("-", ".")
        if symbol in self._by_ticker.index:
            return self._by_ticker.loc[symbol]
        alternative = symbol.replace(".", "-")
        if alternative in self._by_ticker.index:
            return self._by_ticker.loc[alternative]
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
            "up_probability_5d": row["up_probability_5d"],
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
            "prediction_target": "probability that the stock's close is higher 5 trading days later",
            "walk_forward_auc": metrics.get("auc_mean"),
            "walk_forward_ic_mean": metrics.get("ic_mean"),
            "accuracy": metrics.get("accuracy"),
            "baseline_accuracy": metrics.get("baseline_accuracy"),
            "reading_guide": "AUC 0.5 = no skill. Daily stock direction is close to a coin flip; small edges are normal and fragile.",
        }

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

    def stock_snapshot(self, ticker: str) -> dict:
        row = self._row(ticker)
        return {
            "ticker": row["ticker"],
            "company": company_label(row),
            "sector": row.get("sector"),
            "as_of_date": row["date"],
            "up_probability_5d": row["up_probability_5d"],
            "model_view": stance(float(row["up_probability_5d"])),
            "rank": row["rank"],
            "out_of": len(self.scored),
            "signals": {col: row.get(col) for col in SIGNAL_FIELDS if col in row.index},
            "fundamentals_snapshot": {col: row.get(col) for col in SNAPSHOT_FIELDS if col in row.index},
            "notes": "Volatility is annualised. Returns are decimals. Fundamentals are the latest snapshot and are not model inputs.",
        }

    def rank_stocks(self, n: int = 10, direction: str = "top", sector: str | None = None) -> dict:
        n = _int_arg({"n": n}, "n", 10, 1, MAX_ROWS)
        if direction not in ("top", "bottom"):
            raise ToolInputError("direction must be 'top' or 'bottom'.")
        rows = self._sector_rows(sector).sort_values("up_probability_5d", ascending=direction == "bottom").head(n)
        return {"direction": direction, "sector": sector, "as_of_date": self.scored["date"].max(), "stocks": [self._summary_row(r) for _, r in rows.iterrows()]}

    def screen_stocks(
        self,
        sector: str | None = None,
        min_probability: float | None = None,
        max_volatility: float | None = None,
        min_momentum_60d: float | None = None,
        max_pe_ratio: float | None = None,
        sort_by: str = "up_probability_5d",
        ascending: bool = False,
        limit: int = 15,
    ) -> dict:
        args = locals()
        rows = self._sector_rows(sector)
        for key, column, keep_if in [
            ("min_probability", "up_probability_5d", lambda s, v: s >= v),
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
        symbol = self._row(ticker)["ticker"]
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
        symbol = self._row(ticker)["ticker"]
        news = self.data.news
        if news is None or news.empty:
            return {"ticker": symbol, "headlines": [], "note": "No news file is available. Re-run the pipeline to save headlines."}
        rows = news[news["ticker"] == symbol].sort_values("date", ascending=False).head(limit)
        keep = [c for c in ["date", "title", "sentiment_score", "source"] if c in rows.columns]
        return {"ticker": symbol, "headlines": rows[keep]}

    def sector_summary(self) -> dict:
        if "sector" not in self.scored.columns:
            raise ToolInputError("This dataset has no sector information.")
        grouped = self.scored.groupby("sector")
        table = grouped.agg(
            stocks=("ticker", "count"),
            mean_up_probability=("up_probability_5d", "mean"),
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
