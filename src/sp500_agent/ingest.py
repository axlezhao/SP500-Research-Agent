"""Assemble a dataset from live public sources, the Kaggle files, or the synthetic sample.

Live sources are cached under data/raw/live/ and refreshed when older than their maximum age
(prices and macro: 12 hours; membership and fundamentals: 7 days) or when `refresh=True`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, RAW_DIR
from .data_loader import load_fundamentals, load_news, load_prices

LIVE_DIR = RAW_DIR / "live"
DEFAULT_START = "2014-01-01"
MAX_AGE = {"universe": timedelta(days=7), "prices": timedelta(hours=12), "benchmark": timedelta(hours=12), "fundamentals": timedelta(days=7), "macro": timedelta(hours=12)}
# The S&P 500 itself, for comparison: SPY's dividend-adjusted price (a total-return proxy).
INDEX_SYMBOL = "SPY"
EXTREME_DAILY_MOVE = 0.4
# A former member's price history must cover at least this share of its time in the index; otherwise the
# symbol has most likely been reused by a different company (e.g. STI: SunTrust until 2019, another firm later).
MIN_MEMBERSHIP_COVERAGE = 0.5
# Former members' prices are kept this long after they leave the index (enough to close positions), then dropped.
POST_REMOVAL_DAYS = 15


@dataclass
class Dataset:
    source: str
    prices: pd.DataFrame  # ticker, date, close (adjusted for splits and dividends), volume, [raw_close]
    companies: pd.DataFrame  # ticker, company_name, sector, ...
    news: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["ticker", "date", "title"]))
    fundamentals: pd.DataFrame | None = None  # point-in-time, one row per ticker and availability date
    macro: pd.DataFrame | None = None
    membership: pd.DataFrame | None = None
    index_changes: pd.DataFrame | None = None
    index_prices: pd.Series | None = None  # S&P 500 total-return proxy by date
    quality: dict = field(default_factory=dict)


# -- cache ------------------------------------------------------------------------------------------
def _meta_path(cache_dir: Path) -> Path:
    return cache_dir / "meta.json"


def _read_meta(cache_dir: Path) -> dict:
    path = _meta_path(cache_dir)
    return json.loads(path.read_text()) if path.exists() else {}


def _cached(cache_dir: Path, name: str, params: dict, refresh: bool, fetch, log) -> dict[str, pd.DataFrame]:
    """Return {file stem: frame} for one source, refetching when stale, forced, or fetched with other parameters."""
    meta = _read_meta(cache_dir)
    entry = meta.get(name, {})
    files = entry.get("files", [])
    fresh = (
        not refresh
        and entry.get("params") == params
        and files
        and all((cache_dir / f"{stem}.parquet").exists() for stem in files)
        and datetime.now(timezone.utc) - datetime.fromisoformat(entry["fetched_at"]) < MAX_AGE[name]
    )
    if fresh:
        log(f"  {name}: using cache from {entry['fetched_at'][:16].replace('T', ' ')} UTC")
        return {stem: pd.read_parquet(cache_dir / f"{stem}.parquet") for stem in files}
    log(f"  {name}: downloading...")
    frames, extra = fetch()
    cache_dir.mkdir(parents=True, exist_ok=True)
    for stem, frame in frames.items():
        frame.to_parquet(cache_dir / f"{stem}.parquet", index=False)
    meta = _read_meta(cache_dir)
    meta[name] = {"fetched_at": datetime.now(timezone.utc).isoformat(), "params": params, "files": list(frames), **extra}
    _meta_path(cache_dir).write_text(json.dumps(meta, indent=2, default=str))
    return frames


# -- live -------------------------------------------------------------------------------------------
def load_live(
    start: str = DEFAULT_START,
    limit: int | None = None,
    refresh: bool = False,
    cache_dir: Path = LIVE_DIR,
    client=None,
    sec_client=None,
    price_downloader=None,
    log=print,
) -> Dataset:
    from .sources import fred, sec, wikipedia, yahoo
    from .sources.http import HttpClient

    client = client or HttpClient(max_per_second=4)

    def universe():
        constituents = wikipedia.fetch_constituents(client)
        changes = wikipedia.fetch_changes(client)
        membership, report = wikipedia.build_membership(constituents, changes, start)
        return {"constituents": constituents, "index_changes": changes, "membership": membership}, {"report": report}

    parts = _cached(cache_dir, "universe", {"start": start}, refresh, universe, log)
    constituents, changes, membership = parts["constituents"], parts["index_changes"], parts["membership"]

    current = constituents["ticker"].tolist()
    former = sorted(set(membership["ticker"]) - set(current))
    tickers = current[:limit] if limit else current + former

    def prices():
        frame = yahoo.fetch_prices(tickers, start=start, downloader=price_downloader)
        return {"prices": frame}, {"requested": len(tickers)}

    price_frame = _cached(cache_dir, "prices", {"start": start, "tickers": len(tickers), "limit": limit}, refresh, prices, log)["prices"]
    price_frame, cleaning = clean_former_members(price_frame, membership, set(current), start)

    def benchmark():
        return {"benchmark": yahoo.fetch_prices([INDEX_SYMBOL], start=start, downloader=price_downloader)}, {}

    spy = _cached(cache_dir, "benchmark", {"start": start, "symbol": INDEX_SYMBOL}, refresh, benchmark, log)["benchmark"]
    index_prices = spy.set_index("date")["adj_close"].fillna(spy.set_index("date")["close"]).rename(INDEX_SYMBOL) if not spy.empty else None
    priced = sorted(price_frame["ticker"].unique())

    fundamentals, sec_failures = None, {}
    try:
        sec_http = sec_client or sec.sec_client()
    except RuntimeError as exc:
        log(f"  fundamentals: skipped. {exc}")
        sec_http = None
    if sec_http is not None:
        def fundamentals_fetch():
            cik_map = sec.fetch_cik_map(sec_http)
            wiki_ciks = dict(zip(constituents["ticker"], constituents["cik"]))
            ciks = {t: int(wiki_ciks[t]) if pd.notna(wiki_ciks.get(t)) else cik_map[t] for t in priced if pd.notna(wiki_ciks.get(t)) or t in cik_map}
            table, failed = sec.fetch_fundamentals(sec_http, ciks, on_progress=lambda done, total: log(f"    {done}/{total} companies"))
            cik_frame = pd.DataFrame({"ticker": list(ciks), "cik": list(ciks.values())})
            return {"fundamentals": table, "ciks": cik_frame}, {"failed": failed, "without_cik": sorted(set(priced) - set(ciks))}

        parts = _cached(cache_dir, "fundamentals", {"tickers": len(priced), "limit": limit, "start": start}, refresh, fundamentals_fetch, log)
        fundamentals, ciks = parts["fundamentals"], parts["ciks"]
        sec_failures = _read_meta(cache_dir).get("fundamentals", {}).get("failed", {})
    else:
        ciks = pd.DataFrame({"ticker": constituents["ticker"], "cik": constituents["cik"]}).dropna()

    macro = _cached(cache_dir, "macro", {"start": start}, refresh, lambda: ({"macro": fred.fetch_macro(start=start)}, {}), log)["macro"]

    companies = _companies(constituents, changes, ciks, priced)
    prices_out = price_frame.assign(close=price_frame["adj_close"].fillna(price_frame["close"]))[["ticker", "date", "close", "volume", "raw_close"]]
    dataset = Dataset(
        source="live",
        prices=prices_out,
        companies=companies,
        fundamentals=fundamentals,
        macro=macro,
        membership=membership,
        index_changes=changes,
        index_prices=index_prices,
    )
    meta = _read_meta(cache_dir)
    dataset.quality = data_quality(
        dataset,
        requested=tickers,
        former_members=former if not limit else [],
        membership_report=meta.get("universe", {}).get("report", {}),
        sec_failures=sec_failures,
        fetched_at={name: entry.get("fetched_at") for name, entry in meta.items()},
    )
    if "survivorship" in dataset.quality:
        dataset.quality["survivorship"].update(cleaning)
    return dataset


def clean_former_members(prices: pd.DataFrame, membership: pd.DataFrame, current: set[str], since: str) -> tuple[pd.DataFrame, dict]:
    """Drop former members whose price history doesn't match their index membership, and trim the rest.

    Returns the cleaned prices and a report of what was removed.
    """
    since = pd.Timestamp(since)
    last_date = prices["date"].max()
    dropped, trimmed_rows, keep = [], 0, []
    for ticker, rows in prices.groupby("ticker", sort=False):
        if ticker in current:
            keep.append(rows)
            continue
        intervals = membership[membership["ticker"] == ticker]
        if intervals.empty:
            keep.append(rows)
            continue
        member_days = 0
        covered = 0
        for interval in intervals.itertuples():
            begin = since if pd.isna(interval.start) else max(interval.start, since)
            end = last_date if pd.isna(interval.end) else interval.end
            days = pd.bdate_range(begin, end - pd.Timedelta(days=1))
            member_days += len(days)
            covered += int(rows["date"].between(begin, end - pd.Timedelta(days=1)).sum())
        if member_days and covered / member_days < MIN_MEMBERSHIP_COVERAGE:
            dropped.append(ticker)
            continue
        cutoff = intervals["end"].max() + pd.Timedelta(days=POST_REMOVAL_DAYS) if intervals["end"].notna().all() else last_date
        trimmed_rows += int((rows["date"] > cutoff).sum())
        keep.append(rows[rows["date"] <= cutoff])
    cleaned = pd.concat(keep, ignore_index=True) if keep else prices.iloc[0:0]
    return cleaned, {
        "dropped_no_matching_history": len(dropped),
        "dropped_examples": sorted(dropped)[:15],
        "rows_trimmed_after_removal": trimmed_rows,
    }


def _companies(constituents: pd.DataFrame, changes: pd.DataFrame, ciks: pd.DataFrame, priced: list[str]) -> pd.DataFrame:
    current = constituents[["ticker", "company_name", "sector", "sub_industry"]]
    removed = (
        changes.dropna(subset=["removed"])
        .drop_duplicates("removed")
        .rename(columns={"removed": "ticker", "removed_name": "company_name"})[["ticker", "company_name"]]
    )
    removed = removed[~removed["ticker"].isin(current["ticker"])].assign(sector="Unknown", sub_industry=None)
    companies = pd.concat([current, removed], ignore_index=True)
    companies = companies[companies["ticker"].isin(priced)]
    return companies.merge(ciks, on="ticker", how="left").reset_index(drop=True)


# -- kaggle / sample --------------------------------------------------------------------------------
def load_files(source: str = "kaggle", raw_dir: Path = RAW_DIR) -> Dataset:
    prices = load_prices(raw_dir)
    dataset = Dataset(source=source, prices=prices, companies=load_fundamentals(raw_dir), news=load_news(raw_dir))
    dataset.quality = data_quality(dataset)
    return dataset


# -- quality ----------------------------------------------------------------------------------------
def price_checks(prices: pd.DataFrame) -> dict:
    prices = prices.sort_values(["ticker", "date"])
    returns = prices.groupby("ticker")["close"].pct_change()
    extreme = prices.loc[returns.abs() > EXTREME_DAILY_MOVE, ["ticker", "date"]].assign(move=returns[returns.abs() > EXTREME_DAILY_MOVE])
    last = prices.groupby("ticker")["date"].max()
    gaps = prices.groupby("ticker")["date"].diff().dt.days
    years = prices.groupby("ticker")["date"].agg(lambda d: (d.max() - d.min()).days / 365.25)
    return {
        "rows": int(len(prices)),
        "tickers": int(prices["ticker"].nunique()),
        "start": str(prices["date"].min().date()),
        "end": str(prices["date"].max().date()),
        "duplicate_rows": int(prices.duplicated(["ticker", "date"]).sum()),
        "extreme_daily_moves": int(len(extreme)),
        "extreme_examples": [
            {"ticker": r.ticker, "date": str(r.date.date()), "move": round(float(r.move), 3)}
            for r in extreme.reindex(extreme["move"].abs().sort_values(ascending=False).index).head(5).itertuples()
        ],
        "tickers_ending_early": int((last < prices["date"].max() - pd.Timedelta(days=10)).sum()),
        "longest_gap_days": int(gaps.max()) if gaps.notna().any() else 0,
        "median_history_years": round(float(years.median()), 1),
    }


def data_quality(dataset: Dataset, requested=None, former_members=None, membership_report=None, sec_failures=None, fetched_at=None) -> dict:
    report: dict = {"source": dataset.source, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "prices": price_checks(dataset.prices)}
    priced = set(dataset.prices["ticker"])
    if requested is not None:
        missing = sorted(set(requested) - priced)
        report["prices"]["requested"] = len(requested)
        report["prices"]["missing"] = len(missing)
        report["prices"]["missing_examples"] = missing[:15]
    if former_members:
        covered = len(set(former_members) & priced)
        report["survivorship"] = {
            "former_members": len(former_members),
            "former_members_with_prices": covered,
            "coverage": round(covered / len(former_members), 3),
            "note": "Stocks that left the index and have no price data (often delisted after mergers or failures) cannot be backtested, so results still lean optimistic.",
        }
    if membership_report:
        report["membership"] = membership_report
    if dataset.fundamentals is not None:
        latest = dataset.fundamentals.sort_values("available_date").groupby("ticker").tail(1)
        age = (pd.Timestamp(dataset.prices["date"].max()) - latest["available_date"]).dt.days
        report["fundamentals"] = {
            "tickers_with_data": int(dataset.fundamentals["ticker"].nunique()),
            "share_of_priced_tickers": round(dataset.fundamentals["ticker"].nunique() / max(len(priced), 1), 3),
            "median_days_since_last_filing": int(age.median()) if len(age) else None,
            "failed": len(sec_failures or {}),
            "failed_examples": dict(list((sec_failures or {}).items())[:5]),
        }
    if dataset.macro is not None and not dataset.macro.empty:
        report["macro"] = {col: str(dataset.macro.loc[dataset.macro[col].notna(), "date"].max().date()) for col in dataset.macro.columns if col != "date"}
    if fetched_at:
        report["fetched_at"] = fetched_at
    return report


def save_dataset_extras(dataset: Dataset, directory: Path = PROCESSED_DIR) -> None:
    """Persist what the app and agent need besides the feature table."""
    directory.mkdir(parents=True, exist_ok=True)
    if dataset.index_prices is not None:
        dataset.index_prices.rename("close").rename_axis("date").reset_index().to_parquet(directory / "benchmark.parquet", index=False)
    elif (directory / "benchmark.parquet").exists():
        (directory / "benchmark.parquet").unlink()
    for name in ["fundamentals", "macro", "membership", "index_changes"]:
        frame = getattr(dataset, name)
        path = directory / f"{name}.parquet"
        if frame is not None:
            frame.to_parquet(path, index=False)
        elif path.exists():
            path.unlink()  # don't leave a previous source's file behind
    dataset.companies.to_parquet(directory / "companies.parquet", index=False)
    (directory / "data_quality.json").write_text(json.dumps(dataset.quality, indent=2, default=_json_default))


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return str(value)
