"""Assemble a dataset from live public sources, the Kaggle files, or the synthetic sample.

Live sources are cached under data/raw/live/ and refreshed when older than their maximum age
(prices and macro: 12 hours; membership, fundamentals, filings and factors: 7 days; insider trades:
30 days) or when `refresh=True`.
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
MAX_AGE = {
    "universe": timedelta(days=7), "prices": timedelta(hours=12), "benchmark": timedelta(hours=12), "fundamentals": timedelta(days=7),
    "events": timedelta(days=7), "macro": timedelta(hours=12), "french": timedelta(days=7), "insiders": timedelta(days=30),
}
# Bump when the fundamentals table gains columns, so caches built by older code are refetched.
FUNDAMENTALS_VERSION = 2
# The S&P 500 itself, for comparison: SPY's dividend-adjusted price (a total-return proxy).
INDEX_SYMBOL = "SPY"
EXTREME_DAILY_MOVE = 0.4
# A former member's price history must cover at least this share of its time in the index; otherwise the
# symbol has most likely been reused by a different company (e.g. STI: SunTrust until 2019, another firm later).
MIN_MEMBERSHIP_COVERAGE = 0.5
# Former members' prices are kept this long after they leave the index, then dropped. It must cover the
# longest holding period (21 sessions plus the execution lag, about 31 calendar days with holidays), or a
# position opened just before the removal would lose its exit price and look like a delisting.
POST_REMOVAL_DAYS = 45
# A stock whose prices stop while it is still trading in the backtest is either acquired (its last price is
# about the deal price: delisting return 0) or delisted for poor performance, where the last quoted price
# overstates what holders got. Shumway (1997) estimates about -30% for the latter.
PERFORMANCE_DELISTING_RETURN = -0.30
PERFORMANCE_WORDS = ("bankrupt", "chapter 11", "delist", "default", "insolv", "liquidat")


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
    eps: pd.DataFrame | None = None  # quarterly EPS as first reported (ticker, period_end, eps, filed)
    earnings_dates: pd.DataFrame | None = None  # earnings releases, 8-K item 2.02 (ticker, date, accepted)
    insiders: pd.DataFrame | None = None  # insider open-market purchases and sales (ticker, filed, accession, code, shares, value)
    factors: pd.DataFrame | None = None  # Kenneth French daily factor returns
    delistings: pd.DataFrame | None = None  # stocks whose prices end early (ticker, last_date, reason, delisting_return)
    quality: dict = field(default_factory=dict)

    @property
    def delisting_returns(self) -> dict:
        if self.delistings is None or self.delistings.empty:
            return {}
        return dict(zip(self.delistings["ticker"], self.delistings["delisting_return"]))


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
    insiders: bool = False,
    factors: bool = True,
) -> Dataset:
    from .sources import fred, french, sec, wikipedia, yahoo
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

    fundamentals, eps, sec_failures = None, None, {}
    industries, earnings_dates, insider_trades = None, None, None
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
            table, eps_table, failed = sec.fetch_company_facts(sec_http, ciks, on_progress=lambda done, total: log(f"    {done}/{total} companies"))
            cik_frame = pd.DataFrame({"ticker": list(ciks), "cik": list(ciks.values())})
            return {"fundamentals": table, "eps": eps_table, "ciks": cik_frame}, {"failed": failed, "without_cik": sorted(set(priced) - set(ciks))}

        params = {"tickers": len(priced), "limit": limit, "start": start, "version": FUNDAMENTALS_VERSION}
        parts = _cached(cache_dir, "fundamentals", params, refresh, fundamentals_fetch, log)
        fundamentals, eps, ciks = parts["fundamentals"], parts["eps"], parts["ciks"]
        sec_failures = _read_meta(cache_dir).get("fundamentals", {}).get("failed", {})
        cik_dict = {t: int(c) for t, c in zip(ciks["ticker"], ciks["cik"])}

        def events_fetch():
            table, releases, failed = sec.fetch_company_events(sec_http, cik_dict, since=start, on_progress=lambda done, total: log(f"    {done}/{total} filing indexes"))
            return {"industries": table, "earnings_dates": releases}, {"failed": failed}

        try:
            events = _cached(cache_dir, "events", {"tickers": len(cik_dict), "start": start}, refresh, events_fetch, log)
            industries, earnings_dates = events["industries"], events["earnings_dates"]
        except Exception as exc:  # industry codes and release dates improve the research but aren't essential
            log(f"  events: skipped ({type(exc).__name__}: {exc})")
        if insiders:
            from .sources import insiders as insider_source

            def insiders_fetch():
                trades, failed = insider_source.fetch_insider_trades(sec_http, cik_dict, start, on_progress=lambda q: log(f"    insider trades {q}"))
                return {"insiders": trades}, {"failed": failed}

            insider_trades = _cached(cache_dir, "insiders", {"tickers": len(cik_dict), "start": start}, refresh, insiders_fetch, log)["insiders"]
    else:
        ciks = pd.DataFrame({"ticker": constituents["ticker"], "cik": constituents["cik"]}).dropna()

    macro = _cached(cache_dir, "macro", {"start": start}, refresh, lambda: ({"macro": fred.fetch_macro(start=start)}, {}), log)["macro"]
    factor_frame = None
    if factors:
        try:
            factor_frame = _cached(cache_dir, "french", {}, refresh, lambda: ({"french": french.fetch_factors(client)}, {}), log)["french"]
        except Exception as exc:
            log(f"  french: skipped ({type(exc).__name__}: {exc})")

    companies = _companies(constituents, changes, ciks, priced, industries)
    prices_out = live_price_table(price_frame)
    dataset = Dataset(
        source="live",
        prices=prices_out,
        companies=companies,
        fundamentals=fundamentals,
        macro=macro,
        membership=membership,
        index_changes=changes,
        index_prices=index_prices,
        eps=eps,
        earnings_dates=earnings_dates,
        insiders=insider_trades,
        factors=factor_frame,
        delistings=delistings(price_frame, changes, membership),
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


def live_price_table(price_frame: pd.DataFrame) -> pd.DataFrame:
    """Prices for the features: dividend-adjusted close (total return), as-traded close, volume, the split factor,
    and highs and lows rescaled to the dividend-adjusted basis (for spread estimates)."""
    adjusted = price_frame["adj_close"].fillna(price_frame["close"])
    factor = adjusted / price_frame["close"]
    out = price_frame.assign(close=adjusted)
    columns = ["ticker", "date", "close", "volume", "raw_close"]
    if {"high", "low"} <= set(price_frame.columns):
        out = out.assign(high=price_frame["high"] * factor, low=price_frame["low"] * factor)
        columns += ["high", "low"]
    if "split_factor" in price_frame.columns:
        columns.append("split_factor")
    return out[columns]


def delistings(prices: pd.DataFrame, changes: pd.DataFrame | None, membership: pd.DataFrame | None) -> pd.DataFrame:
    """Stocks whose price history stops before the data ends, other than because the index dropped them.

    Each gets a delisting return for the features: PERFORMANCE_DELISTING_RETURN when Wikipedia's removal
    reason mentions bankruptcy or delisting, or the price fell by more than half over its last quarter; 0
    (a merger at roughly the last price) otherwise.
    """
    columns = ["ticker", "last_date", "reason", "delisting_return"]
    if prices.empty:
        return pd.DataFrame(columns=columns)
    data_end = prices["date"].max()
    last = prices.sort_values("date").groupby("ticker").agg(last_date=("date", "max"), last_close=("close", "last"))
    quarter_ago = prices.sort_values("date").groupby("ticker")["close"].apply(lambda c: c.iloc[-63] if len(c) >= 63 else np.nan)
    last["quarter_return"] = last["last_close"] / quarter_ago - 1
    ended = last[last["last_date"] < data_end - pd.Timedelta(days=10)].copy()
    if membership is not None and not membership.empty:
        # Prices trimmed POST_REMOVAL_DAYS after an index removal didn't end: the stock kept trading.
        removal = membership.groupby("ticker")["end"].max()
        trimmed = removal.reindex(ended.index).notna() & (ended["last_date"] >= removal.reindex(ended.index) + pd.Timedelta(days=POST_REMOVAL_DAYS - 7))
        ended = ended[~trimmed]
    reasons = {}
    if changes is not None and "reason" in changes.columns:
        removed = changes.dropna(subset=["removed"]).drop_duplicates("removed", keep="first")
        reasons = dict(zip(removed["removed"], removed["reason"].astype(str)))
    ended["reason"] = [reasons.get(t, "") for t in ended.index]
    poor = ended["reason"].astype(str).str.lower().str.contains("|".join(PERFORMANCE_WORDS)) | (ended["quarter_return"] < -0.5)
    ended["delisting_return"] = np.where(poor, PERFORMANCE_DELISTING_RETURN, 0.0)
    return ended.reset_index()[columns]


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


def _companies(constituents: pd.DataFrame, changes: pd.DataFrame, ciks: pd.DataFrame, priced: list[str], industries: pd.DataFrame | None = None) -> pd.DataFrame:
    """Company names, sectors and industries for every priced ticker.

    Current members carry their GICS sector from Wikipedia. Former members have none on record; giving them
    a placeholder such as "Unknown" would hand the model a flag meaning "this stock will leave the index",
    which is information from the future. Their sector comes from their SEC industry (SIC) code instead,
    and stays missing when there is none. `industry` (the SIC major group) is defined the same way for all.
    """
    from .sources.sec import sic_industry, sic_sector

    current = constituents[["ticker", "company_name", "sector", "sub_industry"]]
    removed = (
        changes.dropna(subset=["removed"])
        .drop_duplicates("removed")
        .rename(columns={"removed": "ticker", "removed_name": "company_name"})[["ticker", "company_name"]]
    )
    removed = removed[~removed["ticker"].isin(current["ticker"])].assign(sector=None, sub_industry=None)
    companies = pd.concat([current, removed], ignore_index=True)
    companies = companies[companies["ticker"].isin(priced)]
    companies = companies.merge(ciks, on="ticker", how="left")
    if industries is not None and not industries.empty:
        companies = companies.merge(industries[["ticker", "sic", "sic_description"]], on="ticker", how="left")
        missing = companies["sector"].isna()
        companies.loc[missing, "sector"] = companies.loc[missing, "sic"].map(sic_sector)
        companies["industry"] = companies["sic"].map(sic_industry)
    return companies.reset_index(drop=True)


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
    if dataset.companies is not None and "sic" in dataset.companies.columns:
        report["industries"] = {
            "share_with_sic_code": round(float(dataset.companies["sic"].notna().mean()), 3),
            "former_members_with_sector": int(dataset.companies.loc[dataset.companies["sub_industry"].isna(), "sector"].notna().sum()),
        }
    if dataset.earnings_dates is not None:
        report["earnings_releases"] = {"releases": int(len(dataset.earnings_dates)), "tickers": int(dataset.earnings_dates["ticker"].nunique())}
    if dataset.delistings is not None:
        report["delistings"] = {
            "stocks_ending_early": int(len(dataset.delistings)),
            "assumed_performance_delistings": int((dataset.delistings["delisting_return"] < 0).sum()),
            "note": "Kept in the backtest to their last price (mergers) or with a -30% delisting return (failures), rather than dropped.",
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
    for name in ["fundamentals", "macro", "membership", "index_changes", "eps", "earnings_dates", "insiders", "factors", "delistings"]:
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
