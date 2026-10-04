"""Point-in-time fundamentals and filings from SEC EDGAR (free, no key; needs a contact User-Agent).

The XBRL "company facts" API lists every number a company has reported, with the date it was filed.
From it this module builds, per company, a timeline of what was publicly known on each date:
trailing-twelve-month revenue and net income, year-on-year revenue growth, equity, liabilities and
shares outstanding. Each period keeps the value from its first filing, so later restatements never
leak backwards, and a value only becomes usable after its filing date.

SEC asks for a User-Agent naming the requester with a contact email (set SEC_USER_AGENT) and at most
10 requests per second.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .http import HttpClient

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

# Candidate XBRL concepts per metric, most preferred first. Companies switch concepts over the years
# (e.g. Revenues -> RevenueFromContractWithCustomer... after ASC 606), so periods are merged across them.
FLOW_CONCEPTS = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "RevenuesNetOfInterestExpense",
    ],
    "net_income": ["NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic", "ProfitLoss"],
}
INSTANT_CONCEPTS = {
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "liabilities": ["Liabilities"],
    "assets": ["Assets"],
}
SHARE_CONCEPTS = [("dei", "EntityCommonStockSharesOutstanding"), ("us-gaap", "CommonStockSharesOutstanding")]
QUARTER_DAYS = (80, 120)  # includes 16-week first quarters of 52/53-week calendars (e.g. Kroger)
ANNUAL_DAYS = (350, 380)
PIT_COLUMNS = ["available_date", "period_end", "revenue_ttm", "net_income_ttm", "revenue_growth_yoy", "equity", "liabilities", "shares_outstanding"]


def sec_client(user_agent: str | None = None) -> HttpClient:
    user_agent = user_agent or os.environ.get("SEC_USER_AGENT")
    if not user_agent or "@" not in user_agent:
        raise RuntimeError(
            "SEC EDGAR needs a User-Agent with a contact email. Add a line like\n"
            "  SEC_USER_AGENT=YourName you@example.com\nto .env (see .env.example)."
        )
    return HttpClient(user_agent=user_agent, max_per_second=8)


def fetch_cik_map(client: HttpClient) -> dict[str, int]:
    data = client.get_json(TICKERS_URL)
    return {row["ticker"].upper().replace("-", "."): int(row["cik_str"]) for row in data.values()}


def _facts_frame(facts: dict, taxonomy: str, concepts: list[str]) -> pd.DataFrame:
    """Rows from the first concept that reports each period; later concepts only fill gaps."""
    frames = []
    for priority, concept in enumerate(concepts):
        units = facts.get("facts", {}).get(taxonomy, {}).get(concept, {}).get("units", {})
        for unit, rows in units.items():
            if unit in ("USD", "shares"):
                frames.append(pd.DataFrame(rows).assign(priority=priority))
    if not frames:
        return pd.DataFrame(columns=["start", "end", "val", "filed", "form", "accn", "priority"])
    df = pd.concat(frames, ignore_index=True)
    df = df[df["form"].astype(str).str.match(r"^(10-K|10-Q|20-F|40-F)")]
    if "start" not in df:
        df["start"] = None
    for col in ["start", "end", "filed"]:
        df[col] = _ns(df[col])
    return df.dropna(subset=["end", "filed", "val"])


def _ns(values) -> pd.Series:
    """Dates at nanosecond resolution: pandas 3 parses some inputs as microseconds, and merges need one unit."""
    return pd.to_datetime(values, errors="coerce").astype("datetime64[ns]")


def first_reported(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """For each period, keep the value as first filed (the point-in-time value), preferring better concepts on ties."""
    return df.sort_values(["filed", "priority"]).drop_duplicates(keys, keep="first")


def ttm_from_facts(df: pd.DataFrame) -> pd.DataFrame:
    """Trailing-twelve-month sums with the date they became public.

    Quarterly facts are chained four at a time; a missing fourth quarter is derived as the annual value
    minus the first three quarters. Companies that only report annually fall back to the annual value.
    Returns columns: end, ttm, available.
    """
    if df.empty:
        return pd.DataFrame(columns=["end", "ttm", "available"])
    df = first_reported(df.dropna(subset=["start"]), ["start", "end"])
    days = (df["end"] - df["start"]).dt.days
    quarters = df[days.between(*QUARTER_DAYS)][["start", "end", "val", "filed"]].sort_values("end")
    annual = df[days.between(*ANNUAL_DAYS)][["start", "end", "val", "filed"]]

    derived = []
    for row in annual.itertuples(index=False):
        inside = quarters[(quarters["start"] >= row.start - pd.Timedelta(days=7)) & (quarters["end"] <= row.end + pd.Timedelta(days=7))]
        if len(inside) == 3 and not (abs(inside["end"] - row.end) <= pd.Timedelta(days=7)).any():
            derived.append({"start": inside["end"].max() + pd.Timedelta(days=1), "end": row.end, "val": row.val - inside["val"].sum(), "filed": row.filed})
    if derived:
        quarters = pd.concat([quarters, pd.DataFrame(derived)], ignore_index=True)
    quarters = quarters.sort_values("end").drop_duplicates("end").reset_index(drop=True)

    out = []
    for i in range(3, len(quarters)):
        window = quarters.iloc[i - 3 : i + 1]
        gaps = (window["start"].iloc[1:].values - window["end"].iloc[:-1].values) / np.timedelta64(1, "D")
        if np.all(np.abs(gaps - 1) <= 10):  # four back-to-back quarters
            out.append({"end": window["end"].iloc[-1], "ttm": window["val"].sum(), "available": window["filed"].max()})
    ttm = pd.DataFrame(out, columns=["end", "ttm", "available"]).astype({"end": "datetime64[ns]", "available": "datetime64[ns]"})
    fallback = annual.rename(columns={"val": "ttm", "filed": "available"})[["end", "ttm", "available"]]
    covered = fallback["end"].map(lambda e: bool(((ttm["end"] - e).abs() <= pd.Timedelta(days=7)).any())) if len(ttm) else False
    fallback = fallback[~covered] if len(ttm) else fallback
    return pd.concat([ttm, fallback], ignore_index=True).sort_values("end").reset_index(drop=True)


def _instant(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["end", "value", "available"])
    df = first_reported(df, ["end"])
    return df.rename(columns={"val": "value", "filed": "available"})[["end", "value", "available"]].sort_values("end")


def _shares(facts: dict) -> pd.DataFrame:
    frames = []
    for taxonomy, concept in SHARE_CONCEPTS:
        df = _facts_frame(facts, taxonomy, [concept])
        if not df.empty:
            # Multi-class companies report one cover-page figure per class: add them up within a filing.
            df = df.groupby(["accn", "end", "form"], as_index=False).agg(val=("val", "sum"), filed=("filed", "min"))
            frames.append(df.assign(priority=len(frames), start=pd.NaT))
    return _instant(pd.concat(frames, ignore_index=True)) if frames else _instant(pd.DataFrame())


def _known_as_of(records: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """At each availability date, the most recent period known so far (ignores late filings of old periods)."""
    records = records.dropna(subset=[value_col]).sort_values(["available", "end"])
    keep = records["end"] > records["end"].cummax().shift(1).fillna(pd.Timestamp.min)
    return records[keep]


def point_in_time(facts: dict) -> pd.DataFrame:
    """One row per date on which the company's publicly known fundamentals changed."""
    revenue = ttm_from_facts(_facts_frame(facts, "us-gaap", FLOW_CONCEPTS["revenue"]))
    income = ttm_from_facts(_facts_frame(facts, "us-gaap", FLOW_CONCEPTS["net_income"]))
    if not revenue.empty:
        previous = revenue[["end", "ttm", "available"]].rename(columns={"end": "prev_end", "ttm": "prev_ttm", "available": "prev_available"})
        matched = pd.merge_asof(
            revenue.sort_values("end").assign(target=lambda r: r["end"] - pd.Timedelta(days=365)).sort_values("target"),
            previous.sort_values("prev_end"),
            left_on="target", right_on="prev_end", direction="nearest", tolerance=pd.Timedelta(days=20),
        )
        revenue = matched.assign(growth=lambda m: m["ttm"] / m["prev_ttm"] - 1)
    series = {
        "revenue_ttm": revenue.rename(columns={"ttm": "value"}),
        "net_income_ttm": income.rename(columns={"ttm": "value"}),
        "revenue_growth_yoy": revenue[["end", "growth", "available"]].rename(columns={"growth": "value"}) if "growth" in revenue else pd.DataFrame(),
        "equity": _instant(_facts_frame(facts, "us-gaap", INSTANT_CONCEPTS["equity"])),
        "liabilities": _instant(_facts_frame(facts, "us-gaap", INSTANT_CONCEPTS["liabilities"])),
        "shares_outstanding": _shares(facts),
    }
    events = sorted({d for s in series.values() if not s.empty for d in s["available"].dropna()})
    if not events:
        return pd.DataFrame(columns=PIT_COLUMNS)
    timeline = pd.DataFrame({"available_date": _ns(pd.Series(events))})
    period_end = pd.Series(pd.NaT, index=timeline.index, dtype="datetime64[ns]")
    for name, records in series.items():
        if records.empty:
            timeline[name] = np.nan
            continue
        known = _known_as_of(records[["end", "value", "available"]].astype({"available": "datetime64[ns]", "end": "datetime64[ns]"}), "value")
        merged = pd.merge_asof(timeline, known.rename(columns={"available": "available_date"}).sort_values("available_date"), on="available_date", direction="backward")
        timeline[name] = merged["value"].astype(float)
        if name in ("revenue_ttm", "net_income_ttm"):
            period_end = period_end.combine(merged["end"], lambda a, b: b if pd.isna(a) or (pd.notna(b) and b > a) else a)
    timeline["period_end"] = period_end
    # Some filers (many banks and insurers) never tag total liabilities: back them out of total assets.
    if timeline["liabilities"].isna().all():
        assets_rows = _instant(_facts_frame(facts, "us-gaap", INSTANT_CONCEPTS["assets"]))
        if not assets_rows.empty:
            known = _known_as_of(assets_rows, "value").rename(columns={"available": "available_date"})
            merged = pd.merge_asof(timeline[["available_date"]], known.sort_values("available_date"), on="available_date", direction="backward")
            timeline["liabilities"] = merged["value"].astype(float) - timeline["equity"]
    return timeline[PIT_COLUMNS]


def fetch_fundamentals(client: HttpClient, ciks: dict[str, int], workers: int = 4, on_progress=None) -> tuple[pd.DataFrame, dict[str, str]]:
    """Point-in-time fundamentals for many tickers. Returns (table with a ticker column, {ticker: why it failed})."""
    def one(item):
        ticker, cik = item
        try:
            return ticker, point_in_time(client.get_json(FACTS_URL.format(cik=cik))), None
        except Exception as exc:  # missing filer, network error, unexpected layout: skip this company, keep the reason
            return ticker, None, f"{type(exc).__name__}: {str(exc)[:120]}"

    frames, failed = [], {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (ticker, frame, error) in enumerate(pool.map(one, ciks.items()), start=1):
            if frame is None or frame.empty:
                failed[ticker] = error or "no usable facts"
            else:
                frames.append(frame.assign(ticker=ticker))
            if on_progress and done % 50 == 0:
                on_progress(done, len(ciks))
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["ticker", *PIT_COLUMNS])
    return table, failed


def fetch_filings(client: HttpClient, cik: int, ticker: str, forms: tuple[str, ...] = ("10-K", "10-Q", "8-K"), limit: int = 15) -> pd.DataFrame:
    """Most recent filings of the given types, with links to the documents."""
    recent = client.get_json(SUBMISSIONS_URL.format(cik=cik)).get("filings", {}).get("recent", {})
    df = pd.DataFrame(recent)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "form", "filed", "report_date", "description", "url"])
    df = df[df["form"].isin(forms)].head(limit)
    accession = df["accessionNumber"].str.replace("-", "", regex=False)
    return pd.DataFrame(
        {
            "ticker": ticker,
            "form": df["form"],
            "filed": pd.to_datetime(df["filingDate"]),
            "report_date": pd.to_datetime(df.get("reportDate"), errors="coerce"),
            "description": df.get("primaryDocDescription", pd.Series(index=df.index, dtype=object)).fillna("").replace("", None),
            "items": df.get("items", pd.Series(index=df.index, dtype=object)).fillna("").replace("", None),
            "url": [f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}" for acc, doc in zip(accession, df["primaryDocument"])],
        }
    ).reset_index(drop=True)


class FilingsFetcher:
    """Recent SEC filings per ticker, fetched on demand and cached for the session."""

    def __init__(self, client: HttpClient, ciks: dict[str, int]):
        self.client = client
        self.ciks = ciks
        self._cache: dict[str, pd.DataFrame] = {}

    @classmethod
    def from_env(cls, ciks: dict[str, int]) -> "FilingsFetcher | None":
        try:
            return cls(sec_client(), ciks)
        except RuntimeError:
            return None

    def fetch(self, ticker: str, forms: tuple[str, ...] = ("10-K", "10-Q", "8-K"), limit: int = 40) -> pd.DataFrame:
        if ticker not in self.ciks:
            raise KeyError(ticker)
        if ticker not in self._cache:
            self._cache[ticker] = fetch_filings(self.client, self.ciks[ticker], ticker, forms=("10-K", "10-Q", "8-K", "DEF 14A", "S-1", "SC 13D"), limit=limit)
        filings = self._cache[ticker]
        return filings[filings["form"].isin(forms)]
