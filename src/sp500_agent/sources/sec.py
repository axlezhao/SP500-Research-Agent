"""Point-in-time fundamentals and filings from SEC EDGAR (free, no key; needs a contact User-Agent).

The XBRL "company facts" API lists every number a company has reported, with the date it was filed.
From it this module builds, per company, a timeline of what was publicly known on each date:
trailing-twelve-month revenue and net income, year-on-year revenue growth, equity, liabilities and
shares outstanding. Each period keeps the value from its first filing, so later restatements never
leak backwards, and a value only becomes usable after its filing date.

The submissions API adds each company's SIC industry code (also for companies that have since been
delisted, so former index members get a sector without hindsight) and the dates of its earnings
releases (8-K item 2.02), which time the earnings-surprise signal.

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
    "gross_profit": ["GrossProfit"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
}
EPS_CONCEPTS = ["EarningsPerShareDiluted", "EarningsPerShareBasic", "EarningsPerShareBasicAndDiluted"]
INSTANT_CONCEPTS = {
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "liabilities": ["Liabilities"],
    "assets": ["Assets"],
}
SHARE_CONCEPTS = [("dei", "EntityCommonStockSharesOutstanding"), ("us-gaap", "CommonStockSharesOutstanding")]
QUARTER_DAYS = (80, 120)  # includes 16-week first quarters of 52/53-week calendars (e.g. Kroger)
HALF_DAYS = (170, 200)
NINE_MONTH_DAYS = (260, 290)
ANNUAL_DAYS = (350, 380)
PIT_COLUMNS = [
    "available_date", "period_end", "revenue_ttm", "net_income_ttm", "revenue_growth_yoy", "equity", "liabilities", "shares_outstanding",
    "gross_profit_ttm", "operating_cash_flow_ttm", "assets",
]
EPS_COLUMNS = ["period_end", "eps", "filed"]


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
            if unit in ("USD", "shares", "USD/shares"):
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
    quarters, annual = quarterly_values(df)
    return _chain_ttm(quarters, annual)


def quarterly_values(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(single quarters, annual values) from duration facts, as first reported.

    Many cash-flow items (and some income items) are only reported year-to-date in 10-Qs: 3, 6 and 9
    months, then 12 in the 10-K. Each quarter is recovered as the difference between consecutive
    year-to-date values with the same start; the fourth quarter as the annual value minus the first three.
    """
    if df.empty:
        empty = pd.DataFrame(columns=["start", "end", "val", "filed"])
        return empty, empty
    df = first_reported(df.dropna(subset=["start"]), ["start", "end"])
    days = (df["end"] - df["start"]).dt.days
    quarters = df[days.between(*QUARTER_DAYS)][["start", "end", "val", "filed"]].sort_values("end")
    annual = df[days.between(*ANNUAL_DAYS)][["start", "end", "val", "filed"]]
    ytd = df[days.between(*HALF_DAYS) | days.between(*NINE_MONTH_DAYS)][["start", "end", "val", "filed"]]
    if not ytd.empty:
        shorter = pd.concat([quarters, ytd], ignore_index=True)
        derived = []
        for row in ytd.itertuples(index=False):
            # The year-to-date value one quarter shorter, with the same fiscal-year start.
            prior = shorter[(abs(shorter["start"] - row.start) <= pd.Timedelta(days=3)) & (shorter["end"] < row.end - pd.Timedelta(days=60)) & (shorter["end"] > row.end - pd.Timedelta(days=120))]
            if len(prior) == 1:
                p = prior.iloc[0]
                derived.append({"start": p["end"] + pd.Timedelta(days=1), "end": row.end, "val": row.val - p["val"], "filed": max(row.filed, p["filed"])})
        if derived:
            derived = pd.DataFrame(derived)
            derived = derived[~derived["end"].map(lambda e: bool((abs(quarters["end"] - e) <= pd.Timedelta(days=7)).any()))]
            quarters = pd.concat([quarters, derived], ignore_index=True).sort_values("end")
    derived = []
    for row in annual.itertuples(index=False):
        inside = quarters[(quarters["start"] >= row.start - pd.Timedelta(days=7)) & (quarters["end"] <= row.end + pd.Timedelta(days=7))]
        if len(inside) == 3 and not (abs(inside["end"] - row.end) <= pd.Timedelta(days=7)).any():
            derived.append({"start": inside["end"].max() + pd.Timedelta(days=1), "end": row.end, "val": row.val - inside["val"].sum(), "filed": row.filed})
    if derived:
        quarters = pd.concat([quarters, pd.DataFrame(derived)], ignore_index=True)
    quarters = quarters.sort_values("end").drop_duplicates("end").reset_index(drop=True)
    return quarters, annual.reset_index(drop=True)


def _chain_ttm(quarters: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    if quarters.empty and annual.empty:
        return pd.DataFrame(columns=["end", "ttm", "available"])
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
    gross = ttm_from_facts(_facts_frame(facts, "us-gaap", FLOW_CONCEPTS["gross_profit"]))
    if gross.empty and not revenue.empty:
        # Many filers tag cost of revenue but not gross profit: revenue minus cost for the same period.
        cost = ttm_from_facts(_facts_frame(facts, "us-gaap", FLOW_CONCEPTS["cost_of_revenue"]))
        if not cost.empty:
            both = revenue[["end", "ttm", "available"]].merge(cost, on="end", suffixes=("", "_cost"))
            gross = both.assign(ttm=both["ttm"] - both["ttm_cost"], available=both[["available", "available_cost"]].max(axis=1))[["end", "ttm", "available"]]
    series = {
        "revenue_ttm": revenue.rename(columns={"ttm": "value"}),
        "net_income_ttm": income.rename(columns={"ttm": "value"}),
        "revenue_growth_yoy": revenue[["end", "growth", "available"]].rename(columns={"growth": "value"}) if "growth" in revenue else pd.DataFrame(),
        "equity": _instant(_facts_frame(facts, "us-gaap", INSTANT_CONCEPTS["equity"])),
        "liabilities": _instant(_facts_frame(facts, "us-gaap", INSTANT_CONCEPTS["liabilities"])),
        "shares_outstanding": _shares(facts),
        "gross_profit_ttm": gross.rename(columns={"ttm": "value"}),
        "operating_cash_flow_ttm": ttm_from_facts(_facts_frame(facts, "us-gaap", FLOW_CONCEPTS["operating_cash_flow"])).rename(columns={"ttm": "value"}),
        "assets": _instant(_facts_frame(facts, "us-gaap", INSTANT_CONCEPTS["assets"])),
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


def quarterly_eps(facts: dict) -> pd.DataFrame:
    """Diluted EPS per fiscal quarter as first reported, with the filing date of the 10-Q or 10-K.

    The fourth quarter is the annual figure minus the first three (an approximation for EPS, since the
    share count moves during the year, and the standard one). Returns columns: period_end, eps, filed.
    """
    for concept in EPS_CONCEPTS:  # one concept throughout: basic and diluted EPS don't mix
        frame = _facts_frame(facts, "us-gaap", [concept])
        quarters, _ = quarterly_values(frame)
        if len(quarters) >= 4:
            return quarters.rename(columns={"end": "period_end", "val": "eps"})[EPS_COLUMNS].reset_index(drop=True)
    return pd.DataFrame(columns=EPS_COLUMNS)


def fetch_company_facts(client: HttpClient, ciks: dict[str, int], workers: int = 4, on_progress=None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Point-in-time fundamentals and quarterly EPS for many tickers, from one company-facts download each.

    Returns (fundamentals with a ticker column, EPS with a ticker column, {ticker: why it failed}).
    """
    def one(item):
        ticker, cik = item
        try:
            facts = client.get_json(FACTS_URL.format(cik=cik))
            return ticker, point_in_time(facts), quarterly_eps(facts), None
        except Exception as exc:  # missing filer, network error, unexpected layout: skip this company, keep the reason
            return ticker, None, None, f"{type(exc).__name__}: {str(exc)[:120]}"

    frames, eps_frames, failed = [], [], {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (ticker, frame, eps, error) in enumerate(pool.map(one, ciks.items()), start=1):
            if frame is None or frame.empty:
                failed[ticker] = error or "no usable facts"
            else:
                frames.append(frame.assign(ticker=ticker))
            if eps is not None and not eps.empty:
                eps_frames.append(eps.assign(ticker=ticker))
            if on_progress and done % 50 == 0:
                on_progress(done, len(ciks))
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["ticker", *PIT_COLUMNS])
    eps = pd.concat(eps_frames, ignore_index=True) if eps_frames else pd.DataFrame(columns=["ticker", *EPS_COLUMNS])
    return table, eps, failed


def fetch_fundamentals(client: HttpClient, ciks: dict[str, int], workers: int = 4, on_progress=None) -> tuple[pd.DataFrame, dict[str, str]]:
    """Point-in-time fundamentals for many tickers. Returns (table with a ticker column, {ticker: why it failed})."""
    table, _, failed = fetch_company_facts(client, ciks, workers, on_progress)
    return table, failed


# -- submissions: industry codes and earnings-release dates ---------------------------------------------
SUBMISSIONS_FILE_URL = "https://data.sec.gov/submissions/{name}"


def _filings_frame(block: dict) -> pd.DataFrame:
    df = pd.DataFrame({key: block.get(key, []) for key in ["form", "filingDate", "acceptanceDateTime", "items"]})
    return df


def earnings_release_dates(filings: pd.DataFrame) -> pd.DataFrame:
    """Sessions on which each earnings release (8-K item 2.02) first reached the market.

    EDGAR's acceptanceDateTime is Eastern time despite its "Z" suffix. A release accepted at or after
    the 4 pm close can only move the next session, so it is dated to the following day; the features
    then align it to the next trading day. Returns columns: date, accepted.
    """
    if filings.empty:
        return pd.DataFrame(columns=["date", "accepted"])
    releases = filings[filings["form"].astype(str).str.startswith("8-K") & filings["items"].astype(str).str.contains("2.02", regex=False)]
    accepted = pd.to_datetime(releases["acceptanceDateTime"].astype(str).str.replace("Z", "", regex=False), errors="coerce")
    accepted = accepted.fillna(pd.to_datetime(releases["filingDate"], errors="coerce") + pd.Timedelta(hours=8))
    after_close = accepted.dt.hour >= 16
    dates = accepted.dt.normalize() + pd.to_timedelta(after_close.astype(int), unit="D")
    out = pd.DataFrame({"date": _ns(dates), "accepted": _ns(accepted)}).dropna()
    return out.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def fetch_submissions(client: HttpClient, cik: int, since: str | pd.Timestamp | None = None) -> tuple[dict, pd.DataFrame]:
    """(company info with sic and sic_description, every filing since `since`).

    The main file holds the most recent filings; older ones are paged into extra files, fetched only
    when they reach back past `since`.
    """
    data = client.get_json(SUBMISSIONS_URL.format(cik=cik))
    info = {"sic": pd.to_numeric(data.get("sic"), errors="coerce"), "sic_description": data.get("sicDescription") or None}
    filings = data.get("filings", {})
    frames = [_filings_frame(filings.get("recent", {}))]
    since = pd.Timestamp(since) if since is not None else None
    for extra in filings.get("files", []):
        if since is not None and pd.to_datetime(extra.get("filingTo"), errors="coerce") < since:
            continue
        frames.append(_filings_frame(client.get_json(SUBMISSIONS_FILE_URL.format(name=extra["name"]))))
    table = pd.concat(frames, ignore_index=True)
    if since is not None and not table.empty:
        table = table[pd.to_datetime(table["filingDate"], errors="coerce") >= since]
    return info, table.reset_index(drop=True)


def fetch_company_events(client: HttpClient, ciks: dict[str, int], since: str | None = None, workers: int = 4, on_progress=None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Industry codes and earnings-release dates for many tickers.

    Returns (ticker, sic, sic_description), (ticker, date, accepted) and {ticker: why it failed}.
    """
    def one(item):
        ticker, cik = item
        try:
            info, filings = fetch_submissions(client, cik, since)
            return ticker, info, earnings_release_dates(filings), None
        except Exception as exc:
            return ticker, None, None, f"{type(exc).__name__}: {str(exc)[:120]}"

    infos, releases, failed = [], [], {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (ticker, info, dates, error) in enumerate(pool.map(one, ciks.items()), start=1):
            if info is None:
                failed[ticker] = error
                continue
            infos.append({"ticker": ticker, **info})
            if not dates.empty:
                releases.append(dates.assign(ticker=ticker))
            if on_progress and done % 50 == 0:
                on_progress(done, len(ciks))
    industries = pd.DataFrame(infos, columns=["ticker", "sic", "sic_description"])
    earnings = pd.concat(releases, ignore_index=True)[["ticker", "date", "accepted"]] if releases else pd.DataFrame(columns=["ticker", "date", "accepted"])
    return industries, earnings, failed


# SIC code ranges mapped to the closest GICS sector. Used for companies with no GICS sector on record
# (stocks that left the index), so their sector is a real industry rather than a flag saying "left the
# index". First match wins; ranges are inclusive.
SIC_SECTORS = [
    (2830, 2836, "Health Care"), (3840, 3851, "Health Care"), (5122, 5122, "Health Care"), (8000, 8099, "Health Care"),
    (3570, 3579, "Information Technology"), (3600, 3611, "Information Technology"), (3614, 3619, "Information Technology"),
    (3640, 3699, "Information Technology"), (3820, 3829, "Information Technology"), (3861, 3861, "Information Technology"),
    (5045, 5045, "Information Technology"), (7370, 7379, "Information Technology"),
    (1200, 1399, "Energy"), (2900, 2999, "Energy"),
    (4900, 4949, "Utilities"), (4960, 4999, "Utilities"),
    (6500, 6553, "Real Estate"), (6798, 6798, "Real Estate"),
    (6000, 6799, "Financials"), (9995, 9995, "Financials"),
    (2700, 2799, "Communication Services"), (4800, 4899, "Communication Services"), (7800, 7899, "Communication Services"),
    (100, 999, "Consumer Staples"), (2000, 2199, "Consumer Staples"), (2840, 2844, "Consumer Staples"), (5140, 5149, "Consumer Staples"),
    (5331, 5331, "Consumer Staples"), (5399, 5399, "Consumer Staples"), (5400, 5499, "Consumer Staples"), (5912, 5912, "Consumer Staples"),
    (1531, 1531, "Consumer Discretionary"), (2200, 2399, "Consumer Discretionary"), (2500, 2599, "Consumer Discretionary"),
    (3021, 3021, "Consumer Discretionary"), (3100, 3199, "Consumer Discretionary"), (3630, 3639, "Consumer Discretionary"),
    (3710, 3716, "Consumer Discretionary"), (3900, 3999, "Consumer Discretionary"), (5200, 5999, "Consumer Discretionary"),
    (7000, 7099, "Consumer Discretionary"), (7200, 7299, "Consumer Discretionary"), (7500, 7599, "Consumer Discretionary"),
    (7900, 7999, "Consumer Discretionary"),
    (1000, 1099, "Materials"), (1400, 1499, "Materials"), (2400, 2499, "Materials"), (2600, 2699, "Materials"),
    (2800, 2899, "Materials"), (3000, 3099, "Materials"), (3200, 3399, "Materials"),
    (1500, 1799, "Industrials"), (3400, 3599, "Industrials"), (3612, 3629, "Industrials"), (3700, 3799, "Industrials"),
    (3800, 3899, "Information Technology"), (4000, 4799, "Industrials"), (4950, 4959, "Industrials"), (5000, 5199, "Industrials"),
    (7300, 7399, "Industrials"), (8100, 8999, "Industrials"),
]


def sic_sector(sic) -> str | None:
    """The GICS-style sector for a four-digit SIC code, or None when it is missing or unmapped."""
    if sic is None or pd.isna(sic):
        return None
    code = int(sic)
    return next((sector for low, high, sector in SIC_SECTORS if low <= code <= high), None)


def sic_industry(sic) -> str | None:
    """A coarse industry key: the two-digit SIC major group (e.g. 'SIC 28' chemicals and drugs)."""
    if sic is None or pd.isna(sic):
        return None
    return f"SIC {int(sic) // 100:02d}"


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
