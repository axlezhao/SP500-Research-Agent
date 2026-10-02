"""S&P 500 membership from Wikipedia: today's constituents plus the dated list of index changes.

Walking the changes backwards from today's list reconstructs who was in the index on any date,
which lets the model train and the backtest trade only on stocks that were actually members then.
Wikipedia is community-maintained; tickers that were later renamed can leave small gaps, which
build_membership counts in its `unresolved` report.
"""

from __future__ import annotations

import io

import pandas as pd

from .http import HttpClient

CONSTITUENTS_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CHANGES_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"


def normalise_ticker(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().upper().replace("-", ".")
    return text or None


def parse_constituents(html: str) -> pd.DataFrame:
    table = next(t for t in pd.read_html(io.StringIO(html)) if "Symbol" in t.columns)
    out = pd.DataFrame(
        {
            "ticker": table["Symbol"].map(normalise_ticker),
            "company_name": table["Security"],
            "sector": table["GICS Sector"],
            "sub_industry": table.get("GICS Sub-Industry"),
            "cik": pd.to_numeric(table.get("CIK"), errors="coerce").astype("Int64"),
            "date_added": pd.to_datetime(table.get("Date added"), errors="coerce"),
        }
    )
    return out.dropna(subset=["ticker"]).drop_duplicates("ticker").reset_index(drop=True)


def parse_changes(html: str) -> pd.DataFrame:
    table = max(pd.read_html(io.StringIO(html)), key=len)
    if isinstance(table.columns, pd.MultiIndex):
        table.columns = ["_".join(dict.fromkeys(str(level) for level in col)) for col in table.columns]
    columns = {c.lower().replace(" ", "_"): c for c in table.columns}
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(table[columns["effective_date"]], errors="coerce"),
            "added": table[columns["added_ticker"]].map(normalise_ticker),
            "added_name": table[columns["added_security"]],
            "removed": table[columns["removed_ticker"]].map(normalise_ticker),
            "removed_name": table[columns["removed_security"]],
            "reason": table[columns["reason"]] if "reason" in columns else None,
        }
    )
    return out.dropna(subset=["date"]).sort_values("date", ascending=False).reset_index(drop=True)


def fetch_constituents(client: HttpClient) -> pd.DataFrame:
    return parse_constituents(client.get_text(CONSTITUENTS_URL))


def fetch_changes(client: HttpClient) -> pd.DataFrame:
    return parse_changes(client.get_text(CHANGES_URL))


def build_membership(constituents: pd.DataFrame, changes: pd.DataFrame, since: str | pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """Membership intervals [start, end) per ticker for dates on or after `since`.

    start is NaT when the stock was already a member at `since`; end is NaT for current members.
    """
    since = pd.Timestamp(since)
    open_start: dict[str, pd.Timestamp | None] = {t: None for t in constituents["ticker"]}
    open_end: dict[str, pd.Timestamp | None] = {t: None for t in constituents["ticker"]}
    intervals = []
    unresolved = 0
    for change in changes[changes["date"] >= since].itertuples(index=False):  # newest first
        added = change.added if isinstance(change.added, str) else None
        removed = change.removed if isinstance(change.removed, str) else None  # NaN is truthy, so check the type
        if added:
            if added in open_end:
                intervals.append((added, change.date, open_end.pop(added)))
                open_start.pop(added, None)
            else:
                unresolved += 1  # e.g. a ticker later renamed: we can't place it in today's list
        if removed:
            if removed in open_end:
                unresolved += 1  # removed while (by our reconstruction) already out
            else:
                open_end[removed] = change.date
    for ticker, end in open_end.items():
        intervals.append((ticker, pd.NaT, end))
    membership = pd.DataFrame(intervals, columns=["ticker", "start", "end"])
    membership[["start", "end"]] = membership[["start", "end"]].apply(pd.to_datetime)
    report = {
        "changes_used": int((changes["date"] >= since).sum()),
        "unresolved_changes": unresolved,
        "tickers_ever_in_index": int(membership["ticker"].nunique()),
        "current_members": int(len(constituents)),
    }
    return membership.sort_values(["ticker", "start"], na_position="first").reset_index(drop=True), report


def membership_flags(frame: pd.DataFrame, membership: pd.DataFrame) -> pd.Series:
    """Boolean Series aligned to `frame` (needs ticker and date): was the stock in the index that day?"""
    keys = frame[["ticker", "date"]].reset_index()
    merged = keys.merge(membership, on="ticker", how="inner")
    inside = (merged["start"].isna() | (merged["date"] >= merged["start"])) & (merged["end"].isna() | (merged["date"] < merged["end"]))
    member_index = merged.loc[inside, "index"].unique()
    return pd.Series(frame.index.isin(member_index), index=frame.index)
