"""Insider trades from SEC's Form 3/4/5 bulk data sets (free; needs the same contact User-Agent as EDGAR).

SEC publishes every insider filing as quarterly tab-separated tables. This module keeps open-market
purchases (transaction code P) and sales (S) of common stock by each company's officers, directors and
large holders, dated by when the filing reached EDGAR. Purchases are the informative side: insiders sell
for many reasons (diversification, taxes, pre-planned 10b5-1 sales) but buy for one.

The files are large (about 10-20 MB zipped per quarter), so the pipeline only downloads them when asked
(`run_pipeline.py --insiders`) and caches them.
"""

from __future__ import annotations

import io
import zipfile

import pandas as pd

from .http import HttpClient

DATASET_URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{year}q{quarter}_form345.zip"
COLUMNS = ["ticker", "filed", "accession", "code", "shares", "value"]


def quarters_since(start: str | pd.Timestamp, today: pd.Timestamp | None = None) -> list[tuple[int, int]]:
    """(year, quarter) pairs from `start` through the last completed quarter."""
    start = pd.Timestamp(start)
    last = pd.Timestamp(today or pd.Timestamp.today()).to_period("Q") - 1
    periods = pd.period_range(start.to_period("Q"), last, freq="Q")
    return [(p.year, p.quarter) for p in periods]


def _read_tsv(archive: zipfile.ZipFile, name: str, columns: list[str]) -> pd.DataFrame:
    member = next(n for n in archive.namelist() if n.upper().endswith(name))
    with archive.open(member) as handle:
        frame = pd.read_csv(handle, sep="\t", dtype=str, usecols=lambda c: c.upper() in columns, quoting=3, on_bad_lines="skip")
    frame.columns = [c.upper() for c in frame.columns]
    return frame


def parse_dataset(content: bytes, ciks: dict[int, str]) -> pd.DataFrame:
    """Open-market purchases and sales for the companies in `ciks` ({cik: ticker}) from one quarterly zip."""
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        submissions = _read_tsv(archive, "SUBMISSION.TSV", ["ACCESSION_NUMBER", "FILING_DATE", "ISSUERCIK", "DOCUMENT_TYPE"])
        trades = _read_tsv(archive, "NONDERIV_TRANS.TSV", ["ACCESSION_NUMBER", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD"])
    submissions["ISSUERCIK"] = pd.to_numeric(submissions["ISSUERCIK"], errors="coerce")
    submissions = submissions[submissions["ISSUERCIK"].isin(ciks)]
    trades = trades[trades["TRANS_CODE"].isin(["P", "S"])]
    merged = trades.merge(submissions, on="ACCESSION_NUMBER", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=COLUMNS)
    shares = pd.to_numeric(merged["TRANS_SHARES"], errors="coerce")
    price = pd.to_numeric(merged["TRANS_PRICEPERSHARE"], errors="coerce")
    out = pd.DataFrame({
        "ticker": merged["ISSUERCIK"].astype(int).map(ciks),
        # SEC writes dates like 31-MAR-2023.
        "filed": pd.to_datetime(merged["FILING_DATE"], format="%d-%b-%Y", errors="coerce").astype("datetime64[ns]"),
        "accession": merged["ACCESSION_NUMBER"],
        "code": merged["TRANS_CODE"],
        "shares": shares,
        "value": shares * price,
    })
    return out.dropna(subset=["ticker", "filed"]).reset_index(drop=True)


def fetch_insider_trades(client: HttpClient, ciks: dict[str, int], start: str, on_progress=None) -> tuple[pd.DataFrame, list[str]]:
    """Purchases and sales since `start` for the given tickers. Returns (trades, quarters that failed)."""
    by_cik = {int(cik): ticker for ticker, cik in ciks.items()}
    frames, failed = [], []
    for year, quarter in quarters_since(start):
        try:
            frames.append(parse_dataset(client.get(DATASET_URL.format(year=year, quarter=quarter)).content, by_cik))
        except Exception as exc:  # a missing or malformed quarter shouldn't stop the rest
            failed.append(f"{year}Q{quarter}: {type(exc).__name__}")
        if on_progress:
            on_progress(f"{year}Q{quarter}")
    trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    return trades.drop_duplicates(), failed
