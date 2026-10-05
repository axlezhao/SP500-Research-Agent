"""Daily factor returns from Kenneth French's data library (free, no key).

Used as an independent check on the project's home-made factors: the Fama-French five factors
(market, size, value, profitability, investment), momentum and short-term reversal, built by French
from all NYSE, AMEX and NASDAQ stocks. Values are published as percent per day, about two months in
arrears, so the most recent backtest periods have no factor data.
"""

from __future__ import annotations

import io
import re
import zipfile

import pandas as pd

from .http import HttpClient

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
FILES = {
    "F-F_Research_Data_5_Factors_2x3_daily": {"Mkt-RF": "MKT", "SMB": "SMB", "HML": "HML", "RMW": "RMW", "CMA": "CMA", "RF": "RF"},
    "F-F_Momentum_Factor_daily": {"Mom": "MOM"},
    "F-F_ST_Reversal_Factor_daily": {"ST_Rev": "STREV"},
}
FACTOR_LABELS = {
    "MKT": "Market (excess of T-bills)",
    "SMB": "Size",
    "HML": "Value",
    "RMW": "Profitability (robust minus weak)",
    "CMA": "Investment (conservative minus aggressive)",
    "MOM": "Momentum",
    "STREV": "Short-term reversal",
}


def parse_daily_csv(text: str, columns: dict[str, str]) -> pd.DataFrame:
    """The daily table in one of French's CSV files: header row starting with a comma, then YYYYMMDD rows."""
    lines = text.replace("\r", "").split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith(","))
    header = [c.strip() for c in lines[start].split(",")]
    rows = []
    for line in lines[start + 1 :]:
        if not re.match(r"^\s*\d{8}\s*,", line):
            break
        rows.append([c.strip() for c in line.split(",")])
    frame = pd.DataFrame(rows, columns=["date", *header[1:]])
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d").astype("datetime64[ns]")
    keep = {raw: name for raw, name in columns.items() if raw in frame.columns}
    if len(keep) < len(columns):
        # Column names have varied over the years (e.g. "Mom   " or "ST_Rev"); match ignoring case and spaces.
        normalised = {c.replace(" ", "").lower(): c for c in frame.columns}
        keep = {normalised[raw.replace(" ", "").lower()]: name for raw, name in columns.items() if raw.replace(" ", "").lower() in normalised}
    out = frame[["date", *keep]].rename(columns=keep)
    for col in keep.values():
        out[col] = pd.to_numeric(out[col], errors="coerce") / 100  # percent -> decimal
    return out


def _unzip_text(content: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        return archive.read(name).decode("latin-1")


def fetch_factors(client: HttpClient | None = None) -> pd.DataFrame:
    """Daily frame: date, MKT, SMB, HML, RMW, CMA, RF, MOM, STREV (decimal returns)."""
    client = client or HttpClient(max_per_second=2)
    merged = None
    for name, columns in FILES.items():
        frame = parse_daily_csv(_unzip_text(client.get(BASE_URL.format(name=name)).content), columns)
        merged = frame if merged is None else merged.merge(frame, on="date", how="outer")
    return merged.sort_values("date").reset_index(drop=True)


def period_factor_returns(factors: pd.DataFrame, dates, holding_days: int, lag: int) -> pd.DataFrame:
    """Compound daily factor returns over each backtest holding window.

    A position entered `lag` sessions after the signal date and held `holding_days` sessions earns the
    returns of sessions lag+1 .. lag+holding_days after it. Windows the factor data doesn't cover are NaN.
    """
    daily = factors.set_index("date").sort_index()
    index = daily.index
    values = daily.to_numpy()
    out = []
    for date in pd.to_datetime(pd.Index(dates)).astype("datetime64[ns]"):
        i = index.searchsorted(date)
        start, end = i + lag + 1, i + lag + holding_days + 1
        if i >= len(index) or index[i] != date or end > len(index):
            out.append([float("nan")] * values.shape[1])
            continue
        out.append(((1 + values[start:end]).prod(axis=0) - 1).tolist())
    return pd.DataFrame(out, columns=daily.columns, index=pd.Index(dates, name="date"))
