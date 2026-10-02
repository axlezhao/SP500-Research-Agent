"""Macro series from FRED (Federal Reserve Bank of St. Louis), via the public CSV endpoint (no key).

Values are shifted one day before use: a daily series is published after its date closes, so on
date d the model may only know values up to d - 1.
"""

from __future__ import annotations

import io

import pandas as pd

from .http import HttpClient

CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
SERIES = {
    "DGS3MO": "tbill_3m",  # 3-month Treasury bill, % a year (the risk-free rate)
    "DGS10": "treasury_10y",  # 10-year Treasury yield, %
    "VIXCLS": "vix",  # CBOE volatility index
}


def parse_series(csv_text: str, series_id: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(csv_text))
    date_col = df.columns[0]
    values = pd.to_numeric(df[series_id], errors="coerce")  # FRED marks missing days with "."
    return pd.Series(values.values, index=pd.to_datetime(df[date_col]).astype("datetime64[ns]"), name=series_id).dropna()


def fetch_macro(client: HttpClient | None = None, start: str = "2000-01-01") -> pd.DataFrame:
    """Daily frame: date, tbill_3m, treasury_10y, vix, term_spread (10y minus 3m)."""
    client = client or HttpClient(max_per_second=2)
    columns = {}
    for series_id, name in SERIES.items():
        columns[name] = parse_series(client.get_text(CSV_URL, {"id": series_id, "cosd": start}), series_id)
    macro = pd.DataFrame(columns).sort_index()
    macro["term_spread"] = macro["treasury_10y"] - macro["tbill_3m"]
    return macro.rename_axis("date").reset_index()
