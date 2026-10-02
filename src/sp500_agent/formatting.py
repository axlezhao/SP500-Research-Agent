"""Display helpers shared by the CLI brief and the Streamlit app."""

from __future__ import annotations

import pandas as pd

CONSTRUCTIVE_THRESHOLD = 0.60
CAUTIOUS_THRESHOLD = 0.40
COMPANY_NAME_COLUMNS = ["displayname", "longname", "shortname", "company_name"]


def _missing(value) -> bool:
    return value is None or pd.isna(value)


def fmt_pct(value) -> str:
    return "n/a" if _missing(value) else f"{float(value) * 100:.1f}%"


def fmt_prob(value) -> str:
    return "n/a" if _missing(value) else f"{float(value):.1%}"


def fmt_price(value) -> str:
    return "n/a" if _missing(value) else f"${float(value):,.2f}"


def fmt_money(value) -> str:
    """Large dollar amounts such as market cap and revenue."""
    if _missing(value):
        return "n/a"
    value = float(value)
    for divisor, suffix in [(1e12, "T"), (1e9, "B"), (1e6, "M")]:
        if abs(value) >= divisor:
            return f"${value / divisor:.2f}{suffix}" if suffix == "T" else f"${value / divisor:.1f}{suffix}"
    return f"${value:,.0f}"


def fmt_num(value, decimals: int = 2) -> str:
    return "n/a" if _missing(value) else f"{float(value):.{decimals}f}"


def stance(probability: float) -> str:
    if probability >= CONSTRUCTIVE_THRESHOLD:
        return "constructive"
    if probability <= CAUTIOUS_THRESHOLD:
        return "cautious"
    return "neutral"


def company_label(row: pd.Series) -> str:
    for col in COMPANY_NAME_COLUMNS:
        if col in row and not _missing(row[col]):
            return str(row[col])
    return str(row["ticker"])
