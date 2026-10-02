from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from .config import (
    FUNDAMENTALS_FILE_CANDIDATES,
    MARKET_CLOSE_HOUR,
    MARKET_TIMEZONE,
    NEWS_FILE_CANDIDATES,
    PRICE_FILE_CANDIDATES,
    RAW_DIR,
)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )
    rename_map = {
        "symbol": "ticker",
        "stock": "ticker",
        "adj_close": "close",
        "adjusted_close": "close",
        "closing_price": "close",
        "headline": "title",
        "pubdate": "date",
        "published_date": "date",
        "datetime": "date",
        "marketcap": "market_cap",
        "trailingpe": "pe_ratio",
        "totalrevenue": "revenue",
        "profitmargins": "profit_margin",
        "debttoequity": "debt_to_equity",
        "returnonequity": "roe",
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


def _read_csv_head(path: Path) -> set[str]:
    return set(_normalize_columns(pd.read_csv(path, nrows=5)).columns)


def _all_csv_files(raw_dir: Path = RAW_DIR) -> list[Path]:
    # Per-ticker files in price_data/ are loaded together by _load_kaggle_price_folder,
    # so they are never candidates when guessing which single file holds a table.
    return sorted(
        path
        for path in raw_dir.rglob("*.csv")
        if "__MACOSX" not in path.parts and "price_data" not in path.parts
    )


def _kaggle_dataset_dir(raw_dir: Path = RAW_DIR) -> Path | None:
    """The folder that contains price_data/, wherever the download or ZIP placed it."""
    if not raw_dir.exists():
        return None
    for price_dir in sorted(raw_dir.rglob("price_data")):
        if price_dir.is_dir() and "__MACOSX" not in price_dir.parts:
            return price_dir.parent
    return None


def _to_market_time(values: pd.Series) -> pd.Series:
    """Parse timestamps as naive US market time.

    Values with a UTC offset are converted to market time; values without one are taken as already
    in market time. The two are parsed separately because pandas drops one kind when a column mixes them.
    """
    if is_datetime64_any_dtype(values):
        parsed = pd.to_datetime(values)
        return parsed.dt.tz_convert(MARKET_TIMEZONE).dt.tz_localize(None) if parsed.dt.tz is not None else parsed
    text = values.astype("string").str.strip()
    has_offset = text.str.contains(r"\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*(?:Z|[+-]\d{2}:?\d{2})$", regex=True, na=False)
    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if has_offset.any():
        aware = pd.to_datetime(text[has_offset], errors="coerce", utc=True, format="mixed")
        result[has_offset] = aware.dt.tz_convert(MARKET_TIMEZONE).dt.tz_localize(None).astype("datetime64[ns]")
    if (~has_offset).any():
        result[~has_offset] = pd.to_datetime(text[~has_offset], errors="coerce", format="mixed").astype("datetime64[ns]")
    return result


def _score_file(path: Path, wanted: str) -> int:
    name = path.name.lower()
    columns = _read_csv_head(path)
    score = 0
    if wanted == "prices":
        if "close" not in columns:
            return 0
        score += 5 * len({"ticker", "date", "close"}.intersection(columns))
        score += 2 * any(word in name for word in ["price", "stock", "historical"])
    elif wanted == "fundamentals":
        score += 5 * ("ticker" in columns)
        score += 2 * len({"sector", "market_cap", "pe_ratio", "revenue", "profit_margin"}.intersection(columns))
        score += 2 * any(word in name for word in ["fundamental", "financial", "company", "info"])
        if name == "01_company_info.csv":
            score += 20
    elif wanted == "news":
        score += 5 * len({"ticker", "date"}.intersection(columns))
        score += 4 * bool({"title", "summary", "content", "description"}.intersection(columns))
        score += 2 * any(word in name for word in ["news", "headline", "sentiment"])
        if name == "02_company_news_sentiment.csv":
            score += 20
    return score


def _find_file(candidates: list[str], wanted: str, raw_dir: Path = RAW_DIR) -> Path:
    dataset_dir = _kaggle_dataset_dir(raw_dir)
    search_dirs = [raw_dir]
    if dataset_dir is not None:
        search_dirs.insert(0, dataset_dir)

    for base_dir in search_dirs:
        for name in candidates:
            path = base_dir / name
            if path.exists():
                return path

    csv_files = _all_csv_files(raw_dir)
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. Run python download_kaggle.py, use --from-zip, or use --make-sample."
        )

    scored = [(path, _score_file(path, wanted)) for path in csv_files]
    scored.sort(key=lambda item: item[1], reverse=True)
    best_path, best_score = scored[0]
    if best_score <= 0:
        raise FileNotFoundError(
            f"Could not infer a {wanted} file. Check data/raw or rename a CSV to a known name. "
            "If you imported the Kaggle data before per-ticker folders were preserved, "
            "re-run the import with --overwrite."
        )
    return best_path


def _load_kaggle_price_folder(raw_dir: Path = RAW_DIR) -> pd.DataFrame | None:
    dataset_dir = _kaggle_dataset_dir(raw_dir)
    if dataset_dir is None:
        return None

    price_dir = dataset_dir / "price_data"
    if not price_dir.exists():
        return None

    frames = []
    for csv_file in sorted(price_dir.glob("*.csv")):
        frame = _normalize_columns(pd.read_csv(csv_file))
        if "ticker" not in frame.columns:
            frame["ticker"] = csv_file.stem.upper()
        frames.append(frame)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_prices(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = _load_kaggle_price_folder(raw_dir)
    if df is None:
        df = _normalize_columns(pd.read_csv(_find_file(PRICE_FILE_CANDIDATES, "prices", raw_dir)))

    required = {"ticker", "date", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Price data is missing required columns: {sorted(missing)}")
    df["date"] = _to_market_time(df["date"]).dt.normalize()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna(subset=["ticker", "date", "close"])


def load_fundamentals(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = _normalize_columns(pd.read_csv(_find_file(FUNDAMENTALS_FILE_CANDIDATES, "fundamentals", raw_dir)))
    if "ticker" not in df.columns:
        raise ValueError("Fundamentals file is missing required column: ticker")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df.drop_duplicates(subset=["ticker"], keep="last")


def load_news(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = _normalize_columns(pd.read_csv(_find_file(NEWS_FILE_CANDIDATES, "news", raw_dir)))
    required = {"ticker", "date"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"News file is missing required columns: {sorted(missing)}")
    if "title" not in df.columns:
        text_columns = [col for col in ["summary", "content", "description"] if col in df.columns]
        if not text_columns:
            raise ValueError("News file needs title/headline, summary, content, or description.")
        df["title"] = df[text_columns[0]]
    if "sentiment" not in df.columns and "lm_sentiment" in df.columns:
        df["sentiment"] = df["lm_sentiment"]
    timestamps = _to_market_time(df["date"])
    # News published at or after the close can only affect the next session.
    after_close = timestamps.dt.hour >= MARKET_CLOSE_HOUR
    df["date"] = timestamps.dt.normalize() + pd.to_timedelta(after_close.astype(int), unit="D")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df.dropna(subset=["ticker", "date"])
