"""Forward test: the only truly out-of-sample evidence.

Every pipeline run records the rankings it would trade today. Later runs score each recorded ranking
once its forward returns are known. Unlike any backtest, nothing about these rankings could have been
tuned on the returns they are judged against.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import REPORT_DIR, ResearchSpec
from .stats import information_coefficients

SIGNALS_DIR = REPORT_DIR / "live_signals"


def record_signals(scored: pd.DataFrame, bundle: dict, directory: Path = SIGNALS_DIR) -> Path:
    """Save today's ranking (one file per as-of date; a rerun on the same date replaces it)."""
    directory.mkdir(parents=True, exist_ok=True)
    as_of = pd.Timestamp(scored["date"].max())
    frame = scored[["date", "ticker", "model_probability", "rank"]].assign(
        model=bundle.get("model_name"), target=(bundle.get("spec") or {}).get("label"), trained_through=bundle.get("trained_through"),
    )
    path = directory / f"{as_of.date()}.parquet"
    frame.to_parquet(path, index=False)
    return path


def evaluate_signals(features: pd.DataFrame, spec: ResearchSpec, directory: Path = SIGNALS_DIR) -> pd.DataFrame:
    """For each recorded ranking whose returns are now known: rank IC and top-minus-bottom quintile return."""
    if not directory.exists():
        return pd.DataFrame()
    outcomes = features[["ticker", "date", spec.tradable_column]].rename(columns={spec.tradable_column: "realised"})
    rows = []
    for path in sorted(directory.glob("*.parquet")):
        recorded = pd.read_parquet(path).merge(outcomes, on=["ticker", "date"], how="left").dropna(subset=["realised"])
        if len(recorded) < 5:
            continue
        ic = information_coefficients(recorded, "model_probability", "realised")
        quintile = pd.qcut(recorded["model_probability"].rank(method="first"), 5, labels=False)
        spread = recorded.loc[quintile == 4, "realised"].mean() - recorded.loc[quintile == 0, "realised"].mean()
        rows.append({"as_of": recorded["date"].iloc[0], "stocks": len(recorded), "ic": float(ic.iloc[0]) if len(ic) else None, "top_minus_bottom": spread, "model": recorded["model"].iloc[0]})
    return pd.DataFrame(rows)
