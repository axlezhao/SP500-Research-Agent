"""End-to-end research run: data summary, model comparison, backtest, saved artifacts and a markdown report."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .attribution import FACTOR_LABELS, attribute
from .backtest import MARKET, BacktestConfig, BacktestResult, config_dict, run_backtest
from .config import EXECUTION_LAG_DAYS, EXPERIMENT_SPECS, MODEL_PATH, PRODUCTION_SPEC, RESEARCH_DIR, ResearchSpec
from .model import available_features, investable, train_final_model
from .validation import calibration_table, compare_models, feature_importance, signal_report, summarise, walk_forward

STRATEGY_NAMES = {"long_only": "Long top 20%", "long_short": "Long-short", "benchmark": "Equal-weight benchmark", MARKET: "S&P 500 (SPY)"}
EXPERIMENT_MODEL = "logistic_regression"

ARTIFACT_FILES = {
    "eda": "eda.json",
    "model_comparison": "model_comparison.csv",
    "fold_metrics": "fold_metrics.csv",
    "oos_predictions": "oos_predictions.parquet",
    "calibration": "calibration.csv",
    "feature_importance": "feature_importance.csv",
    "signal_ic": "signal_ic.csv",
    "backtest_returns": "backtest_returns.parquet",
    "backtest_summary": "backtest_summary.csv",
    "quantile_returns": "quantile_returns.csv",
    "backtest_config": "backtest_config.json",
    "attribution": "attribution.csv",
    "factor_returns": "factor_returns.parquet",
    "experiments": "experiments.csv",
    "spec": "spec.json",
    "report": "research_report.md",
}


@dataclass
class ResearchRun:
    eda: dict
    comparison: pd.DataFrame
    fold_metrics: pd.DataFrame
    best_model: str
    predictions: pd.DataFrame
    calibration: pd.DataFrame
    importance: pd.DataFrame
    signals: pd.DataFrame
    backtest: BacktestResult
    bundle: dict
    report: str
    quality: dict | None = None
    spec: ResearchSpec = PRODUCTION_SPEC
    attribution: pd.DataFrame | None = None
    factors: pd.DataFrame | None = None
    experiments: pd.DataFrame | None = None


def exploratory_summary(features: pd.DataFrame, spec: ResearchSpec = PRODUCTION_SPEC) -> dict:
    universe = investable(features)
    labelled = universe.dropna(subset=["future_return_5d"])
    returns = labelled["future_return_5d"]
    numeric, _ = available_features(labelled)
    news_days = universe["news_count_20d"] > 0 if "news_count_20d" in universe else pd.Series(False, index=universe.index)
    return {
        "rows": len(features),
        "index_member_rows": len(universe),
        "tickers": int(universe["ticker"].nunique()),
        "model_features": numeric,
        "fundamentals_coverage": float(universe["fundamentals_as_of"].notna().mean()) if "fundamentals_as_of" in universe else None,
        "sectors": int(features["sector"].nunique()) if "sector" in features else 0,
        "start": str(features["date"].min().date()),
        "end": str(features["date"].max().date()),
        "trading_days": int(features["date"].nunique()),
        "target": spec.description,
        "target_rate": float(labelled[spec.target_column].mean()),
        "up_rate_5d": float((labelled["future_return_5d"] > 0).mean()),
        "forward_return_5d": {
            "mean": float(returns.mean()),
            "std": float(returns.std()),
            "skew": float(returns.skew()),
            "excess_kurtosis": float(returns.kurt()),
            "p01": float(returns.quantile(0.01)),
            "p99": float(returns.quantile(0.99)),
        },
        "share_of_rows_with_recent_news": float(news_days.mean()),
        "missing_share": {col: float(universe[col].isna().mean()) for col in numeric},
    }


def run_research(
    features: pd.DataFrame,
    n_splits: int = 5,
    backtest_config: BacktestConfig | None = None,
    output_dir: Path | None = RESEARCH_DIR,
    model_path: Path | None = MODEL_PATH,
    max_rows: int = 150_000,
    quality: dict | None = None,
    spec: ResearchSpec = PRODUCTION_SPEC,
    index_prices: pd.Series | None = None,
    experiment_specs: tuple[ResearchSpec, ...] | None = EXPERIMENT_SPECS,
) -> ResearchRun:
    """Compare models for `spec`, backtest and attribute the best, compare research setups, save everything."""
    eda = exploratory_summary(features, spec)
    comparison, results = compare_models(features, n_splits=n_splits, max_rows=max_rows, spec=spec)
    best = comparison.iloc[0]["model"]
    best_result = results[best]
    predictions = best_result.predictions
    # 3-month T-bill yield (FRED, % a year) as the cash return for Sharpe ratios, when available.
    risk_free = features.groupby("date")["tbill_3m"].first() / 100 if "tbill_3m" in features.columns else None
    config = backtest_config_for(spec, backtest_config)
    backtest = run_backtest(predictions, config, risk_free=risk_free, index_prices=index_prices)
    attribution, factors = attribute(features, backtest.returns, spec.horizon, spec.tradable_column)
    experiments = (
        run_experiments(features, experiment_specs, n_splits, max_rows, backtest_config, risk_free, index_prices, production=spec)
        if experiment_specs
        else None
    )
    validation = comparison.iloc[0].to_dict()
    bundle = train_final_model(features, best, validation=validation, max_rows=max_rows, model_path=model_path, spec=spec)
    run = ResearchRun(
        eda=eda,
        comparison=comparison,
        fold_metrics=pd.concat([r.fold_metrics.assign(model=name) for name, r in results.items()], ignore_index=True),
        best_model=best,
        predictions=predictions,
        calibration=calibration_table(predictions),
        importance=feature_importance(best_result),
        signals=signal_report(features, predictions, spec),
        backtest=backtest,
        bundle=bundle,
        report="",
        quality=quality,
        spec=spec,
        attribution=attribution,
        factors=factors,
        experiments=experiments,
    )
    run.report = render_report(run)
    if output_dir is not None:
        save_artifacts(run, output_dir)
    return run


def backtest_config_for(spec: ResearchSpec, base: BacktestConfig | None = None) -> BacktestConfig:
    """Hold for the prediction horizon; a sector-relative target trades sector-neutral portfolios."""
    return replace(base or BacktestConfig(), holding_days=spec.horizon, neutralize="sector" if spec.relative_to == "sector" else "none")


def run_experiments(
    features: pd.DataFrame,
    specs: tuple[ResearchSpec, ...],
    n_splits: int,
    max_rows: int,
    base_config: BacktestConfig | None,
    risk_free: pd.Series | None,
    index_prices: pd.Series | None,
    production: ResearchSpec = PRODUCTION_SPEC,
) -> pd.DataFrame:
    """Each research setup walked forward with the same model, backtested and attributed.

    The production setup is chosen by a rule fixed in advance: the highest out-of-sample rank-IC t-stat.
    """
    rows = []
    for spec in specs:
        try:
            result = walk_forward(features, EXPERIMENT_MODEL, n_splits, max_rows, spec)
        except ValueError as exc:  # e.g. no sector has enough members for a sector-relative target
            rows.append({"setup": spec.label, "horizon": spec.horizon, "relative_to": spec.relative_to, "production": spec == production, "note": str(exc)})
            continue
        quality = summarise(result)
        config = backtest_config_for(spec, base_config)
        backtest = run_backtest(result.predictions, config, risk_free=risk_free, index_prices=index_prices)
        stats = backtest.summary.set_index("strategy")
        attribution, _ = attribute(features, backtest.returns, spec.horizon, spec.tradable_column)
        alpha = attribution.set_index("strategy") if not attribution.empty else pd.DataFrame()
        reference = MARKET if MARKET in stats.index else "benchmark"
        rows.append({
            "setup": spec.label,
            "horizon": spec.horizon,
            "relative_to": spec.relative_to,
            "portfolio": "sector-neutral" if config.neutralize == "sector" else "top/bottom 20% overall",
            "production": spec == production,
            "auc": quality["auc_mean"],
            "ic_mean": quality["ic_mean"],
            "ic_tstat": quality["ic_tstat"],
            "long_only_cagr": stats.loc["long_only", "cagr"],
            "long_only_sharpe": stats.loc["long_only", "sharpe"],
            "reference": reference,
            "reference_cagr": stats.loc[reference, "cagr"],
            "reference_sharpe": stats.loc[reference, "sharpe"],
            "long_short_cagr": stats.loc["long_short", "cagr"],
            "long_short_sharpe": stats.loc["long_short", "sharpe"],
            "long_short_alpha": alpha.loc["long_short", "alpha_annual"] if "long_short" in alpha.index else np.nan,
            "long_short_alpha_t": alpha.loc["long_short", "alpha_tstat"] if "long_short" in alpha.index else np.nan,
            "long_only_alpha": alpha.loc["long_only", "alpha_annual"] if "long_only" in alpha.index else np.nan,
            "long_only_alpha_t": alpha.loc["long_only", "alpha_tstat"] if "long_only" in alpha.index else np.nan,
            "turnover": stats.loc["long_only", "avg_turnover"],
            "rebalances": int(stats.loc["long_only", "periods"]),
        })
    return pd.DataFrame(rows)


def save_artifacts(run: ResearchRun, output_dir: Path = RESEARCH_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = {key: output_dir / name for key, name in ARTIFACT_FILES.items()}
    path["eda"].write_text(json.dumps(run.eda, indent=2))
    run.comparison.to_csv(path["model_comparison"], index=False)
    run.fold_metrics.to_csv(path["fold_metrics"], index=False)
    run.predictions.to_parquet(path["oos_predictions"], index=False)
    run.calibration.to_csv(path["calibration"], index=False)
    run.importance.to_csv(path["feature_importance"], index=False)
    run.signals.to_csv(path["signal_ic"], index=False)
    run.backtest.returns.to_parquet(path["backtest_returns"], index=False)
    run.backtest.summary.to_csv(path["backtest_summary"], index=False)
    run.backtest.quantile_returns.to_csv(path["quantile_returns"], index=False)
    path["backtest_config"].write_text(json.dumps(config_dict(run.backtest.config), indent=2))
    path["spec"].write_text(json.dumps(run.spec.as_dict(), indent=2))
    for key, frame in [("attribution", run.attribution), ("experiments", run.experiments)]:
        if frame is not None and not frame.empty:
            frame.to_csv(path[key], index=False)
        elif path[key].exists():
            path[key].unlink()
    if run.factors is not None:
        run.factors.to_parquet(path["factor_returns"], index=False)
    path["report"].write_text(run.report, encoding="utf-8")


def load_artifacts(output_dir: Path = RESEARCH_DIR) -> dict:
    """Saved research outputs keyed like ARTIFACT_FILES; missing files are None."""
    artifacts: dict = {}
    for key, name in ARTIFACT_FILES.items():
        path = output_dir / name
        if not path.exists():
            artifacts[key] = None
        elif name.endswith(".csv"):
            try:
                artifacts[key] = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                artifacts[key] = None
        elif name.endswith(".parquet"):
            artifacts[key] = pd.read_parquet(path)
        elif name.endswith(".json"):
            artifacts[key] = json.loads(path.read_text())
        else:
            artifacts[key] = path.read_text(encoding="utf-8")
    return artifacts


def _pct(value) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value * 100:.2f}%"


def _num(value, decimals: int = 3) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value:.{decimals}f}"


def _table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, divider, *body])


def _data_and_caveats(eda: dict, quality: dict, has_cash: bool) -> tuple[str, str]:
    source = quality.get("source", "unknown")
    lines = [f"- Source: **{ {'live': 'live public data (Wikipedia, Yahoo Finance, SEC EDGAR, FRED)', 'kaggle': 'Kaggle dataset', 'sample': 'synthetic random-walk sample'}.get(source, source) }**."]
    if eda.get("index_member_rows") is not None and eda["index_member_rows"] != eda["rows"]:
        lines.append(f"- Model and backtest use only stocks that were S&P 500 members on each date ({eda['index_member_rows']:,} of {eda['rows']:,} rows).")
    if eda.get("fundamentals_coverage") is not None:
        lines.append(f"- Point-in-time SEC fundamentals available on {_pct(eda['fundamentals_coverage'])} of those rows.")
    prices = quality.get("prices", {})
    if prices.get("extreme_daily_moves"):
        lines.append(f"- Data checks: {prices['extreme_daily_moves']} daily moves above 40% (kept: usually real events such as bankruptcies or spin-offs, sometimes data errors); longest gap between trading days {prices.get('longest_gap_days')} days.")
    if eda.get("model_features"):
        lines.append(f"- Model inputs ({len(eda['model_features'])}): {', '.join(eda['model_features'])}.")

    caveats = []
    survivorship = quality.get("survivorship")
    if survivorship:
        caveats.append(
            f"- **Survivorship bias, reduced but not removed.** Index membership is reconstructed from Wikipedia's change log, "
            f"and {survivorship['former_members_with_prices']} of {survivorship['former_members']} former members have price data "
            f"({_pct(survivorship['coverage'])}). The rest, often delisted after mergers or failures, are missing, which still flatters long strategies."
        )
    else:
        caveats.append("- **Survivorship bias.** The data holds today's S&P 500 members only. Stocks that were removed (often after falling) are missing, which flatters every long strategy and the benchmark.")
    if eda.get("fundamentals_coverage") is None:
        caveats.append("- **Snapshot fundamentals** are shown in briefs but excluded from the model because no history is available.")
    else:
        caveats.append("- **Fundamentals** come from XBRL tags in SEC filings, as first reported; tagging differs between companies, and ratios for unusual share structures are dropped by a plausibility check.")
    if has_cash:
        caveats.append("- Sharpe ratios for long-only and the benchmark are in excess of 3-month Treasury bills; long-short is self-financing.")
    return "\n".join(lines), "\n".join(caveats)


def _attribution_text(attribution: pd.DataFrame | None) -> str:
    if attribution is None or attribution.empty:
        return "Not enough data for a factor attribution."
    columns = {"strategy": "strategy", "alpha_annual": "alpha / year", "alpha_tstat": "alpha t"}
    for factor in FACTOR_LABELS:
        if f"beta_{factor}" in attribution.columns:
            columns[f"beta_{factor}"] = f"{factor} beta"
    columns["r_squared"] = "R²"
    table = attribution[[c for c in columns if c in attribution.columns]].rename(columns=columns)
    table["strategy"] = table["strategy"].map(STRATEGY_NAMES).fillna(table["strategy"])
    for col in table.columns[1:]:
        table[col] = table[col].map(_pct if col == "alpha / year" else (lambda v: _num(v, 2)))
    return (
        "Each strategy's returns regressed on market, size, value and momentum factors built from the same members and dates "
        "(long-only in excess of T-bills). Alpha is the part the factors don't explain; |t| above about 2 would be meaningful.\n\n"
        + _table(table)
    )


def _experiments_text(experiments: pd.DataFrame | None) -> str:
    if experiments is None or experiments.empty:
        return ""
    skipped = experiments[experiments["auc"].isna()] if "auc" in experiments else experiments
    experiments = experiments.dropna(subset=["auc"]) if "auc" in experiments else experiments.iloc[0:0]
    note = "".join(f"\n\nNot evaluated: {row.setup} ({row.note})." for row in skipped.itertuples()) if "note" in skipped else ""
    if experiments.empty:
        return "## 4. Experiments: what was tried" + note
    table = pd.DataFrame({
        "setup": experiments["setup"] + experiments["production"].map(lambda p: " (production)" if p else ""),
        "portfolio": experiments["portfolio"],
        "AUC": experiments["auc"].map(lambda v: _num(v, 3)),
        "rank IC": experiments["ic_mean"].map(lambda v: _num(v, 4)),
        "IC t": experiments["ic_tstat"].map(lambda v: _num(v, 2)),
        "long-only CAGR": experiments["long_only_cagr"].map(_pct),
        "reference CAGR": experiments["reference_cagr"].map(_pct),
        "long-short CAGR": experiments["long_short_cagr"].map(_pct),
        "long-short alpha (t)": [f"{_pct(a)} ({_num(t, 2)})" for a, t in zip(experiments["long_short_alpha"], experiments["long_short_alpha_t"])],
        "turnover": experiments["turnover"].map(lambda v: _num(v, 2)),
    })
    reference = "the S&P 500" if (experiments["reference"] == MARKET).all() else "the equal-weight benchmark"
    return (
        "## 4. Experiments: what was tried\n\n"
        f"The same model ({EXPERIMENT_MODEL.replace('_', ' ')}) under each research setup, walked forward and backtested the same way. "
        f"Reference CAGR is {reference} over the same periods. The production setup is chosen by a rule fixed in advance: "
        "the highest out-of-sample rank-IC t-stat.\n\n" + _table(table) + note
    )


def render_report(run: ResearchRun) -> str:
    eda, best = run.eda, run.comparison.iloc[0]
    comparison = pd.DataFrame(
        {
            "model": run.comparison["model"],
            "AUC (mean ± sd)": [f"{a:.3f} ± {s:.3f}" for a, s in zip(run.comparison["auc_mean"], run.comparison["auc_std"])],
            "accuracy": run.comparison["accuracy"].map(_num),
            "baseline acc.": run.comparison["baseline_accuracy"].map(_num),
            "Brier": run.comparison["brier"].map(_num),
            "IC mean": run.comparison["ic_mean"].map(lambda v: _num(v, 4)),
            "IC t-stat": run.comparison["ic_tstat"].map(lambda v: _num(v, 2)),
        }
    )
    summary = run.backtest.summary
    backtest = pd.DataFrame(
        {
            "strategy": summary["strategy"].map(STRATEGY_NAMES).fillna(summary["strategy"]),
            "total return": summary["total_return"].map(_pct),
            "CAGR": summary["cagr"].map(_pct),
            "volatility": summary["ann_volatility"].map(_pct),
            "Sharpe": summary["sharpe"].map(lambda v: _num(v, 2)),
            "max drawdown": summary["max_drawdown"].map(_pct),
            "hit rate": summary["hit_rate"].map(_pct),
            "turnover / rebalance": summary["avg_turnover"].map(lambda v: _num(v, 2)),
        }
    )
    quantiles = run.backtest.quantile_returns.assign(mean_return=lambda q: q["mean_return"].map(_pct))
    importance = run.importance.head(10).assign(
        importance_mean=lambda d: d["importance_mean"].map(lambda v: _num(v, 4)),
        importance_std=lambda d: d["importance_std"].map(lambda v: _num(v, 4)),
    )
    signals = run.signals[["signal", "ic_mean", "ic_tstat"]].assign(
        ic_mean=lambda d: d["ic_mean"].map(lambda v: _num(v, 4)), ic_tstat=lambda d: d["ic_tstat"].map(lambda v: _num(v, 2))
    )
    calibration = run.calibration.assign(
        mean_predicted=lambda d: d["mean_predicted"].map(_pct), actual_rate=lambda d: d["actual_rate"].map(_pct)
    )
    cfg = run.backtest.config
    fr = eda["forward_return_5d"]
    long_short = summary.set_index("strategy").loc["long_short"]
    verdict = (
        "The out-of-sample evidence for a predictive edge is weak: treat the rankings as a teaching example, not a trading signal."
        if best["auc_mean"] < 0.53 or not (long_short.get("ic_tstat", 0) > 2)
        else "The model shows a modest out-of-sample edge. Check it survives realistic costs, other periods and the survivorship caveat before trusting it."
    )
    data_lines, caveats = _data_and_caveats(eda, run.quality or {}, "cash" in run.backtest.returns.columns)
    attribution_text = _attribution_text(run.attribution)
    experiments_text = _experiments_text(run.experiments)
    news_line = (
        f"- Rows with at least one news article in the previous 20 sessions: {_pct(eda['share_of_rows_with_recent_news'])}."
        if eda["share_of_rows_with_recent_news"] > 0
        else "- News: no historical headlines in this source, so news is not a model input (the agent fetches current headlines on demand)."
    )
    return f"""# S&P 500 Research Report

Generated from {eda['tickers']} stocks, {eda['start']} to {eda['end']} ({eda['trading_days']} trading days, {eda['rows']:,} rows).

**Prediction target:** will a stock {run.spec.description}?

**Bottom line.** {verdict}

## 1. Data

{data_lines}
- 5-day forward returns: mean {_pct(fr['mean'])}, standard deviation {_pct(fr['std'])}, skew {fr['skew']:.2f}, excess kurtosis {fr['excess_kurtosis']:.2f} (fat tails; 1st/99th percentiles {_pct(fr['p01'])} / {_pct(fr['p99'])}).
- Prediction target: will the stock {eda['target']}? Base rate {_pct(eda['target_rate'])}; the classifier has to beat always predicting the more common answer. (Share of 5-day windows that ended up: {_pct(eda['up_rate_5d'])}.)
{news_line}

## 2. Model comparison (walk-forward)

{run.comparison.iloc[0]['folds']:.0f} expanding-window folds; each trains only on dates before its test block, with a {run.spec.horizon}-session gap so labels never overlap. Out-of-sample period: {pd.Timestamp(best['oos_start']).date()} to {pd.Timestamp(best['oos_end']).date()}.

{_table(comparison)}

Selected model: **{run.best_model}** (highest mean AUC). AUC 0.5 means no skill. IC is the average daily rank correlation between the prediction and the realised {run.spec.horizon}-day return; its t-stat uses non-overlapping dates only (|t| > 2 is the usual bar).

### Calibration of {run.best_model}

{_table(calibration)}

### What drives the model (permutation importance, last fold)

{_table(importance)}

### Single-signal benchmark

Information coefficient of each input on its own over the same period. If a single feature matches the model, the model adds little.

{_table(signals)}

## 3. Backtest

Every {cfg.holding_days} sessions, rank stocks by the out-of-sample probability. Enter {EXECUTION_LAG_DAYS} session after the signal and hold for {cfg.holding_days} sessions. Long-only buys the top {cfg.quantile:.0%}{" of each sector" if cfg.neutralize == "sector" else ""}; long-short also shorts the bottom {cfg.quantile:.0%}{" of each sector" if cfg.neutralize == "sector" else ""}; the equal-weight benchmark holds every member equally{"; the S&P 500 is SPY with dividends reinvested" if MARKET in set(summary["strategy"]) else ""}. Costs: {cfg.cost_bps:g} bps per unit of weight traded.

{_table(backtest)}

Return by prediction quintile (Q5 = highest predicted probability):

{_table(quantiles)}

### Where the returns come from

{attribution_text}

{experiments_text}

## 5. Caveats

{caveats}
- **Costs and execution** are simplified: a flat cost per unit traded, no market impact, no borrow cost for shorts.
- **Multiple testing.** Three models and {len(run.experiments) if run.experiments is not None else 1} research setups were compared; the best results are somewhat optimistic for that reason alone.
- Educational project, not investment advice.
"""
