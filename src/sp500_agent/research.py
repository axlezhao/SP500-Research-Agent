"""End-to-end research run: data summary, model comparison, backtests, robustness checks, saved artifacts and a markdown report.

Everything here runs on the research window: dates before HOLDOUT_START, with labels that would reach
into the holdout blanked. The holdout is scored separately, once, by `evaluate_holdout`.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .attribution import FACTOR_LABELS, attribute, french_attribution
from .backtest import (
    MARKET,
    BacktestConfig,
    BacktestResult,
    config_dict,
    cost_sensitivity,
    daily_return_panel,
    run_backtest,
    run_staggered,
)
from .config import (
    EXECUTION_LAG_DAYS,
    EXPERIMENT_SPECS,
    HOLDOUT_START,
    HORIZONS,
    MODEL_PATH,
    PRODUCTION_SPEC,
    RESEARCH_DIR,
    RETRAIN_EVERY,
    TRADING_DAYS_PER_YEAR,
    ResearchSpec,
)
from .model import available_features, investable, linear_model_for, train_final_model
from .stats import bootstrap_alpha, bootstrap_performance, deflated_sharpe, probability_of_backtest_overfitting
from .validation import (
    calibration_table,
    compare_models,
    feature_importance,
    ic_breakdown,
    ic_decay,
    ic_summary,
    information_coefficients,
    signal_report,
    summarise,
    walk_forward,
)

STRATEGY_NAMES = {"long_only": "Long top 20%", "long_short": "Long-short", "benchmark": "Equal-weight benchmark", MARKET: "S&P 500 (SPY)"}
# The portfolio the production backtest runs, fixed before looking at results: the top fifth, kept until a
# holding drops out of the top two-fifths, ranked on 3-day-average scores, with costs that vary by stock.
DEFAULT_PORTFOLIO = BacktestConfig(exit_quantile=0.4, smooth_days=3, cost_model="spread")
# Portfolio constructions compared on the same predictions (the first is the original weekly rebalance).
CONSTRUCTIONS = {
    "Top 20%, no buffer (original)": dict(exit_quantile=None, smooth_days=1, construction="quantile"),
    "Buffer: sell below top 40%": dict(exit_quantile=0.4, smooth_days=1, construction="quantile"),
    "Buffer + 3-day score average": dict(exit_quantile=0.4, smooth_days=3, construction="quantile"),
    "Optimiser: cost-aware, beta/size/value neutral": dict(exit_quantile=None, smooth_days=3, construction="optimizer"),
}

ARTIFACT_FILES = {
    "eda": "eda.json",
    "model_comparison": "model_comparison.csv",
    "fold_metrics": "fold_metrics.csv",
    "oos_predictions": "oos_predictions.parquet",
    "calibration": "calibration.csv",
    "feature_importance": "feature_importance.csv",
    "signal_ic": "signal_ic.csv",
    "ic_breakdown": "ic_breakdown.csv",
    "ic_decay": "ic_decay.csv",
    "backtest_returns": "backtest_returns.parquet",
    "backtest_summary": "backtest_summary.csv",
    "quantile_returns": "quantile_returns.csv",
    "backtest_config": "backtest_config.json",
    "constructions": "constructions.csv",
    "staggered": "staggered.csv",
    "cost_sensitivity": "cost_sensitivity.csv",
    "bootstrap": "bootstrap.csv",
    "robustness": "robustness.json",
    "attribution": "attribution.csv",
    "factor_returns": "factor_returns.parquet",
    "french_attribution": "french_attribution.csv",
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
    ic_breakdown: pd.DataFrame | None = None
    ic_decay: pd.DataFrame | None = None
    constructions: pd.DataFrame | None = None
    staggered: pd.DataFrame | None = None
    cost_sensitivity: pd.DataFrame | None = None
    bootstrap: pd.DataFrame | None = None
    french_attribution: pd.DataFrame | None = None
    robustness: dict = field(default_factory=dict)


def research_window(features: pd.DataFrame, holdout_start=HOLDOUT_START) -> pd.DataFrame:
    """Features before the holdout, with every label whose forward window reaches into it blanked.

    A 5-day label on the last research day is computed from holdout prices; keeping it would let the
    holdout leak into training and evaluation through the back door.
    """
    if holdout_start is None or str(holdout_start).lower() in ("", "none"):
        return features
    cut = pd.Timestamp(holdout_start)
    sessions = np.sort(features["date"].unique())
    first_holdout = int(np.searchsorted(sessions, np.datetime64(cut)))
    research = features[features["date"] < cut].copy()
    if research.empty:
        raise ValueError(f"No data before the holdout start {cut.date()}.")
    position = pd.Series(np.arange(len(sessions)), index=sessions).reindex(research["date"]).to_numpy()
    for horizon in HORIZONS:
        reaches = position + horizon + EXECUTION_LAG_DAYS >= first_holdout
        columns = [c for c in research.columns if c.endswith(f"_{horizon}d") and c.startswith(("target_", "future_return", "tradable_return"))]
        research.loc[reaches, columns] = np.nan
    return research


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
        "target_kind": spec.target,
        "target_rate": float(labelled[spec.binary_column].mean()) if spec.binary_column in labelled else float(labelled[spec.target_column].mean()),
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


def backtest_config_for(spec: ResearchSpec, base: BacktestConfig | None = None) -> BacktestConfig:
    """Hold for the prediction horizon; a sector-relative target trades sector-neutral portfolios."""
    return replace(base or DEFAULT_PORTFOLIO, holding_days=spec.horizon, neutralize="sector" if spec.relative_to == "sector" else "none")


def _sharpe_per_period(returns: pd.Series) -> float:
    returns = returns.dropna()
    return float(returns.mean() / returns.std()) if len(returns) > 2 and returns.std() > 0 else float("nan")


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
    retrain_every: int | None = RETRAIN_EVERY,
    tune: bool = True,
    holdout_start=HOLDOUT_START,
    french_factors: pd.DataFrame | None = None,
    bootstrap_reps: int = 2000,
    log=lambda message: None,
) -> ResearchRun:
    """Compare models for `spec`, backtest and stress-test the best, compare research setups, save everything."""
    research = research_window(features, holdout_start)
    eda = exploratory_summary(research, spec)
    eda["holdout_start"] = None if research is features else str(pd.Timestamp(holdout_start).date())
    eda["retrain_every"] = retrain_every
    log("  comparing models...")
    comparison, results = compare_models(research, n_splits=n_splits, max_rows=max_rows, spec=spec, retrain_every=retrain_every, tune=tune, log=log)
    best = comparison.iloc[0]["model"]
    best_result = results[best]
    predictions = best_result.predictions
    # 3-month T-bill yield (FRED, % a year) as the cash return for Sharpe ratios, when available.
    risk_free = research.groupby("date")["tbill_3m"].first() / 100 if "tbill_3m" in research.columns else None
    daily = daily_return_panel(research)
    config = backtest_config_for(spec, backtest_config)
    log("  backtesting...")
    backtest = run_backtest(predictions, config, risk_free=risk_free, index_prices=index_prices, daily_returns=daily)
    attribution, factors = attribute(research, backtest.returns, spec.horizon, spec.tradable_column)
    french = french_attribution(backtest.returns, french_factors, spec.horizon)[0] if french_factors is not None and not french_factors.empty else None

    # Portfolio constructions on the same predictions, then trials for the overfitting statistics.
    log("  portfolio constructions, staggered starts and cost sensitivity...")
    construction_rows, trials = [], {}
    for name, overrides in CONSTRUCTIONS.items():
        variant = run_backtest(predictions, replace(config, **overrides), risk_free=risk_free, index_prices=index_prices, daily_returns=daily)
        construction_rows.append(_construction_row(name, variant, production=(variant.config == config)))
        trials[f"{best} / {name}"] = variant.returns.set_index("date")["long_short"]
    constructions = pd.DataFrame(construction_rows)
    staggered = run_staggered(predictions, config if config.construction == "quantile" else replace(config, construction="quantile"), risk_free, index_prices, daily)
    sensitivity, breakeven = cost_sensitivity(backtest)
    periods_per_year = TRADING_DAYS_PER_YEAR / spec.horizon
    cash = backtest.returns["cash"] if "cash" in backtest.returns.columns else None
    bootstrap = bootstrap_performance(backtest.returns, [s for s in ["long_only", "long_short", "benchmark", MARKET] if s in backtest.returns], periods_per_year, cash, reps=bootstrap_reps)
    factor_frame = factors.drop(columns="date").set_index(backtest.returns.index) if factors is not None else None
    alpha_ci = {}
    if factor_frame is not None:
        for strategy in ["long_only", "long_short"]:
            y = backtest.returns[strategy] - (cash if strategy == "long_only" and cash is not None else 0.0)
            alpha_ci[strategy] = bootstrap_alpha(y, factor_frame, periods_per_year, reps=bootstrap_reps)

    for name, result in results.items():
        if name == best:
            continue
        other = run_backtest(result.predictions, config, risk_free=risk_free, index_prices=index_prices, daily_returns=None if config.construction == "quantile" else daily)
        trials[f"{name} / production portfolio"] = other.returns.set_index("date")["long_short"]

    experiments, experiment_trials = (None, {})
    if experiment_specs:
        log("  research-setup experiments...")
        experiments, experiment_trials = run_experiments(
            research, experiment_specs, n_splits, max_rows, backtest_config, risk_free, index_prices, production=spec, retrain_every=retrain_every, tune=tune, daily_returns=daily,
        )
        trials.update({k: v for k, v in experiment_trials.items() if k not in trials})
    robustness = _overfitting(backtest, trials, spec, n_models=len(results), n_setups=len(experiment_specs or ()), constructions=len(CONSTRUCTIONS))
    robustness["breakeven_cost"] = breakeven
    robustness["alpha_confidence"] = alpha_ci
    robustness["selection_rule"] = "Model: highest Newey-West t-stat of the out-of-sample rank IC. Setup: fixed in config.PRODUCTION_SPEC before these experiments; the rule's pick is reported alongside."
    if experiments is not None and "ic_tstat" in experiments and experiments["ic_tstat"].notna().any():
        robustness["setup_with_highest_ic_tstat"] = experiments.loc[experiments["ic_tstat"].idxmax(), "setup"]

    log("  diagnostics...")
    signals = signal_report(research, predictions, spec)
    top_signals = [s for s in signals["signal"] if s != "model_probability"][:6]
    decay = ic_decay(research, predictions, top_signals)
    validation = comparison.iloc[0].to_dict()
    chosen_params = best_result.params[-1] if best_result.params else None
    log("  training the final model on all data...")
    bundle = train_final_model(features, best, validation=validation, max_rows=max_rows, model_path=model_path, spec=spec, params=chosen_params, tune=tune)
    run = ResearchRun(
        eda=eda,
        comparison=comparison,
        fold_metrics=pd.concat([r.fold_metrics.assign(model=name) for name, r in results.items()], ignore_index=True),
        best_model=best,
        predictions=predictions,
        calibration=calibration_table(predictions),
        importance=feature_importance(best_result),
        signals=signals,
        backtest=backtest,
        bundle=bundle,
        report="",
        quality=quality,
        spec=spec,
        attribution=attribution,
        factors=factors,
        experiments=experiments,
        ic_breakdown=ic_breakdown(predictions, spec),
        ic_decay=decay,
        constructions=constructions,
        staggered=staggered,
        cost_sensitivity=sensitivity,
        bootstrap=bootstrap,
        french_attribution=french,
        robustness=robustness,
    )
    run.report = render_report(run)
    if output_dir is not None:
        save_artifacts(run, output_dir)
    return run


def _construction_row(name: str, result: BacktestResult, production: bool) -> dict:
    stats = result.summary.set_index("strategy")
    return {
        "construction": name,
        "production": production,
        "long_only_cagr": stats.loc["long_only", "cagr"],
        "long_only_sharpe": stats.loc["long_only", "sharpe"],
        "excess_vs_equal_weight": stats.loc["long_only", "cagr"] - stats.loc["benchmark", "cagr"],
        "information_ratio": stats.loc["long_only"].get("information_ratio"),
        "long_only_turnover": stats.loc["long_only", "avg_turnover"],
        "long_only_cost_drag": stats.loc["long_only", "cost_drag"],
        "long_short_cagr": stats.loc["long_short", "cagr"],
        "long_short_sharpe": stats.loc["long_short", "sharpe"],
        "long_short_turnover": stats.loc["long_short", "avg_turnover"],
        "long_short_cost_drag": stats.loc["long_short", "cost_drag"],
    }


def _overfitting(backtest: BacktestResult, trials: dict[str, pd.Series], spec: ResearchSpec, n_models: int, n_setups: int, constructions: int) -> dict:
    """Deflated Sharpe ratios for the production portfolios and the probability of backtest overfitting.

    The trials are every backtest this run produced at the production horizon (models x constructions x
    setups); the number of configurations tried overall counts toward the deflation.
    """
    same_horizon = {k: v for k, v in trials.items() if v.index.isin(backtest.returns["date"]).mean() > 0.5}
    trial_sharpes = [_sharpe_per_period(v) for v in same_horizon.values()]
    n_trials = max(len(trials), n_models + n_setups + constructions - 1)
    returns = backtest.returns
    active = returns["long_only"] - returns["benchmark"]
    out = {
        "trials_counted": n_trials,
        "long_short": deflated_sharpe(returns["long_short"], trial_sharpes, n_trials),
        "long_only_vs_equal_weight": deflated_sharpe(active, trial_sharpes, n_trials),
    }
    matrix = pd.DataFrame(same_horizon)
    if matrix.shape[1] >= 2:
        out["pbo"] = probability_of_backtest_overfitting(matrix)
    return out


def run_experiments(
    features: pd.DataFrame,
    specs: tuple[ResearchSpec, ...],
    n_splits: int,
    max_rows: int,
    base_config: BacktestConfig | None,
    risk_free: pd.Series | None,
    index_prices: pd.Series | None,
    production: ResearchSpec = PRODUCTION_SPEC,
    retrain_every: int | None = None,
    tune: bool = False,
    daily_returns: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Each research setup walked forward with the same kind of linear model, backtested and attributed.

    Returns the comparison table and each setup's long-short returns (for the overfitting statistics).
    """
    rows, trials = [], {}
    for spec in specs:
        model = linear_model_for(spec)
        try:
            result = walk_forward(features, model, n_splits, max_rows, spec, retrain_every, tune)
        except ValueError as exc:  # e.g. no sector has enough members for a sector-relative target
            rows.append({"setup": spec.label, "horizon": spec.horizon, "relative_to": spec.relative_to, "target": spec.target, "production": spec == production, "note": str(exc)})
            continue
        quality = summarise(result)
        config = backtest_config_for(spec, base_config)
        if config.construction == "optimizer":
            config = replace(config, construction="quantile")
        backtest = run_backtest(result.predictions, config, risk_free=risk_free, index_prices=index_prices)
        trials[f"{model} / {spec.label}"] = backtest.returns.set_index("date")["long_short"]
        stats = backtest.summary.set_index("strategy")
        attribution, _ = attribute(features, backtest.returns, spec.horizon, spec.tradable_column)
        alpha = attribution.set_index("strategy") if not attribution.empty else pd.DataFrame()
        rows.append({
            "setup": spec.label,
            "horizon": spec.horizon,
            "relative_to": spec.relative_to,
            "target": spec.target,
            "model": model,
            "portfolio": "sector-neutral" if config.neutralize == "sector" else "top/bottom 20% overall",
            "production": spec == production,
            "auc": quality["auc_mean"],
            "ic_mean": quality["ic_mean"],
            "ic_tstat": quality["ic_tstat"],
            "long_only_cagr": stats.loc["long_only", "cagr"],
            "long_only_sharpe": stats.loc["long_only", "sharpe"],
            "reference": "benchmark",
            "reference_cagr": stats.loc["benchmark", "cagr"],
            "reference_sharpe": stats.loc["benchmark", "sharpe"],
            "sp500_cagr": stats.loc[MARKET, "cagr"] if MARKET in stats.index else np.nan,
            "long_short_cagr": stats.loc["long_short", "cagr"],
            "long_short_sharpe": stats.loc["long_short", "sharpe"],
            "long_short_alpha": alpha.loc["long_short", "alpha_annual"] if "long_short" in alpha.index else np.nan,
            "long_short_alpha_t": alpha.loc["long_short", "alpha_tstat"] if "long_short" in alpha.index else np.nan,
            "long_only_alpha": alpha.loc["long_only", "alpha_annual"] if "long_only" in alpha.index else np.nan,
            "long_only_alpha_t": alpha.loc["long_only", "alpha_tstat"] if "long_only" in alpha.index else np.nan,
            "turnover": stats.loc["long_only", "avg_turnover"],
            "rebalances": int(stats.loc["long_only", "periods"]),
        })
    return pd.DataFrame(rows), trials


# -- holdout --------------------------------------------------------------------------------------------
def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True, cwd=Path(__file__).parent).stdout.strip()
    except Exception:
        return None


def evaluate_holdout(
    features: pd.DataFrame,
    model_name: str,
    params: dict | None,
    spec: ResearchSpec = PRODUCTION_SPEC,
    holdout_start=HOLDOUT_START,
    backtest_config: BacktestConfig | None = None,
    retrain_every: int = RETRAIN_EVERY,
    max_rows: int = 150_000,
    index_prices: pd.Series | None = None,
    log_path: Path | None = RESEARCH_DIR / "holdout_log.jsonl",
) -> dict:
    """Score the frozen production setup (model, hyperparameters, portfolio) on the holdout, once.

    The model is walked forward through the holdout exactly as in research (refit every quarter on all
    earlier data) but with nothing re-chosen. Every evaluation is appended to `log_path` with a timestamp
    and the git commit, so repeated looks at the holdout are visible; after the first, it is no longer
    untouched.
    """
    cut = pd.Timestamp(holdout_start)
    result = walk_forward(features, model_name, max_rows=max_rows, spec=spec, retrain_every=retrain_every, params=params or {}, test_start=cut)
    predictions = result.predictions
    risk_free = features.groupby("date")["tbill_3m"].first() / 100 if "tbill_3m" in features.columns else None
    config = backtest_config_for(spec, backtest_config)
    daily = daily_return_panel(features) if config.construction == "optimizer" else None
    backtest = run_backtest(predictions, config, risk_free=risk_free, index_prices=index_prices, daily_returns=daily)
    stats = backtest.summary.set_index("strategy")
    ic = ic_summary(information_coefficients(predictions, "probability"), step=spec.horizon)
    previous = 0
    if log_path is not None and log_path.exists():
        previous = sum(1 for line in log_path.read_text().splitlines() if line.strip())
    record = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "holdout_start": str(cut.date()),
        "holdout_end": str(predictions["date"].max().date()),
        "spec": spec.as_dict(),
        "model": model_name,
        "params": params or {},
        "portfolio": config_dict(config),
        "previous_evaluations": previous,
        **{k: v for k, v in ic.items()},
        "long_only_cagr": stats.loc["long_only", "cagr"],
        "equal_weight_cagr": stats.loc["benchmark", "cagr"],
        "long_short_cagr": stats.loc["long_short", "cagr"],
        "long_short_sharpe": stats.loc["long_short", "sharpe"],
        "rebalances": int(stats.loc["long_only", "periods"]),
    }
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    return record


# -- artifacts ------------------------------------------------------------------------------------------
def save_artifacts(run: ResearchRun, output_dir: Path = RESEARCH_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = {key: output_dir / name for key, name in ARTIFACT_FILES.items()}
    path["eda"].write_text(json.dumps(run.eda, indent=2, default=str))
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
    path["robustness"].write_text(json.dumps(_json_ready(run.robustness), indent=2))
    for key in ["attribution", "experiments", "ic_breakdown", "ic_decay", "constructions", "staggered", "cost_sensitivity", "bootstrap", "french_attribution"]:
        frame = getattr(run, key)
        if frame is not None and not frame.empty:
            frame.to_csv(path[key], index=False)
        elif path[key].exists():
            path[key].unlink()
    if run.factors is not None:
        run.factors.to_parquet(path["factor_returns"], index=False)
    path["report"].write_text(run.report, encoding="utf-8")


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


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


# -- report ---------------------------------------------------------------------------------------------
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
    if eda.get("holdout_start"):
        lines.append(f"- **Holdout:** everything here uses data before {eda['holdout_start']}; labels reaching into the holdout are blanked. The holdout is scored separately with `run_pipeline.py --evaluate-holdout`.")
    if eda.get("index_member_rows") is not None and eda["index_member_rows"] != eda["rows"]:
        lines.append(f"- Model and backtest use only stocks that were S&P 500 members on each date ({eda['index_member_rows']:,} of {eda['rows']:,} rows).")
    if eda.get("fundamentals_coverage") is not None:
        lines.append(f"- Point-in-time SEC fundamentals available on {_pct(eda['fundamentals_coverage'])} of those rows.")
    prices = quality.get("prices", {})
    if prices.get("extreme_daily_moves"):
        lines.append(f"- Data checks: {prices['extreme_daily_moves']} daily moves above 40% (kept: usually real events such as bankruptcies or spin-offs, sometimes data errors); longest gap between trading days {prices.get('longest_gap_days')} days.")
    if quality.get("delistings"):
        d = quality["delistings"]
        lines.append(f"- {d['stocks_ending_early']} stocks stop trading before the data ends; they stay in the backtest until their last price ({d['assumed_performance_delistings']} with a -30% delisting return for apparent failures).")
    if eda.get("model_features"):
        lines.append(f"- Model inputs ({len(eda['model_features'])}), each ranked within its date and mapped to a normal score: {', '.join(eda['model_features'])}.")

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


def _attribution_text(attribution: pd.DataFrame | None, labels: dict = FACTOR_LABELS) -> str:
    if attribution is None or attribution.empty:
        return "Not enough data for a factor attribution."
    columns = {"strategy": "strategy", "alpha_annual": "alpha / year", "alpha_tstat": "alpha t"}
    for factor in labels:
        if f"beta_{factor}" in attribution.columns:
            columns[f"beta_{factor}"] = f"{factor} beta"
    columns["r_squared"] = "R²"
    table = attribution[[c for c in columns if c in attribution.columns]].rename(columns=columns)
    table["strategy"] = table["strategy"].map(STRATEGY_NAMES).fillna(table["strategy"])
    for col in table.columns[1:]:
        table[col] = table[col].map(_pct if col == "alpha / year" else (lambda v: _num(v, 2)))
    return _table(table)


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
        "model": experiments["model"],
        "portfolio": experiments["portfolio"],
        "rank IC": experiments["ic_mean"].map(lambda v: _num(v, 4)),
        "IC t (NW)": experiments["ic_tstat"].map(lambda v: _num(v, 2)),
        "long-only CAGR": experiments["long_only_cagr"].map(_pct),
        "equal-weight CAGR": experiments["reference_cagr"].map(_pct),
        "long-short CAGR": experiments["long_short_cagr"].map(_pct),
        "long-short alpha (t)": [f"{_pct(a)} ({_num(t, 2)})" for a, t in zip(experiments["long_short_alpha"], experiments["long_short_alpha_t"])],
        "turnover": experiments["turnover"].map(lambda v: _num(v, 2)),
    })
    return (
        "## 4. Experiments: what was tried\n\n"
        "Each research setup with the matching linear model (logistic regression for yes/no targets, ridge for ranked ones), "
        "walked forward and backtested the same way. The reference is the equal-weight universe, the like-for-like comparison for "
        "an equal-weighted portfolio of members.\n\n" + _table(table) + note
    )


def _robustness_text(run: ResearchRun) -> str:
    r = run.robustness or {}
    lines = ["## 5. Is it luck?\n"]
    trials = r.get("trials_counted")
    for key, label in [("long_short", "Long-short"), ("long_only_vs_equal_weight", "Long-only minus equal-weight")]:
        d = r.get(key) or {}
        if d:
            lines.append(
                f"- **{label}:** per-period Sharpe {_num(d.get('sharpe_per_period'), 3)}; probabilistic Sharpe (chance the true Sharpe is above 0) "
                f"{_pct(d.get('probabilistic_sharpe'))}; **deflated Sharpe {_pct(d.get('deflated_sharpe'))}** after {trials} configurations "
                f"(the best of {trials} unskilled tries would be expected to reach {_num(d.get('expected_max_sharpe_per_period'), 3)}). Above 95% would be convincing."
            )
    pbo = r.get("pbo") or {}
    if pbo:
        lines.append(f"- **Probability of backtest overfitting: {_pct(pbo.get('pbo'))}** across {pbo.get('strategies')} backtests at this horizon ({pbo.get('combinations')} in-sample/out-of-sample splits). Near 50% or above means picking the best backtest is no better than picking at random.")
    for strategy, ci in (r.get("alpha_confidence") or {}).items():
        if ci:
            lines.append(f"- {STRATEGY_NAMES.get(strategy, strategy)} alpha, 95% bootstrap interval: {_pct(ci.get('alpha_low'))} to {_pct(ci.get('alpha_high'))} a year.")
    if run.bootstrap is not None and not run.bootstrap.empty:
        table = pd.DataFrame({
            "strategy": run.bootstrap["strategy"].map(STRATEGY_NAMES).fillna(run.bootstrap["strategy"]),
            "CAGR 95% interval": [f"{_pct(a)} to {_pct(b)}" for a, b in zip(run.bootstrap["cagr_low"], run.bootstrap["cagr_high"])],
            "Sharpe 95% interval": [f"{_num(a, 2)} to {_num(b, 2)}" for a, b in zip(run.bootstrap["sharpe_low"], run.bootstrap["sharpe_high"])],
            "P(Sharpe ≤ 0)": run.bootstrap["prob_sharpe_not_positive"].map(_pct),
        })
        lines.append("\nStationary-bootstrap intervals (blocks of consecutive periods resampled, so volatility clusters are kept):\n\n" + _table(table))
    if r.get("setup_with_highest_ic_tstat"):
        lines.append(f"\nThe selection rule (highest IC t-stat) would pick: **{r['setup_with_highest_ic_tstat']}**; production is **{run.spec.label}**.")
    return "\n".join(lines)


def render_report(run: ResearchRun) -> str:
    eda, best = run.eda, run.comparison.iloc[0]
    comparison = pd.DataFrame(
        {
            "model": run.comparison["model"],
            "rank IC": run.comparison["ic_mean"].map(lambda v: _num(v, 4)),
            "IC t (Newey-West)": run.comparison["ic_tstat"].map(lambda v: _num(v, 2)),
            "IC t (non-overlapping)": run.comparison["ic_tstat_nonoverlap"].map(lambda v: _num(v, 2)),
            "AUC (mean ± sd)": [f"{a:.3f} ± {s:.3f}" for a, s in zip(run.comparison["auc_mean"], run.comparison["auc_std"])],
            "Brier": run.comparison["brier"].map(_num),
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
            "cost drag / year": summary["cost_drag"].map(_pct),
            "turnover / rebalance": summary["avg_turnover"].map(lambda v: _num(v, 2)),
        }
    )
    quantiles = run.backtest.quantile_returns.assign(mean_return=lambda q: q["mean_return"].map(_pct))
    importance = run.importance.head(10).assign(
        importance_mean=lambda d: d["importance_mean"].map(lambda v: _num(v, 4)),
        importance_std=lambda d: d["importance_std"].map(lambda v: _num(v, 4)),
    )
    signals = run.signals[["signal", "ic_mean", "ic_tstat"]].head(15).assign(
        ic_mean=lambda d: d["ic_mean"].map(lambda v: _num(v, 4)), ic_tstat=lambda d: d["ic_tstat"].map(lambda v: _num(v, 2))
    )
    calibration = run.calibration.assign(
        mean_predicted=lambda d: d["mean_predicted"].map(_pct), actual_rate=lambda d: d["actual_rate"].map(_pct)
    )
    cfg = run.backtest.config
    fr = eda["forward_return_5d"]
    stats = summary.set_index("strategy")
    deflated = ((run.robustness or {}).get("long_short") or {}).get("deflated_sharpe")
    edge = best["ic_tstat"] > 2 and stats.loc["long_short", "cagr"] > 0 and (deflated or 0) > 0.95
    verdict = (
        "The model shows an out-of-sample edge that survives costs and the deflation for the number of configurations tried. Check it on the holdout before trusting it."
        if edge
        else "The out-of-sample evidence for a tradable edge is weak: treat the rankings as a teaching example, not a trading signal."
    )
    data_lines, caveats = _data_and_caveats(eda, run.quality or {}, "cash" in run.backtest.returns.columns)
    news_line = (
        f"- Rows with at least one news article in the previous 20 sessions: {_pct(eda['share_of_rows_with_recent_news'])}."
        if eda["share_of_rows_with_recent_news"] > 0
        else "- News: no historical headlines in this source, so news is not a model input (the agent fetches current headlines on demand)."
    )
    target_line = (
        f"- Prediction target: the stock's percentile rank among {'all members' if run.spec.relative_to == 'market' else 'its sector'} by volatility-scaled {run.spec.horizon}-day return (mean 0.5 every day). "
        "The model's score is its expected percentile: the probability of beating a randomly chosen peer."
        if run.spec.is_rank
        else f"- Prediction target: will the stock {eda['target']}? Base rate {_pct(eda['target_rate'])}."
    )
    sections = [f"""# S&P 500 Research Report

Generated from {eda['tickers']} stocks, {eda['start']} to {eda['end']} ({eda['trading_days']} trading days, {eda['rows']:,} rows).

**Prediction target:** the probability that a stock will {run.spec.description}.

**Bottom line.** {verdict}

## 1. Data

{data_lines}
- 5-day forward returns: mean {_pct(fr['mean'])}, standard deviation {_pct(fr['std'])}, skew {fr['skew']:.2f}, excess kurtosis {fr['excess_kurtosis']:.2f} (fat tails; 1st/99th percentiles {_pct(fr['p01'])} / {_pct(fr['p99'])}).
{target_line}
{news_line}

## 2. Model comparison (walk-forward)

The model is refit every {eda.get('retrain_every') or 'fold'} sessions on all earlier data ({best['folds']:.0f} refits), with hyperparameters chosen inside each training window and a {run.spec.horizon}-session gap so labels never overlap. Training uses one date per {run.spec.horizon}-session window, so no two training labels share a forward window. Out-of-sample period: {pd.Timestamp(best['oos_start']).date()} to {pd.Timestamp(best['oos_end']).date()}.

{_table(comparison)}

Selected model: **{run.best_model}** (highest Newey-West t-stat of the rank IC; the portfolio trades the ranking, so IC is the measure that matters). The composite is a fixed-sign average of documented anomalies: a fitted model has to beat it to justify itself. IC is the average daily rank correlation between the score and the realised {run.spec.horizon}-day return; the Newey-West t-stat uses every date and corrects for overlapping windows (|t| > 2 is the usual bar). AUC scores the split at the median.

### Calibration of {run.best_model}

{_table(calibration)}

### What drives the model (permutation importance, last refit)

{_table(importance)}

### Single-signal benchmark

Information coefficient of each input on its own over the same period. If a single feature matches the model, the model adds little.

{_table(signals)}
"""]
    if run.ic_breakdown is not None and not run.ic_breakdown.empty:
        breakdown = run.ic_breakdown.assign(ic_mean=lambda d: d["ic_mean"].map(lambda v: _num(v, 4)), ic_tstat=lambda d: d["ic_tstat"].map(lambda v: _num(v, 2)))[["slice", "group", "ic_mean", "ic_tstat", "ic_days"]]
        sections.append("### Where the ranking power lives\n\nThe model's rank IC by year, within each sector, and within each third of the universe by size.\n\n" + _table(breakdown) + "\n")
    if run.ic_decay is not None and not run.ic_decay.empty:
        decay = run.ic_decay.pivot(index="signal", columns="horizon", values="ic_mean")
        decay = decay.map(lambda v: _num(v, 4)).reset_index()
        decay.columns = ["signal", *[f"{h}d" for h in decay.columns[1:]]]
        sections.append("### How long the signals last (IC decay)\n\nRank IC against forward returns of 1 to 63 sessions. A signal that fades within days can't pay for frequent trading.\n\n" + _table(decay) + "\n")

    excess = stats.loc["long_only"].get("excess_cagr_vs_equal_weight")
    buffer = f", kept until it drops out of the top {cfg.exit_quantile:.0%}" if cfg.exit_quantile else ""
    smoothing = f" Scores are averaged over each stock's last {cfg.smooth_days} sessions." if cfg.smooth_days > 1 else ""
    costs = (
        f"Costs: {cfg.cost_bps:g} bps per unit of weight traded for the median stock, scaled per stock by its Corwin-Schultz spread estimate (from daily highs and lows) relative to the median"
        if cfg.cost_model == "spread"
        else f"Costs: {cfg.cost_bps:g} bps per unit of weight traded"
    ) + (f", plus {cfg.borrow_bps:g} bps a year to borrow shorts." if cfg.borrow_bps else ".")
    construction = "a cost-aware optimiser" if cfg.construction == "optimizer" else f"the top {cfg.quantile:.0%}{' of each sector' if cfg.neutralize == 'sector' else ''}{buffer}"
    sections.append(f"""## 3. Backtest

Every {cfg.holding_days} sessions, rank stocks by the out-of-sample score. Enter {EXECUTION_LAG_DAYS} session after the signal and hold for {cfg.holding_days} sessions. Long-only holds {construction}; long-short also shorts the bottom of the ranking; the equal-weight benchmark holds every member equally{"; the S&P 500 is SPY with dividends reinvested" if MARKET in set(summary["strategy"]) else ""}.{smoothing} {costs} Stocks that stop trading stay in until their last price.

{_table(backtest)}

Long-only versus the equal-weight universe (the like-for-like benchmark for an equal-weighted portfolio of members): {_pct(excess)} a year, information ratio {_num(stats.loc['long_only'].get('information_ratio'), 2)}.

Return by prediction quintile (Q5 = highest score):

{_table(quantiles)}
""")
    if run.constructions is not None and not run.constructions.empty:
        c = run.constructions
        table = pd.DataFrame({
            "construction": c["construction"] + c["production"].map(lambda p: " (production)" if p else ""),
            "long-only CAGR": c["long_only_cagr"].map(_pct),
            "vs. equal-weight": c["excess_vs_equal_weight"].map(_pct),
            "long-only cost drag": c["long_only_cost_drag"].map(_pct),
            "long-only turnover": c["long_only_turnover"].map(lambda v: _num(v, 2)),
            "long-short CAGR": c["long_short_cagr"].map(_pct),
            "long-short Sharpe": c["long_short_sharpe"].map(lambda v: _num(v, 2)),
            "long-short cost drag": c["long_short_cost_drag"].map(_pct),
        })
        sections.append("### Portfolio construction: where the costs go\n\nThe same predictions traded four ways. The buffer and the smoothing cut turnover; the optimiser trades only when the expected gain beats the cost, and its long-short is neutral to market beta, size and value.\n\n" + _table(table) + "\n")
    if run.staggered is not None and not run.staggered.empty:
        s = run.staggered
        table = pd.DataFrame({
            "strategy": s["strategy"].map(STRATEGY_NAMES).fillna(s["strategy"]),
            "CAGR mean (min to max)": [f"{_pct(m)} ({_pct(lo)} to {_pct(hi)})" for m, lo, hi in zip(s["cagr_mean"], s["cagr_min"], s["cagr_max"])],
            "Sharpe mean (min to max)": [f"{_num(m, 2)} ({_num(lo, 2)} to {_num(hi, 2)})" for m, lo, hi in zip(s["sharpe_mean"], s["sharpe_min"], s["sharpe_max"])],
        })
        sections.append(f"### Does the rebalance day matter?\n\nThe same backtest started on each of the {int(s['offsets'].max())} sessions of the holding period. The spread is how much of any single backtest is calendar luck.\n\n" + _table(table) + "\n")
    if run.cost_sensitivity is not None and not run.cost_sensitivity.empty:
        cs = run.cost_sensitivity.pivot(index="strategy", columns="cost_bps", values="cagr").map(_pct).reset_index()
        cs.columns = ["strategy", *[f"{c:g} bps" for c in cs.columns[1:]]]
        cs["strategy"] = cs["strategy"].map(STRATEGY_NAMES).fillna(cs["strategy"])
        be = (run.robustness or {}).get("breakeven_cost") or {}
        be_line = ""
        if be:
            parts = []
            if "long_only_vs_equal_weight_bps" in be:
                parts.append(f"long-only beats equal-weight only below {_num(be['long_only_vs_equal_weight_bps'], 1)} bps per unit traded")
            if "long_short_bps" in be:
                parts.append(f"long-short breaks even at {_num(be['long_short_bps'], 1)} bps")
            be_line = "\n\nBreak-even costs: " + "; ".join(parts) + "."
        sections.append("### Net CAGR at different flat costs\n\n" + _table(cs) + be_line + "\n")
    sections.append("### Where the returns come from\n\nEach strategy's returns regressed on market, size, value and momentum factors built from the same members and dates (long-only in excess of T-bills). Alpha is the part the factors don't explain; t-stats use Newey-West standard errors, and |t| above about 2 would be meaningful.\n\n" + _attribution_text(run.attribution) + "\n")
    if run.french_attribution is not None and not run.french_attribution.empty:
        from .sources.french import FACTOR_LABELS as FRENCH_LABELS

        sections.append("Cross-check on Kenneth French's published factors (whole US market; adds profitability, investment and short-term reversal):\n\n" + _attribution_text(run.french_attribution, FRENCH_LABELS) + "\n")
    experiments = _experiments_text(run.experiments)
    if experiments:
        sections.append(experiments + "\n")
    sections.append(_robustness_text(run) + "\n")
    sections.append(f"""## 6. Caveats

{caveats}
- **Costs** are assumed, not measured: the level is the `--cost-bps` setting, and only the relative cost of each stock comes from spread estimates (whose levels overstate large-cap spreads); there is no market-impact model, which matters at scale.
- **Multiple testing.** {len(run.comparison)} models, {len(CONSTRUCTIONS)} portfolio constructions and {len(run.experiments) if run.experiments is not None else 1} research setups were compared; section 5 deflates for that, but only the holdout gives a clean read.
- Educational project, not investment advice.
""")
    return "\n".join(sections)
