from __future__ import annotations

import argparse
import json
import time

from sp500_agent.backtest import BacktestConfig
from sp500_agent.config import EXPERIMENT_SPECS, FEATURES_PATH, HOLDOUT_START, MODEL_PATH, PRODUCTION_SPEC, PROJECT_ROOT, RESEARCH_DIR, RETRAIN_EVERY, SAMPLE_DIR, load_environment
from sp500_agent.features import build_research_features, load_features, save_features, save_news
from sp500_agent.forward import evaluate_signals, record_signals
from sp500_agent.ingest import DEFAULT_START, load_files, load_live, save_dataset_extras
from sp500_agent.kaggle_loader import download_kaggle_dataset, import_from_zip
from sp500_agent.model import load_model, score_latest
from sp500_agent.research import DEFAULT_PORTFOLIO, evaluate_holdout, run_research
from sp500_agent.sample_data import make_sample_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Build features, compare models walk-forward, backtest, and save the research.")
    parser.add_argument("--source", choices=["live", "kaggle", "sample"], help="Data source (default: live; implied by the flags below).")
    parser.add_argument("--make-sample", action="store_true", help="Generate and use the synthetic random-walk sample.")
    parser.add_argument("--download-kaggle", action="store_true", help="Download the Kaggle dataset and use it.")
    parser.add_argument("--from-zip", help="Import a browser-downloaded Kaggle ZIP and use it.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite previously imported Kaggle files.")
    parser.add_argument("--start", default=DEFAULT_START, help=f"Live data: first date to download (default {DEFAULT_START}).")
    parser.add_argument("--limit", type=int, help="Live data: only the first N current members, for a quick run.")
    parser.add_argument("--refresh", action="store_true", help="Live data: ignore the download cache.")
    parser.add_argument("--insiders", action="store_true", help="Live data: also download SEC insider-trading data sets (large; cached for 30 days).")
    parser.add_argument("--folds", type=int, default=5, help="Walk-forward folds when --retrain-every is 0 (default 5).")
    parser.add_argument("--retrain-every", type=int, default=RETRAIN_EVERY, help=f"Refit the model every N sessions out of sample (default {RETRAIN_EVERY}; 0 = use --folds).")
    parser.add_argument("--no-tune", action="store_true", help="Skip the hyperparameter search inside each training window (faster).")
    parser.add_argument("--holdout-start", default=HOLDOUT_START, help=f"First date of the untouched holdout (default {HOLDOUT_START}; 'none' to research on everything).")
    parser.add_argument("--evaluate-holdout", action="store_true", help="Score the saved production model on the holdout, log the evaluation, and stop.")
    parser.add_argument("--cost-model", choices=["spread", "flat"], default=DEFAULT_PORTFOLIO.cost_model, help="Trading costs: --cost-bps for the median stock, scaled per stock by its estimated bid-ask spread; or flat --cost-bps for all.")
    parser.add_argument("--cost-bps", type=float, default=10.0, help="Flat cost per unit of weight traded (and the fallback without highs and lows).")
    parser.add_argument("--borrow-bps", type=float, default=0.0, help="Annual fee for borrowing shorted stock, in bps.")
    parser.add_argument("--quantile", type=float, default=0.2, help="Share of stocks held long (and short) in the backtest.")
    parser.add_argument("--exit-quantile", type=float, default=DEFAULT_PORTFOLIO.exit_quantile, help="Buffer: keep a holding until it leaves this top share (default 0.4; equal to --quantile for none).")
    parser.add_argument("--smooth-days", type=int, default=DEFAULT_PORTFOLIO.smooth_days, help="Rank on each stock's score averaged over this many sessions (default 3).")
    parser.add_argument("--construction", choices=["quantile", "optimizer"], default="quantile", help="Production portfolio: quantile ranks or the cost-aware optimiser.")
    parser.add_argument("--max-rows", type=int, default=150_000, help="Cap on training rows per fit, for speed.")
    parser.add_argument("--bootstrap-reps", type=int, default=2000, help="Resamples for bootstrap confidence intervals.")
    parser.add_argument("--skip-experiments", action="store_true", help="Don't compare the alternative research setups.")
    args = parser.parse_args()
    load_environment()

    config = BacktestConfig(
        cost_bps=args.cost_bps, quantile=args.quantile, exit_quantile=args.exit_quantile, smooth_days=args.smooth_days,
        construction=args.construction, cost_model=args.cost_model, borrow_bps=args.borrow_bps,
    )
    if args.evaluate_holdout:
        features = load_features()
        bundle = load_model()
        print(f"Evaluating {bundle['model_name']} {bundle.get('params') or ''} on the holdout from {args.holdout_start}...")
        record = evaluate_holdout(features, bundle["model_name"], bundle.get("params"), PRODUCTION_SPEC, args.holdout_start, config, args.retrain_every or RETRAIN_EVERY, args.max_rows)
        if record["previous_evaluations"]:
            print(f"Note: the holdout had already been evaluated {record['previous_evaluations']} time(s); it is no longer untouched.")
        print(json.dumps(record, indent=2, default=str))
        return

    source = args.source or ("sample" if args.make_sample else "kaggle" if (args.download_kaggle or args.from_zip) else "live")
    if args.from_zip:
        files = import_from_zip(args.from_zip, overwrite=args.overwrite)
        print(f"Imported {len(files)} CSV files from ZIP into data/raw.")
    if args.download_kaggle:
        files = download_kaggle_dataset(overwrite=args.overwrite)
        print(f"Downloaded/copied {len(files)} Kaggle CSV files into data/raw.")
    if source == "sample" and (args.make_sample or not (SAMPLE_DIR / "sample_prices.csv").exists()):
        make_sample_data(raw_dir=SAMPLE_DIR)

    print(f"Loading data (source: {source})...")
    if source == "live":
        dataset = load_live(start=args.start, limit=args.limit, refresh=args.refresh, insiders=args.insiders)
    else:
        dataset = load_files(source, raw_dir=SAMPLE_DIR) if source == "sample" else load_files(source)
    prices = dataset.quality["prices"]
    print(f"  prices: {prices['tickers']} tickers, {prices['start']} to {prices['end']}")
    if "survivorship" in dataset.quality:
        s = dataset.quality["survivorship"]
        print(f"  former index members with prices: {s['former_members_with_prices']} of {s['former_members']}")
    if "fundamentals" in dataset.quality:
        f = dataset.quality["fundamentals"]
        print(f"  SEC fundamentals: {f['tickers_with_data']} tickers ({f['failed']} failed)")

    print("Building features...")
    features = build_research_features(
        dataset.prices, dataset.companies, dataset.news,
        pit_fundamentals=dataset.fundamentals, macro=dataset.macro, membership=dataset.membership,
        eps=dataset.eps, earnings_dates=dataset.earnings_dates, insiders=dataset.insiders, delisting_returns=dataset.delisting_returns,
    )
    save_features(features)
    save_news(dataset.news)
    save_dataset_extras(dataset)
    print(f"Saved features to {FEATURES_PATH.relative_to(PROJECT_ROOT)} ({len(features):,} rows)")

    print(f"Target: the probability that a stock will {PRODUCTION_SPEC.description}.")
    print(f"Running the walk-forward comparison, backtests{'' if args.skip_experiments else ', research-setup experiments'} and robustness checks...")
    run = run_research(
        features,
        n_splits=args.folds,
        backtest_config=config,
        max_rows=args.max_rows,
        quality=dataset.quality,
        index_prices=dataset.index_prices,
        experiment_specs=None if args.skip_experiments else EXPERIMENT_SPECS,
        retrain_every=args.retrain_every or None,
        tune=not args.no_tune,
        holdout_start=args.holdout_start,
        french_factors=dataset.factors,
        bootstrap_reps=args.bootstrap_reps,
        log=lambda message, start=time.monotonic(): print(f"{message} [{(time.monotonic() - start) / 60:.1f} min]"),
    )
    print(f"Saved model ({run.best_model}) to {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved research artifacts to {RESEARCH_DIR.relative_to(PROJECT_ROOT)}/")

    scored = score_latest(features, run.bundle)
    record_signals(scored, run.bundle)
    forward = evaluate_signals(features, PRODUCTION_SPEC)
    if not forward.empty:
        print(f"\nForward test: {len(forward)} recorded rankings with known outcomes, mean IC {forward['ic'].mean():.4f}, mean top-minus-bottom {forward['top_minus_bottom'].mean():.4%}.")

    columns = ["model", "ic_mean", "ic_tstat", "ic_tstat_nonoverlap", "auc_mean"]
    print("\nWalk-forward model comparison (ranked by the Newey-West IC t-stat):\n")
    print(run.comparison[columns].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    columns = ["strategy", "total_return", "cagr", "sharpe", "max_drawdown", "avg_turnover", "cost_drag"]
    print("\nBacktest (out-of-sample, after costs):\n")
    print(run.backtest.summary[columns].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if run.constructions is not None:
        print("\nPortfolio constructions:\n")
        print(run.constructions[["construction", "long_only_cagr", "excess_vs_equal_weight", "long_only_cost_drag", "long_short_cagr", "long_short_sharpe"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if run.attribution is not None and not run.attribution.empty:
        columns = [c for c in ["strategy", "alpha_annual", "alpha_tstat", "beta_MKT", "beta_SMB", "beta_HML", "beta_MOM", "r_squared"] if c in run.attribution.columns]
        print("\nFactor attribution (Newey-West t-stats):\n")
        print(run.attribution[columns].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if run.experiments is not None and "ic_tstat" in run.experiments:
        columns = [c for c in ["setup", "model", "ic_mean", "ic_tstat", "long_only_cagr", "reference_cagr", "long_short_cagr", "long_short_alpha", "long_short_alpha_t", "turnover"] if c in run.experiments]
        print("\nResearch setups (walk-forward, after costs; reference = equal-weight universe):\n")
        print(run.experiments[columns].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    robustness = run.robustness or {}
    for key in ["long_short", "long_only_vs_equal_weight"]:
        d = robustness.get(key) or {}
        if d:
            print(f"\nDeflated Sharpe ({key}): {d.get('deflated_sharpe', float('nan')):.3f} after {robustness.get('trials_counted')} configurations")
    if robustness.get("pbo"):
        print(f"Probability of backtest overfitting: {robustness['pbo']['pbo']:.3f}")
    print(f"\nFull write-up: {(RESEARCH_DIR / 'research_report.md').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
