from __future__ import annotations

import argparse

from sp500_agent.backtest import BacktestConfig
from sp500_agent.config import FEATURES_PATH, MODEL_PATH, PROJECT_ROOT, RESEARCH_DIR, SAMPLE_DIR, load_environment
from sp500_agent.features import build_research_features, save_features, save_news
from sp500_agent.ingest import DEFAULT_START, load_files, load_live, save_dataset_extras
from sp500_agent.kaggle_loader import download_kaggle_dataset, import_from_zip
from sp500_agent.research import run_research
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
    parser.add_argument("--folds", type=int, default=5, help="Walk-forward folds (default 5).")
    parser.add_argument("--cost-bps", type=float, default=10.0, help="Backtest trading cost per unit of weight traded.")
    parser.add_argument("--quantile", type=float, default=0.2, help="Share of stocks held long (and short) in the backtest.")
    parser.add_argument("--max-rows", type=int, default=150_000, help="Cap on training rows per fit, for speed.")
    args = parser.parse_args()
    load_environment()

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
        dataset = load_live(start=args.start, limit=args.limit, refresh=args.refresh)
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
    )
    save_features(features)
    save_news(dataset.news)
    save_dataset_extras(dataset)
    print(f"Saved features to {FEATURES_PATH.relative_to(PROJECT_ROOT)} ({len(features):,} rows)")

    print(f"Running {args.folds}-fold walk-forward comparison and backtest...")
    run = run_research(
        features,
        n_splits=args.folds,
        backtest_config=BacktestConfig(cost_bps=args.cost_bps, quantile=args.quantile),
        max_rows=args.max_rows,
        quality=dataset.quality,
    )
    print(f"Saved model ({run.best_model}) to {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved research artifacts to {RESEARCH_DIR.relative_to(PROJECT_ROOT)}/")

    columns = ["model", "auc_mean", "auc_std", "accuracy", "baseline_accuracy", "ic_mean", "ic_tstat"]
    print("\nWalk-forward model comparison:\n")
    print(run.comparison[columns].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    columns = ["strategy", "total_return", "cagr", "sharpe", "max_drawdown", "avg_turnover"]
    print("\nBacktest (out-of-sample, after costs):\n")
    print(run.backtest.summary[columns].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nFull write-up: {(RESEARCH_DIR / 'research_report.md').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
