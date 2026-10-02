from __future__ import annotations

import argparse
from collections import Counter

from sp500_agent.backtest import BacktestConfig
from sp500_agent.config import FEATURES_PATH, MODEL_PATH, PROJECT_ROOT, RAW_DIR, RESEARCH_DIR
from sp500_agent.data_loader import load_fundamentals, load_news, load_prices
from sp500_agent.features import build_research_features, save_features, save_news
from sp500_agent.kaggle_loader import download_kaggle_dataset, import_from_zip, list_raw_csvs
from sp500_agent.research import run_research
from sp500_agent.sample_data import make_sample_data


def print_raw_summary() -> None:
    raw_files = list_raw_csvs()
    print(f"{len(raw_files)} CSV files available under {RAW_DIR.relative_to(PROJECT_ROOT)}:")
    for folder, count in sorted(Counter(path.parent for path in raw_files).items()):
        names = sorted(path.name for path in raw_files if path.parent == folder)
        shown = ", ".join(names[:5]) + (f", ... and {count - 5} more" if count > 5 else "")
        print(f"- {folder.relative_to(RAW_DIR) if folder != RAW_DIR else '.'}/: {shown}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-sample", action="store_true")
    parser.add_argument("--download-kaggle", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--from-zip", help="Import CSV files from a browser-downloaded Kaggle ZIP before running.")
    parser.add_argument("--folds", type=int, default=5, help="Walk-forward folds (default 5).")
    parser.add_argument("--cost-bps", type=float, default=10.0, help="Backtest trading cost per unit of weight traded.")
    parser.add_argument("--quantile", type=float, default=0.2, help="Share of stocks held long (and short) in the backtest.")
    parser.add_argument("--max-rows", type=int, default=150_000, help="Cap on training rows per fit, for speed.")
    args = parser.parse_args()

    if args.from_zip:
        files = import_from_zip(args.from_zip, overwrite=args.overwrite)
        print(f"Imported {len(files)} CSV files from ZIP into data/raw.")
    if args.download_kaggle:
        files = download_kaggle_dataset(overwrite=args.overwrite)
        print(f"Downloaded/copied {len(files)} Kaggle CSV files into data/raw.")
    if args.make_sample:
        make_sample_data()

    print_raw_summary()

    prices = load_prices()
    fundamentals = load_fundamentals()
    news = load_news()
    features = build_research_features(prices, fundamentals, news)
    save_features(features)
    save_news(news)
    print(f"Saved features to {FEATURES_PATH.relative_to(PROJECT_ROOT)} ({len(features):,} rows)")

    print(f"Running {args.folds}-fold walk-forward comparison and backtest...")
    run = run_research(
        features,
        n_splits=args.folds,
        backtest_config=BacktestConfig(cost_bps=args.cost_bps, quantile=args.quantile),
        max_rows=args.max_rows,
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
