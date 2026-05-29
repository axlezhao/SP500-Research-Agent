from __future__ import annotations

import argparse

from src.sp500_agent.config import PROCESSED_DIR
from src.sp500_agent.data_loader import load_fundamentals, load_news, load_prices
from src.sp500_agent.features import build_research_features
from src.sp500_agent.kaggle_loader import download_kaggle_dataset, list_raw_csvs
from src.sp500_agent.model import train_return_direction_model
from src.sp500_agent.sample_data import make_sample_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-sample", action="store_true")
    parser.add_argument("--download-kaggle", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--from-zip", help="Import CSV files from a browser-downloaded Kaggle ZIP before running.")
    args = parser.parse_args()

    if args.from_zip:
        from src.sp500_agent.kaggle_loader import import_from_zip

        files = import_from_zip(args.from_zip, overwrite=args.overwrite)
        print(f"Imported {len(files)} CSV files from ZIP into data/raw.")
    if args.download_kaggle:
        files = download_kaggle_dataset(overwrite=args.overwrite)
        print(f"Downloaded/copied {len(files)} Kaggle CSV files into data/raw.")
    if args.make_sample:
        make_sample_data()

    raw_files = list_raw_csvs()
    print("CSV files available in data/raw:")
    for path in raw_files:
        print(f"- {path.name}")

    prices = load_prices()
    fundamentals = load_fundamentals()
    news = load_news()
    features = build_research_features(prices, fundamentals, news)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(PROCESSED_DIR / "research_features.csv", index=False)
    _, report = train_return_direction_model(features)
    print("Saved features to data/processed/research_features.csv")
    print("Saved model to models/return_direction_model.joblib")
    print("\nModel evaluation:\n")
    print(report)


if __name__ == "__main__":
    main()

