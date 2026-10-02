from __future__ import annotations

import argparse
from pathlib import Path

from sp500_agent.config import KAGGLE_DATASET
from sp500_agent.kaggle_loader import download_kaggle_dataset, import_from_zip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=KAGGLE_DATASET)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--from-zip",
        help="Import CSV files from a browser-downloaded Kaggle ZIP instead of using KaggleHub.",
    )
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    if args.from_zip:
        files = import_from_zip(Path(args.from_zip), overwrite=args.overwrite)
    else:
        files = download_kaggle_dataset(
            args.dataset,
            overwrite=args.overwrite,
            retries=args.retries,
        )

    print(f"Downloaded/copied {len(files)} CSV files into data/raw:")
    for path in files:
        print(f"- {path.name}")


if __name__ == "__main__":
    main()
