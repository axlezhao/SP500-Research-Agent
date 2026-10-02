import zipfile

import pandas as pd

from sp500_agent.config import KAGGLE_DIR_NAME
from sp500_agent.data_loader import load_fundamentals, load_news, load_prices
from sp500_agent.kaggle_loader import copy_dataset_tree, import_from_zip


def _write_kaggle_layout(sample_raw_dir, writer):
    """Mimic the Kaggle dataset: per-ticker price files without a ticker column, plus two tables."""
    prices = pd.read_csv(sample_raw_dir / "sample_prices.csv")
    for ticker, group in prices.groupby("ticker"):
        writer(f"price_data/{ticker}.csv", group.drop(columns="ticker").to_csv(index=False))
    writer("01_company_info.csv", pd.read_csv(sample_raw_dir / "sample_fundamentals.csv").to_csv(index=False))
    writer("02_company_news_sentiment.csv", pd.read_csv(sample_raw_dir / "sample_news.csv").to_csv(index=False))
    return set(prices["ticker"])


def _assert_loads_everything(raw_dir, tickers):
    assert set(load_prices(raw_dir)["ticker"]) == tickers
    assert set(load_fundamentals(raw_dir)["ticker"]) == tickers
    assert set(load_news(raw_dir)["ticker"]) == tickers


def test_zip_import_keeps_price_folder(tmp_path, sample_raw_dir):
    zip_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        tickers = _write_kaggle_layout(sample_raw_dir, archive.writestr)
    raw_dir = tmp_path / "raw"
    import_from_zip(zip_path, raw_dir)
    assert (raw_dir / KAGGLE_DIR_NAME / "price_data" / "AAPL.csv").exists()
    _assert_loads_everything(raw_dir, tickers)


def test_zip_with_top_level_folder_and_macos_junk(tmp_path, sample_raw_dir):
    zip_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        tickers = _write_kaggle_layout(sample_raw_dir, lambda name, data: archive.writestr(f"archive/{name}", data))
        archive.writestr("__MACOSX/archive/._01_company_info.csv", "junk")
    raw_dir = tmp_path / "raw"
    import_from_zip(zip_path, raw_dir)
    assert not any("__MACOSX" in path.parts for path in raw_dir.rglob("*"))
    _assert_loads_everything(raw_dir, tickers)


def test_zip_entries_cannot_escape_raw_dir(tmp_path):
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("../../outside.csv", "a,b\n1,2\n")
    raw_dir = tmp_path / "nested" / "raw"
    written = import_from_zip(zip_path, raw_dir)
    assert all(raw_dir in path.parents for path in written)
    assert not (tmp_path / "outside.csv").exists()


def test_kagglehub_download_keeps_price_folder(tmp_path, sample_raw_dir):
    download = tmp_path / "kagglehub_cache"

    def writer(name, data):
        (download / name).parent.mkdir(parents=True, exist_ok=True)
        (download / name).write_text(data)

    tickers = _write_kaggle_layout(sample_raw_dir, writer)
    raw_dir = tmp_path / "raw"
    copy_dataset_tree(download, raw_dir)
    _assert_loads_everything(raw_dir, tickers)
