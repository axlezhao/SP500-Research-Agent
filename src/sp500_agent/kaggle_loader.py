from __future__ import annotations

import shutil
import time
import zipfile
from pathlib import Path

from .config import KAGGLE_DATASET, RAW_DIR


def _import_kagglehub():
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "KaggleHub is not installed. Run: pip install 'kagglehub[pandas-datasets]'"
        ) from exc
    return kagglehub


def copy_csv_files(csv_files: list[Path], raw_dir: Path = RAW_DIR, overwrite: bool = False) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for csv_file in csv_files:
        destination = raw_dir / csv_file.name
        if destination.exists() and not overwrite:
            copied.append(destination)
            continue
        shutil.copy2(csv_file, destination)
        copied.append(destination)
    return copied


def download_kaggle_dataset(
    dataset: str = KAGGLE_DATASET,
    raw_dir: Path = RAW_DIR,
    overwrite: bool = False,
    retries: int = 3,
    retry_wait_seconds: int = 10,
) -> list[Path]:
    """Download the full Kaggle dataset and copy CSV files into data/raw."""
    kagglehub = _import_kagglehub()
    raw_dir.mkdir(parents=True, exist_ok=True)

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            dataset_path = Path(kagglehub.dataset_download(dataset))
            break
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                raise RuntimeError(
                    "KaggleHub could not download the dataset. This usually means the "
                    "network connection to Kaggle's Google Cloud Storage download URL "
                    "was interrupted, blocked, or intercepted by SSL/proxy software.\n\n"
                    "Try one of these:\n"
                    "1. Run again; signed Kaggle URLs sometimes fail transiently.\n"
                    "2. Switch networks or disable VPN/proxy/SSL inspection temporarily.\n"
                    "3. Download the Kaggle ZIP in your browser and run:\n"
                    "   python download_kaggle.py --from-zip /path/to/archive.zip\n\n"
                    "Kaggle does not expose this dataset as a live online database; "
                    "even the pandas DataFrame option still downloads file bytes."
                ) from last_error
            print(
                f"Kaggle download failed on attempt {attempt}/{retries}. "
                f"Retrying in {retry_wait_seconds}s..."
            )
            time.sleep(retry_wait_seconds)

    csv_files = sorted(dataset_path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files were found in downloaded dataset: {dataset_path}")
    return copy_csv_files(csv_files, raw_dir, overwrite=overwrite)


def import_from_zip(zip_path: Path, raw_dir: Path = RAW_DIR, overwrite: bool = False) -> list[Path]:
    """Extract CSV files from a browser-downloaded Kaggle ZIP into data/raw."""
    zip_path = Path(zip_path).expanduser()
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    with zipfile.ZipFile(zip_path) as archive:
        csv_members = [member for member in archive.namelist() if member.lower().endswith(".csv")]
        if not csv_members:
            raise FileNotFoundError(f"No CSV files found inside ZIP: {zip_path}")

        for member in csv_members:
            destination = raw_dir / Path(member).name
            if destination.exists() and not overwrite:
                copied.append(destination)
                continue
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            copied.append(destination)
    return copied


def list_raw_csvs(raw_dir: Path = RAW_DIR) -> list[Path]:
    return sorted(raw_dir.glob("*.csv"))
