from __future__ import annotations

import shutil
import time
import zipfile
from pathlib import Path, PurePosixPath

from .config import KAGGLE_DATASET, KAGGLE_DIR_NAME, RAW_DIR


def _import_kagglehub():
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "KaggleHub is not installed. Run: pip install 'kagglehub[pandas-datasets]'"
        ) from exc
    return kagglehub


def _safe_relative_parts(name: str) -> tuple[str, ...] | None:
    """Return path parts that cannot escape the destination folder, or None to skip the entry."""
    parts = tuple(part for part in PurePosixPath(name.replace("\\", "/")).parts if part not in ("", ".", "..", "/"))
    if not parts or "__MACOSX" in parts:
        return None
    return parts


def copy_dataset_tree(dataset_path: Path, raw_dir: Path = RAW_DIR, overwrite: bool = False) -> list[Path]:
    """Copy every CSV under dataset_path into raw_dir/kaggle_sp500_dataset, keeping subfolders."""
    destination_root = raw_dir / KAGGLE_DIR_NAME
    copied: list[Path] = []
    for csv_file in sorted(Path(dataset_path).rglob("*.csv")):
        parts = _safe_relative_parts(csv_file.relative_to(dataset_path).as_posix())
        if parts is None:
            continue
        destination = destination_root.joinpath(*parts)
        copied.append(destination)
        if destination.exists() and not overwrite:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(csv_file, destination)
    return copied


def download_kaggle_dataset(
    dataset: str = KAGGLE_DATASET,
    raw_dir: Path = RAW_DIR,
    overwrite: bool = False,
    retries: int = 3,
    retry_wait_seconds: int = 10,
) -> list[Path]:
    """Download the full Kaggle dataset and copy its CSV files into data/raw."""
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

    copied = copy_dataset_tree(dataset_path, raw_dir, overwrite=overwrite)
    if not copied:
        raise FileNotFoundError(f"No CSV files were found in downloaded dataset: {dataset_path}")
    return copied


def import_from_zip(zip_path: Path, raw_dir: Path = RAW_DIR, overwrite: bool = False) -> list[Path]:
    """Extract CSV files from a browser-downloaded Kaggle ZIP into data/raw, keeping subfolders."""
    zip_path = Path(zip_path).expanduser()
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    destination_root = raw_dir / KAGGLE_DIR_NAME
    copied: list[Path] = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if not member.lower().endswith(".csv"):
                continue
            parts = _safe_relative_parts(member)
            if parts is None:
                continue
            destination = destination_root.joinpath(*parts)
            copied.append(destination)
            if destination.exists() and not overwrite:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)

    if not copied:
        raise FileNotFoundError(f"No CSV files found inside ZIP: {zip_path}")
    return copied


def list_raw_csvs(raw_dir: Path = RAW_DIR) -> list[Path]:
    if not raw_dir.exists():
        return []
    return sorted(path for path in raw_dir.rglob("*.csv") if "__MACOSX" not in path.parts)
