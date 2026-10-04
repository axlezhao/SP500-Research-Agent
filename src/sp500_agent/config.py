import os
from dataclasses import dataclass
from pathlib import Path


# Where data/, models/, reports/ and web/dist live. Defaults to the source checkout; set SP500_HOME
# when the package is installed elsewhere (e.g. in a container).
PROJECT_ROOT = Path(os.environ.get("SP500_HOME") or Path(__file__).resolve().parents[2])
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLE_DIR = RAW_DIR / "sample"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

FEATURES_PATH = PROCESSED_DIR / "research_features.parquet"
NEWS_PATH = PROCESSED_DIR / "news.parquet"
MODEL_PATH = MODEL_DIR / "return_direction_model.joblib"
# Walk-forward predictions, model comparison, backtest results and the generated research report.
RESEARCH_DIR = REPORT_DIR / "research"

KAGGLE_DATASET = "sadiqguru/s-and-p-500-stock-data-along-with-financials-and-news"
# Kaggle downloads and ZIP imports are extracted here with their folder structure intact,
# so the per-ticker files in price_data/ stay separate from the company and news tables.
KAGGLE_DIR_NAME = "kaggle_sp500_dataset"

# Forward-return horizons computed for every row, in trading days (about a week and a month).
HORIZONS = (5, 21)


@dataclass(frozen=True)
class ResearchSpec:
    """What the model predicts.

    horizon: trading days ahead. relative_to: "market" = beat the median index member that day;
    "sector" = beat the median member of the same sector; "absolute" = the price rises.
    """

    horizon: int = 5
    relative_to: str = "market"

    def __post_init__(self) -> None:
        if self.horizon not in HORIZONS:
            raise ValueError(f"horizon must be one of {HORIZONS}")
        if self.relative_to not in ("market", "sector", "absolute"):
            raise ValueError("relative_to must be market, sector or absolute")

    @property
    def target_column(self) -> str:
        prefix = {"market": "target_beat_median", "sector": "target_beat_sector", "absolute": "target_up"}[self.relative_to]
        return f"{prefix}_{self.horizon}d"

    @property
    def future_column(self) -> str:
        return f"future_return_{self.horizon}d"

    @property
    def tradable_column(self) -> str:
        return f"tradable_return_{self.horizon}d"

    @property
    def description(self) -> str:
        return {
            "market": f"beat the median S&P 500 stock over the next {self.horizon} trading days",
            "sector": f"beat the median stock in its sector over the next {self.horizon} trading days",
            "absolute": f"close higher {self.horizon} trading days later",
        }[self.relative_to]

    @property
    def label(self) -> str:
        return f"{self.horizon}-day, " + {"market": "vs. all members", "sector": "vs. own sector", "absolute": "up or down"}[self.relative_to]

    def as_dict(self) -> dict:
        return {"horizon": self.horizon, "relative_to": self.relative_to, "description": self.description, "label": self.label}


# The setup the pipeline trains and the app serves. Chosen by the experiments in run_pipeline.py
# (highest out-of-sample rank-IC t-stat); see docs/FINDINGS.md.
PRODUCTION_SPEC = ResearchSpec(horizon=5, relative_to="market")
# The setups compared in every research run.
EXPERIMENT_SPECS = (
    ResearchSpec(5, "absolute"),
    ResearchSpec(5, "market"),
    ResearchSpec(5, "sector"),
    ResearchSpec(21, "market"),
    ResearchSpec(21, "sector"),
)
HORIZON_DAYS = PRODUCTION_SPEC.horizon
TARGET_DESCRIPTION = PRODUCTION_SPEC.description
# Backtests trade this many sessions after the signal date (signals use the close; trading at that same close is optimistic).
EXECUTION_LAG_DAYS = 1
TRADING_DAYS_PER_YEAR = 252
# News timestamps are interpreted in US market time; items at or after the close count toward the next session.
MARKET_TIMEZONE = "America/New_York"
MARKET_CLOSE_HOUR = 16

PRICE_FILE_CANDIDATES = [
    "prices.csv",
    "stock_prices.csv",
    "sp500_stocks.csv",
    "sample_prices.csv",
]

FUNDAMENTALS_FILE_CANDIDATES = [
    "01_company_info.csv",
    "fundamentals.csv",
    "financials.csv",
    "sp500_companies.csv",
    "sample_fundamentals.csv",
]

NEWS_FILE_CANDIDATES = [
    "02_company_news_sentiment.csv",
    "news.csv",
    "stock_news.csv",
    "sp500_news.csv",
    "sample_news.csv",
]


def load_environment() -> None:
    """Read keys and settings from PROJECT_ROOT/.env when python-dotenv is installed; real environment variables win."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)
