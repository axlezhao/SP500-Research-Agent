import os
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

# Prediction horizon in trading days.
HORIZON_DAYS = 5
# What the model predicts. "relative": does the stock beat the median index member over the next HORIZON_DAYS
# sessions? This matches how the predictions are used (ranking stocks against each other) and removes the
# market-wide move that dominates "absolute": does the stock's price rise? On live 2014-2026 data the relative
# target roughly doubled the out-of-sample rank IC.
TARGET_MODE = "relative"
TARGET_COLUMNS = {"relative": "target_beat_median_5d", "absolute": "target_up_5d"}
TARGET_DESCRIPTIONS = {
    "relative": "beat the median S&P 500 stock over the next 5 trading days",
    "absolute": "close higher 5 trading days later",
}
TARGET_DESCRIPTION = TARGET_DESCRIPTIONS[TARGET_MODE]
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
