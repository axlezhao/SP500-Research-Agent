from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
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
