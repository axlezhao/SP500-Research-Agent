from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

KAGGLE_DATASET = "sadiqguru/s-and-p-500-stock-data-along-with-financials-and-news"

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

