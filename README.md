# S&P 500 AI Research Agent

Educational data science project using S&P 500 prices, financials, and news.

Dataset:

https://www.kaggle.com/datasets/sadiqguru/s-and-p-500-stock-data-along-with-financials-and-news

## Setup

Install dependencies (this also installs the project in editable mode, so `import sp500_agent` works):

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the versions the project was last tested with. For a looser install, use `pip install -e ".[app,kaggle,dev]"`.

Set up Kaggle credentials if you have not already:

1. Go to your Kaggle account settings.
2. Create an API token.
3. Put `kaggle.json` in `~/.kaggle/kaggle.json`.

## Download the Dataset Automatically

If you imported the Kaggle data with an older version of this project, the per-ticker price files were flattened into `data/raw/`. Remove those files and re-import with `--overwrite`.


Download all dataset files into `data/raw/kaggle_sp500_dataset/` (subfolders such as `price_data/` are kept):

```bash
python download_kaggle.py
```

Or download and run the project pipeline in one command:

```bash
python run_pipeline.py --download-kaggle
```

Kaggle datasets are static files, not live online SQL databases. KaggleHub's pandas DataFrame option still downloads file bytes over HTTPS, so the data must exist locally at least temporarily.

If KaggleHub fails with an SSL error from `storage.googleapis.com`, download the dataset ZIP from Kaggle in your browser, then run:

```bash
python download_kaggle.py --from-zip ~/Downloads/archive.zip
```

Or import the ZIP and run the pipeline in one command:

```bash
python run_pipeline.py --from-zip ~/Downloads/archive.zip
```

If you only want to test the project without Kaggle credentials:

```bash
python run_pipeline.py --make-sample
```

## Run the Agent

After the pipeline has created features and trained the baseline model:

```bash
python -m sp500_agent.agent --ticker AAPL
```

This writes a markdown research brief to `reports/`.

## What the Project Does

- Downloads the full Kaggle dataset with KaggleHub, keeping its folder structure.
- Tries to infer which file contains prices, fundamentals, and news.
- Engineers returns, annualised volatility, 60-day momentum, and news sentiment.
- Trains a baseline 5-day return direction model and evaluates it on a time-ordered holdout.
- Generates an analyst-style research brief from each stock's most recent trading day.

## How the Model Is Evaluated

- **Time-ordered holdout.** The model is trained on earlier dates and tested on later ones, with a 5-session gap so training targets never overlap the test period. A random split would let neighbouring days of the same stock appear on both sides, which inflates the scores (on the random-walk sample data it reported ROC AUC ≈ 0.69 where the true answer is 0.50).
- **Baseline.** The report compares accuracy with always predicting the more common direction.
- **Fundamentals are display-only.** The dataset has one current snapshot of market cap, P/E, revenue, etc. Attaching it to every historical row would leak future information, so these fields appear in briefs but are not model inputs.
- **Current predictions.** The latest rows, whose 5-day outcome is still unknown, are kept for scoring and excluded from training. Tickers whose last price is more than 7 days older than the newest price are not ranked.
- **News.** Timestamps with a timezone are converted to New York time. Items published at or after 4pm count toward the next session, and weekend or holiday news moves to the next trading day. 20-day sentiment is averaged over articles only, so a quiet period reads as unknown rather than neutral.

## Tests

```bash
pytest
```

The suite includes a leakage check: on random-walk sample data the holdout ROC AUC must stay close to 0.5.

This project is for education and research only. It is not financial advice.

## Chatbot GUI

Run the presentation-friendly chatbot app (it reads `data/processed/research_features.parquet` and the trained model):

```bash
streamlit run streamlit_app.py
```

Try prompts like:

- `Analyze NVDA`
- `Top 10 stocks`
- `Weakest 10 stocks`
- `Compare the model view for AAPL`
- `What's the rank of $ON?` (a `$` marks a ticker that is also an English word)
