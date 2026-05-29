# S&P 500 AI Research Agent

Educational data science project using S&P 500 prices, financials, and news.

Dataset:

https://www.kaggle.com/datasets/sadiqguru/s-and-p-500-stock-data-along-with-financials-and-news

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Set up Kaggle credentials if you have not already:

1. Go to your Kaggle account settings.
2. Create an API token.
3. Put `kaggle.json` in `~/.kaggle/kaggle.json`.

## Download the Dataset Automatically

Download all dataset files into `data/raw/`:

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
python -m src.sp500_agent.agent --ticker AAPL
```

This writes a markdown research brief to `reports/`.

## What the Project Does

- Downloads the full Kaggle dataset with KaggleHub.
- Copies all CSV files into `data/raw/`.
- Tries to infer which file contains prices, fundamentals, and news.
- Engineers returns, volatility, momentum, sentiment, and fundamentals.
- Trains a baseline 5-day return direction model.
- Generates an analyst-style research brief.

This project is for education and research only. It is not financial advice.



## Chatbot GUI

Run the presentation-friendly chatbot app:

```bash
streamlit run streamlit_app.py
```

Try prompts like:

- `Analyze NVDA`
- `Top 10 stocks`
- `Weakest 10 stocks`
- `Compare the model view for AAPL`
