# Project Plan

## Done

- Live data from free public sources: Wikipedia (index membership history), Yahoo Finance (prices), SEC EDGAR (point-in-time fundamentals, filings), FRED (macro), Yahoo RSS / Finnhub (news). Cached, rate-limited, with a data-quality report.
- Survivorship-bias reduction: training and backtests use the index members of each date; former members' prices are included where available, and reused tickers are detected and dropped.
- Leakage-free features: time-series, cross-sectional ranks among members, point-in-time value/quality/growth/size factors, lagged macro.
- Relative target (beat the median member), walk-forward validation with an embargo, three-model comparison, calibration, permutation importance, single-signal ICs, generated research report.
- Out-of-sample backtest with costs, next-day execution and T-bill cash returns.
- LLM research agent (DeepSeek or Claude) with 15 tools, including live news and SEC filings; terminal and Streamlit chat.
- Kaggle dataset and synthetic sample kept as offline sources.

## Next

- **Data:** point-in-time sector classification; delisted-stock prices from a source that keeps them (e.g. Tiingo or CRSP via a university licence) to close the survivorship gap; corporate-action adjustment checks.
- **Data science:** monthly horizon alongside 5-day; hyperparameter search nested inside the walk-forward folds; FinBERT sentiment on Finnhub's news history.
- **Quant:** sector-neutral long-short, volatility-scaled sizing, factor attribution (how much of the return is just value or momentum), cost and horizon sensitivity.
- **Agent:** an evaluation set of research questions with graded answers; streaming responses; a tool that runs a custom backtest.
