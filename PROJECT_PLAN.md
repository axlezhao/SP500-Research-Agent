# Project Plan

## Done

- Data pipeline for the Kaggle dataset (prices, company info, news), with a synthetic random-walk sample for testing.
- Leakage-free features: time-series, cross-sectional ranks, sector-relative returns, news aligned to trading sessions.
- Walk-forward validation with an embargo; comparison of logistic regression, random forest and gradient boosting.
- Calibration, permutation importance, single-signal information coefficients, generated research report.
- Out-of-sample backtest: long-only, long-short and equal-weight benchmark, with costs and next-day execution.
- LLM research agent (DeepSeek or Claude) with 11 tools; terminal chat and Streamlit chat.
- Streamlit app with agent chat, backtest, model lab and screener tabs.

## Next

- **Data science:** FinBERT or embedding-based sentiment; hyperparameter search nested inside the walk-forward folds; predict cross-sectional outperformance instead of raw direction.
- **Quant:** point-in-time index membership to remove survivorship bias; volatility-scaled position sizing; sector-neutral long-short; sensitivity of results to cost and holding-period assumptions.
- **Agent:** an evaluation set of research questions with graded answers; streaming responses in the app; a tool that runs a custom backtest with user-chosen parameters.
