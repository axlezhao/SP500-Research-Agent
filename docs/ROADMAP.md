# Roadmap

## Done

**Data**
- **Live sources:** Wikipedia (membership history), Yahoo Finance (prices, SPY), SEC EDGAR (point-in-time fundamentals and filings), FRED (macro), Yahoo RSS or Finnhub (news). All cached and rate-limited, with a data-quality report.
- **Survivorship handling:** training and trading only on each date's members; former members included where prices exist; recycled tickers detected and dropped.
- **Offline sources:** the Kaggle dataset and a synthetic random-walk sample.

**Research**
- **Point in time:** features with no look-ahead, including SEC figures as first reported.
- **Validation:** walk-forward with an embargo; three-model comparison; calibration, permutation importance, single-signal ICs.
- **Five research setups:** 5- and 21-day horizons; peer groups of all members, own sector, or up/down. Chosen by a rule fixed in advance.
- **Backtest and attribution:** costs, next-day execution, sector-neutral option, S&P 500 and equal-weight benchmarks; four-factor attribution.
- **Write-up:** [FINDINGS.md](FINDINGS.md).

**Product**
- **Agent:** DeepSeek or Claude, with 15 tools including live news and SEC filings.
- **Web dashboard:** FastAPI + React, with a chat that streams each tool call; a Streamlit alternative.
- **Engineering:** Docker, GitHub Actions (tests, dashboard build, container smoke test), a demo mode that caps LLM cost.

## Next

**Data**
- Point-in-time sector classification (sectors are currently today's).
- Prices for delisted stocks from a source that keeps them (e.g. Tiingo, or CRSP through a university licence) to close the survivorship gap.
- Automatic detection and correction of unadjusted splits and spin-offs.

**Research**
- Hyperparameter search nested inside the walk-forward folds.
- Turnover control: keep a holding while it stays near the top, to cut the cost drag that decides the 5-day results.
- A longer out-of-sample history, and robustness across sub-periods and cost levels.
- Sentiment from news history (e.g. FinBERT on Finnhub headlines).

**Quant**
- Volatility-scaled position sizing.
- A full factor-neutral portfolio (beta, size, value), to isolate what's left of the signal.

**Agent**
- An evaluation set: 30–50 research questions with checkable answers, scored automatically.
- Answers streamed word by word, with each number linked to the tool result it came from.
- A tool that runs a custom backtest the user describes.

**Operations**
- A hosted demo (see [DEPLOY.md](DEPLOY.md)) with a scheduled daily data refresh.
- Linting and type-checking in CI; a browser smoke test of the dashboard.
