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

**Bias fixes and research process (October 2026)**
- **No placeholder sectors:** former members get a sector from their SEC industry (SIC) code instead of "Unknown", which had told the model which stocks would leave the index.
- **Delisting returns:** stocks that stop trading stay in the backtest to their last price, with a −30% delisting return for apparent failures.
- **Session calendar:** "N sessions ago" counts sessions, not rows, across gaps in a ticker's history; a truncation test guards every feature against look-ahead.
- **Ranked target:** the percentile of the volatility-scaled forward return, learned by regressors; models chosen by the rank IC's Newey-West t-stat instead of AUC.
- **Normalised inputs:** every signal ranked within its date and mapped to a normal score; market conditions only as interactions.
- **Quarterly refits** with hyperparameter search nested inside each training window; training on non-overlapping dates.
- **Baseline:** a fixed-sign composite of documented anomalies that fitted models must beat.
- **Untouched holdout** (from 2025) with a logged, one-time evaluation; a forward-test log of every live ranking.

**New signals**
- Earnings surprise (SUE) timed by 8-K earnings releases, and the earnings-announcement return.
- Short-term reversal within industry; industry momentum; residual momentum.
- Beta, idiosyncratic volatility, MAX, 52-week-high proximity.
- Gross profitability, accruals, asset growth, net share issuance (year-to-date cash flows turned into quarters).
- Insider purchases from SEC's Form 4 data sets (optional).
- A finance sentiment lexicon with negation; optional Loughran-McDonald dictionary or FinBERT.
- IC decay from 1 to 63 sessions; IC by year, sector and size.

**Portfolio construction and statistics**
- Buffer zones and score smoothing to cut turnover; per-stock costs scaled by Corwin-Schultz spread estimates; borrow fees for shorts.
- A cvxpy optimiser with a Ledoit-Wolf risk model, trading costs in the objective, a tracking-error budget (long-only) and beta/size/value neutrality at a volatility target (long-short).
- Staggered rebalance starts; net returns across cost levels and the break-even cost; the equal-weight universe as the main benchmark.
- Newey-West t-stats for ICs and attribution; stationary-bootstrap intervals; deflated Sharpe ratio; probability of backtest overfitting.
- Attribution cross-checked on Kenneth French's factors (Fama-French five, momentum, short-term reversal).

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
- Point-in-time GICS history (current sectors still come from today's list; SIC codes fill in former members).
- Analyst revisions and short interest, if a free point-in-time source turns up (FINRA publishes short volume).
- A longer out-of-sample history: start the data in 2005 to cover 2008.
- LightGBM with a ranking objective (lambdarank) alongside the rank regressors.

**Quant**
- A daily-resolution backtest, so overlapping (Jegadeesh-Titman) tranches can be held at once rather than compared start by start.
- A structural factor risk model (industry and style factors) in place of the Ledoit-Wolf sample covariance.
- Market-impact costs that scale with trade size relative to daily volume.

**Agent**
- An evaluation set: 30–50 research questions with checkable answers, scored automatically.
- Answers streamed word by word, with each number linked to the tool result it came from.
- A tool that runs a custom backtest the user describes.

**Operations**
- A hosted demo (see [DEPLOY.md](DEPLOY.md)) with a scheduled daily data refresh.
- Linting and type-checking in CI; a browser smoke test of the dashboard.
