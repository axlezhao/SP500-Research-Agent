# S&P 500 Research Agent

[![CI](https://github.com/axlezhao/SP500-Research-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/axlezhao/SP500-Research-Agent/actions/workflows/ci.yml)

An end-to-end research platform for S&P 500 stocks, built entirely on free public data. It brings together three disciplines in one codebase:

- **Data science:** point-in-time features from prices, SEC filings, earnings releases and macro data; a ranked target; fitted models compared walk-forward against a fixed-sign anomaly baseline, with an untouched holdout.
- **Quant research:** out-of-sample portfolio backtests with per-stock trading costs, cost-aware and factor-neutral construction, and attribution on home-made and Fama-French factors, with every result deflated for the number of configurations tried.
- **AI agent:** an LLM (DeepSeek or Claude) that answers research questions by calling 15 tools over the data, the model, the backtest, and live news and SEC filings.

All of it is served through a web dashboard: a FastAPI backend and a React + TypeScript frontend.

![Tour of the dashboard: overview, stock page, the research agent answering live, backtest and model lab](docs/media/tour.gif)

## What it found

Can free public data rank S&P 500 stocks well enough to beat the index after costs? **Not in the original setup.** Over May 2020 to September 2026, out of sample (run of 2 October 2026, before the research overhaul described below; re-run the pipeline for current results):

| Portfolio (5-day rebalance) | CAGR | Sharpe | Alpha vs. 4 factors (t) |
|---|---|---|---|
| Model's top 20% | 13.8% | 0.62 | −2.8% (−1.0) |
| Long top 20%, short bottom 20% | −2.6% | −0.10 | −1.5% (−0.4) |
| S&P 500 (SPY) | 17.6% | 0.92 | — |

- **Costs explain the gap.** Before costs, the top-20% portfolio earned 17.6% a year, the same as the index; trading every week then costs 3.8% a year.
- **Other setups don't change that.** A monthly horizon and sector-neutral portfolios were also tested; none shows alpha distinguishable from zero.

[Read the full findings →](docs/FINDINGS.md)

## Quick start

With Docker (nothing else to install):

```bash
cp .env.example .env          # set SEC_USER_AGENT (a contact email SEC requires); an LLM key is optional
docker compose up --build     # first start downloads and builds the data (~4 min), then serves the dashboard
```

Open http://localhost:8000. For an instant start on synthetic data, use `DATA_SOURCE=sample docker compose up --build`.

Without Docker (Python 3.11+, Node 20+):

```bash
pip install -r requirements.txt
cp .env.example .env
python run_pipeline.py                            # download data, run the research, train the model
cd web && npm install && npm run build && cd ..   # build the dashboard
sp500-web                                         # http://127.0.0.1:8000
```

See the [user guide](docs/GUIDE.md) for configuration, the agent and troubleshooting.

## Highlights

**Data**
- **Free sources:** Wikipedia (index membership history), Yahoo Finance (prices), SEC EDGAR (fundamentals, industry codes, earnings-release dates and, optionally, insider trades), FRED (VIX, Treasury yields), Kenneth French's factor library, and Yahoo RSS or Finnhub (news). All cached, with a data-quality report.
- **No look-ahead:** SEC figures are used as first reported, only from the day after filing; earnings surprises from the moment the release reached the market. The model trains and trades only on stocks that were index members on each date. A test rebuilds every feature on truncated data and checks nothing changes.
- **Survivorship and data cleaning:** former members get a sector from their SEC industry code, never a placeholder; stocks that stop trading stay in the backtest to their last price (with a delisting return for failures); "N days ago" counts trading sessions across data gaps. Recycled tickers, such as SunTrust's old symbol now used by another company, are detected and dropped.

**Research**
- **Signals:** short-term reversal within industry, industry and residual momentum, earnings surprise and announcement returns, beta, idiosyncratic volatility, MAX, 52-week high, value, gross profitability, accruals, asset growth, net issuance and insider buying, each ranked within its date.
- **Target and models:** the percentile of each stock's volatility-scaled forward return, learned by ridge, random forest and gradient boosting regressors; chosen by the rank IC's Newey-West t-stat; a fixed-sign composite of known anomalies as the baseline to beat.
- **Walk-forward validation:** refit every quarter, hyperparameters tuned inside each training window, training on non-overlapping dates, a gap as long as the horizon, and an untouched holdout from 2025 that is scored once and logged.
- **Diagnostics:** calibration, permutation importance, single-signal ICs, IC by year, sector and size, and IC decay from 1 to 63 days.
- **Six research setups** compared on every run: 5- or 21-day horizon, yes/no or ranked target, against all members or the stock's own sector.
- **Backtest:** next-day execution; per-stock costs scaled by estimated bid-ask spreads, and borrow fees; buffer zones and score smoothing; a cvxpy optimiser with a Ledoit-Wolf risk model that is neutral to beta, size and value; staggered start days; the equal-weight universe and the S&P 500 as benchmarks.
- **Is it luck?** Bootstrap confidence intervals, the deflated Sharpe ratio, the probability of backtest overfitting, and factor attribution with Newey-West t-stats on both home-made and Fama-French factors.

**Agent and app**
- **Research agent:** 15 tools including point-in-time fundamentals, live SEC filings and news, market conditions and the model's own evidence. Prompted to ground every number in a tool result and to state the model's limits.
- **Dashboard:** overview, screener, stock pages with TradingView charts, a chat that streams each tool call as it happens, and backtest, model and data views. Light and dark themes, ⌘K search, works on phones.
- **Engineering:** 190+ offline tests; GitHub Actions runs them, builds the dashboard and smoke-tests the Docker image. A demo mode caps LLM cost for public deployments.

## Screenshots

| | |
|---|---|
| ![Overview](docs/media/overview.png) | ![Stock page](docs/media/stock.png) |
| ![Research agent](docs/media/agent.png) | ![Backtest](docs/media/backtest.png) |
| ![Model lab](docs/media/model-lab.png) | ![Screener](docs/media/screener.png) |

## Documentation

| Document | What's in it |
|---|---|
| [User guide](docs/GUIDE.md) | Installation, data sources and keys, pipeline options, the dashboard, the agent, troubleshooting |
| [Architecture](docs/ARCHITECTURE.md) | How data flows, the point-in-time rules, modelling and validation, backtest and attribution, agent design, API, tests |
| [Findings](docs/FINDINGS.md) | The research question, method, results, everything that was tried and why it fell short |
| [Deployment](docs/DEPLOY.md) | Putting a public demo online, and the data-licensing and cost decisions involved |
| [Roadmap](docs/ROADMAP.md) | What's done and what could come next |

## Tech stack

**Python:** pandas, scikit-learn, SciPy, cvxpy, FastAPI, yfinance, requests, the OpenAI SDK (for DeepSeek) and the Anthropic SDK. **Frontend:** React 19, TypeScript, Vite, Tailwind CSS 4, TanStack Query, Recharts, TradingView Lightweight Charts. **Ops:** Docker, GitHub Actions, pytest.

---

For education and research only. Nothing here is investment advice.
