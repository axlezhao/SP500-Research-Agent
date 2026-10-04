# S&P 500 AI Research Agent

[![CI](https://github.com/axlezhao/SP500-Research-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/axlezhao/SP500-Research-Agent/actions/workflows/ci.yml)

A research platform on **free public data**, in three layers:

| Layer | What it does | Where |
|---|---|---|
| **Data science** | Point-in-time features from prices, SEC filings and macro data; three models compared with walk-forward validation; calibration, permutation importance, data-quality checks, an auto-generated research report | `ingest.py`, `features.py`, `model.py`, `validation.py`, `research.py` |
| **Quant research** | Out-of-sample backtest on the index members of each date against the S&P 500 (SPY) and an equal-weight benchmark, sector-neutral option, costs, next-day execution, and a four-factor attribution (market, size, value, momentum) | `backtest.py`, `attribution.py` |
| **AI agent** | An LLM (DeepSeek or Claude) answering research questions with 15 tools over the data, the model, the backtest, and live news and SEC filings | `research_tools.py`, `llm_agent.py` |

Everything is served through a **web dashboard**: a FastAPI backend and a React + TypeScript frontend with an overview, screener, stock pages, a streaming agent chat, and backtest, model and data views. The original Streamlit app remains as a lightweight alternative.

![Dashboard tour: overview, stock page, the research agent answering live, backtest, model lab](docs/media/tour.gif)

**Findings.** After costs, no setup beats the S&P 500 with statistically meaningful alpha. Before costs, the best long portfolio matched the index (17.6% a year each), and weekly trading costs then put it 3.9 points behind. The write-up covers what was tried and why each attempt fell short: [docs/FINDINGS.md](docs/FINDINGS.md).

This project is for education and research only. It is not financial advice.

## Quick start

**With Docker** (no Python or Node needed):

```bash
cp .env.example .env        # add SEC_USER_AGENT (a contact email SEC requires) and, optionally, an LLM key
docker compose up --build   # first start builds the data (~4 minutes), then serves http://localhost:8000
```

Use `HOST_PORT=8010 docker compose up` if port 8000 is taken, or `DATA_SOURCE=sample` for an instant start on synthetic data. To put a demo online, see [docs/DEPLOY.md](docs/DEPLOY.md).

**Without Docker:**

```bash
pip install -r requirements.txt
cp .env.example .env        # add SEC_USER_AGENT (a contact email SEC requires) and an LLM key
python run_pipeline.py      # downloads ~12 years of data for ~600 stocks; about 4 minutes the first time
cd web && npm install && npm run build && cd ..   # build the dashboard once (needs Node 20+)
sp500-web                   # open http://127.0.0.1:8000
```

`streamlit run streamlit_app.py` still works if you'd rather not install Node.

For a quick trial run, use `python run_pipeline.py --limit 40`, which takes the first 40 current members and runs in under a minute. To use no network at all, `python run_pipeline.py --make-sample` builds a synthetic random-walk dataset. It's useful for testing, since no model should find an edge in it.

## Data sources

All free; only SEC needs a contact email and only Finnhub needs a key.

| Source | Provides | Access |
|---|---|---|
| **Wikipedia** | Current S&P 500 constituents (sector, CIK) and the dated log of every index change since 1976 | Public pages |
| **Yahoo Finance** (via `yfinance`) | Daily prices adjusted for splits and dividends, split history | No key; unofficial |
| **SEC EDGAR** | Every figure companies report in 10-K/10-Q filings, with filing dates; recent filings list | No key; `SEC_USER_AGENT` with a contact email |
| **FRED** | VIX, 3-month T-bill and 10-year Treasury yields | No key |
| **Yahoo Finance RSS** or **Finnhub** | Recent headlines, fetched when the agent asks | RSS needs no key; Finnhub uses `FINNHUB_API_KEY` |

Downloads are cached in `data/raw/live/`. Prices and macro data refresh after 12 hours; membership and fundamentals after 7 days. `--refresh` forces a fresh download. Yahoo's terms don't allow redistributing its data, so the downloaded data stays out of git (`data/` is ignored).

The Kaggle dataset is still supported: `python run_pipeline.py --download-kaggle`, or `--from-zip ~/Downloads/archive.zip` if KaggleHub has trouble downloading.

## 1. Data science

**Point in time, everywhere.** Every input is limited to what was public at the close of its date:

- **Fundamentals** come from SEC XBRL facts. Each period keeps its *first-reported* value, so later restatements never leak backwards. A fourth quarter is derived as the annual figure minus the first three quarters. A filing becomes usable the day *after* it was filed, and data older than about 18 months counts as unknown.
- **Valuation ratios** (earnings yield, sales yield, book-to-market, P/E) pair each filing's share count with the *as-traded* price (Yahoo's split adjustment undone). Rows whose implied price-to-book is implausible are dropped, which happens when the cover page reports a different share class (e.g. Berkshire's Class A count).
- **Macro series** are lagged one day.
- **Index membership** is reconstructed by walking Wikipedia's change log backwards from today's list. The model trains, ranks and trades only stocks that were members on each date.

**Features.**
- **Price features:** returns, momentum, distance from the 50-day average, annualised volatility, volume ratio.
- **Cross-sectional ranks** among that day's index members: price features plus value (earnings yield, sales yield, book-to-market), quality (margin, ROE), growth and size.
- **Sector-relative return.**
- **Market conditions:** VIX, its 20-day change, and the yield-curve spread.

Features missing on more than 60% of rows, or constant, are left out automatically.

**Research setups.** What the model predicts is a `ResearchSpec` (`config.py`) with two settings:
- **Horizon:** 5 or 21 trading days.
- **Peer group:** beat the median index member, beat the median of its own sector, or simply rise.

Every run compares five setups with the same model, walk-forward and backtested identically. Sector-relative setups trade sector-neutral portfolios. Production (`PRODUCTION_SPEC`) is chosen by a rule fixed in advance, the highest out-of-sample rank-IC t-stat, so backtest results can't be cherry-picked. Currently that's *beat the median member over 5 trading days*. Predicting relative rather than absolute moves removes the market-wide swing that is the same for every stock, and it raised the rank IC by 40%.

**Validation.** Expanding-window walk-forward folds, with a gap between training and test labels as long as the prediction horizon. Logistic regression, random forest and gradient boosting are scored by AUC, accuracy against a majority-class baseline, Brier score and information coefficient (IC; t-stat from non-overlapping dates). The best model is refit on all labelled data. Also reported:
- **Calibration:** do the predicted probabilities match how often the event happened?
- **Permutation importance:** how much the score drops when each feature is shuffled.
- **Single-feature ICs:** each feature's own IC, next to the model's.

**Data quality.** Each run writes `data/processed/data_quality.json` covering:
- **Coverage:** of every source.
- **Survivorship:** how many former members have usable prices.
- **Reused tickers:** symbols dropped because they now belong to a different company.
- **Price anomalies:** daily moves above 40% and gaps between trading days.

The research report and the app's Data tab summarise it.

Results go to `reports/research/`, including a readable `research_report.md`.

## 2. Quant research

`backtest.py` turns the walk-forward predictions into portfolios:

- Every horizon (5 or 21 sessions), rank that day's index members by predicted probability. Enter one session later and hold for the horizon, so holding periods never overlap.
- **Long-only** buys the top 20%, **long-short** also shorts the bottom 20% (within each sector for sector-relative setups). They're compared with the **S&P 500** (SPY with dividends, from Yahoo) and an **equal-weight** portfolio of all members.
- Costs are charged on every unit of weight traded, from the previous period's drifted weights.
- Sharpe ratios for long-only and the benchmark are measured in excess of 3-month T-bills (FRED). Long-short is self-financing, so it uses raw returns.
- Reported: total return, CAGR, volatility, Sharpe, maximum drawdown, hit rate, turnover, information ratio against both benchmarks, return by prediction quintile, and IC at each rebalance.
- **Attribution** (`attribution.py`) regresses each strategy on market, size, value and momentum factors built from the same members and dates. It separates real skill (alpha) from exposure to known factors. The current long portfolio is the market plus a small-cap tilt (alpha −2.8% a year, t = −1.0).

**Honest result (live data, 2014–2026).** No setup produces alpha distinguishable from zero after costs, and the production long portfolio trails the S&P 500. Before costs it matched the index, so weekly trading costs account for the whole gap. The monthly horizon did better in the backtest but not significantly, and the selection rule keeps it out of production. Details: [docs/FINDINGS.md](docs/FINDINGS.md).

**Caveats.**
- **Survivorship bias:** reduced, not removed. Former members delisted after mergers or failures often have no Yahoo data; the report states the coverage.
- **Unadjusted corporate actions:** Yahoo occasionally misses a spin-off adjustment. These show up as extreme moves in the quality report, and the agent warns about them.
- **Costs:** flat, with no market impact or short-borrow fees.

## 3. AI agent

`llm_agent.py` runs a tool-calling loop over these tools (`research_tools.py`):

| Group | Tools |
|---|---|
| Overview | `dataset_overview` (as-of date, model quality, data coverage) |
| Stocks | `search_companies`, `stock_snapshot`, `rank_stocks`, `screen_stocks`, `compare_stocks`, `price_history`, `sector_summary` |
| Fundamentals | `fundamentals_history` (point-in-time SEC data with filing dates) |
| Market | `macro_snapshot` (VIX, rates, yield curve vs. history), `index_changes` (S&P 500 additions and removals) |
| Live lookups | `recent_news` (Yahoo RSS or Finnhub), `recent_filings` (SEC EDGAR 10-K/10-Q/8-K with links) |
| Evidence | `model_performance`, `backtest_results` |

The tools validate their arguments and return errors the model can recover from. `stock_snapshot` also flags recent moves that look like unadjusted corporate actions. The system prompt tells the agent to:
- ground every number in a tool result;
- state the data's as-of date;
- treat headlines as context rather than proof;
- explain the model's weak edge honestly;
- not give buy or sell instructions.

**Providers.** Put a key in `.env`:

| Provider | Key | Default model | Override |
|---|---|---|---|
| DeepSeek (OpenAI-compatible) | `DEEPSEEK_API_KEY` | `deepseek-v4-pro` | `DEEPSEEK_MODEL=deepseek-flash` |
| Claude (Anthropic API) | `ANTHROPIC_API_KEY` | `claude-opus-5-5`, medium effort | `CLAUDE_AGENT_MODEL`, `CLAUDE_AGENT_EFFORT` |

With both keys set, DeepSeek is used unless `LLM_PROVIDER=anthropic`. The Claude backend:
- caches the system prompt and tools;
- turns on Anthropic's server-side fallback, which retries a declined request on a recommended fallback model.

Conversation history is append-only in both backends.

```bash
sp500-chat                                                        # interactive terminal chat
sp500-chat "Compare NVDA and AMD on growth, valuation and recent filings"
sp500-brief --ticker AAPL                                         # markdown brief, no LLM needed
```

In the app, the **Research agent** tab shows each tool call and its result and draws price charts. Without an LLM key it falls back to a rule-based assistant.

## Web dashboard

`sp500-web` (`src/sp500_agent/api/server.py`) serves a JSON API and the built React app from one origin. Pass `--port` if 8000 is taken.

| View | What it shows |
|---|---|
| **Overview** | Model quality at a glance, highest- and lowest-ranked members, backtest curve, sector tilt, rates and VIX |
| **Screener** | Every ranked member with probability, stance, returns, volatility and point-in-time fundamentals; sortable and filterable |
| **Stock** | TradingView price chart with volume and ranges, percentile against other members, SEC fundamentals history with charts, live news and filings, one-click agent brief |
| **Research agent** | Chat that streams each tool call as it happens (expandable to the raw result), then a formatted answer; conversation kept per browser tab |
| **Backtest** | Growth of $1, drawdowns, quintile returns, rolling rank IC, statistics and caveats |
| **Model lab** | Walk-forward comparison, AUC by fold, calibration, permutation importance, single-signal ICs, report download |
| **Data** | Sources and fetch times, survivorship coverage, quality checks, macro charts, recent index changes |

| | |
|---|---|
| ![Overview](docs/media/overview.png) | ![Stock page](docs/media/stock.png) |
| ![Research agent](docs/media/agent.png) | ![Backtest](docs/media/backtest.png) |
| ![Model lab](docs/media/model-lab.png) | ![Screener](docs/media/screener.png) |

Regenerate these with `python scripts/capture_screenshots.py --url http://127.0.0.1:8000` (needs Google Chrome and `pip install playwright`).

Built for daily use: light and dark themes (following the system or a toggle), ⌘K ticker search, keyboard-accessible tables and controls, layouts that work down to phone width, and a colour-blind-checked chart palette.

**API.** `GET /api/overview`, `/api/stocks`, `/api/stocks/{ticker}` (plus `/prices`, `/fundamentals`, `/news`, `/filings`), `/api/backtest`, `/api/model`, `/api/data`; `POST /api/agent/chat` streams server-sent events (`tool_start`, `tool_end`, `answer`); `POST /api/reload` picks up a new pipeline run without restarting. Interactive docs at `/docs`.

**Frontend development.** Run `sp500-web --reload` and, in `web/`, `npm run dev` (port 5173, proxying `/api`). Stack: Vite, React 19, TypeScript, Tailwind CSS 4, TanStack Query, Recharts, TradingView Lightweight Charts.

## Pipeline options

```
python run_pipeline.py [--source live|kaggle|sample] [--start 2014-01-01] [--limit N] [--refresh]
                       [--folds 5] [--cost-bps 10] [--quantile 0.2] [--max-rows 150000]
```

## Tests

```bash
pytest
```

About 140 tests run offline in roughly 25 seconds, and GitHub Actions runs them on every push together with the dashboard build and a Docker smoke test:
- **Connectors:** every connector is tested against canned responses: Wikipedia HTML, Yahoo frames, SEC XBRL JSON (restatements, derived Q4, share classes), FRED CSV, RSS and Finnhub JSON, HTTP retries.
- **Ingest:** the cached ingest runs end to end against fake services, including dropping reused tickers.
- **Point in time:** a filing must be invisible on its filing day, macro data is lagged, and only index members are ranked and traded.
- **Leakage:** on random-walk data the walk-forward AUC stays near 0.5.
- **Backtest:** arithmetic, including excess-of-cash Sharpe, sector-neutral portfolios and matching S&P 500 windows.
- **Research setups:** monthly targets and embargo, sector-relative targets, and factor attribution recovering known exposures.
- **Tools and agent:** all 15 tools, and both agent loops with scripted fake clients.
- **API:** every endpoint, including the streamed chat events, the rule-based fallback, demo-mode limits and serving the frontend's client-side routes.

## Project layout

```
run_pipeline.py            data → features → walk-forward research → backtest → model + report
web/                       React + TypeScript dashboard (Vite); built into web/dist and served by sp500-web
Dockerfile, docker-compose.yml, docker/   container build and first-start entrypoint
docs/                      FINDINGS.md (research write-up), DEPLOY.md, media/ (screenshots, tour)
scripts/                   capture_screenshots.py
.github/workflows/ci.yml   tests, dashboard build, Docker smoke test
streamlit_app.py           lightweight alternative UI
src/sp500_agent/
  sources/                 wikipedia.py, yahoo.py, sec.py, fred.py, news.py, http.py (rate limits, retries)
  ingest.py                assemble live / Kaggle / sample data, cache, clean, quality report
  features.py              time-series, cross-sectional, point-in-time fundamental and macro features
  model.py                 model zoo, final fit, scoring index members on the latest date
  validation.py            walk-forward folds, model comparison, IC, calibration, importance
  backtest.py              portfolio backtest (overall or sector-neutral) against the S&P 500 and equal weight
  attribution.py           four-factor attribution: alpha vs. market, size, value, momentum
  research.py              orchestrates a research run, saves artifacts, writes the report
  research_tools.py        the agent's tools
  llm_agent.py             DeepSeek and Claude tool-calling agents, terminal chat
  api/server.py            FastAPI backend: JSON endpoints, streaming agent chat, serves the dashboard
  chat.py, agent.py        rule-based chat and markdown briefs (no LLM needed)
  charts.py                Altair charts for the app
tests/                     pytest suite
```
