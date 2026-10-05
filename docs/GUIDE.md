# User guide

How to install, configure and use the S&P 500 Research Agent. For how it works internally, see [ARCHITECTURE.md](ARCHITECTURE.md).

- [Install](#install)
- [Configure](#configure)
- [Data sources](#data-sources)
- [Run the research pipeline](#run-the-research-pipeline)
- [Use the dashboard](#use-the-dashboard)
- [Use the research agent](#use-the-research-agent)
- [Other interfaces](#other-interfaces)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

## Install

**Option 1: Docker.** Needs only Docker.

```bash
cp .env.example .env
docker compose up --build
```

The first start downloads about 12 years of data for about 600 stocks and runs the research (around 4 minutes), then serves the dashboard at http://localhost:8000. Later starts reuse the data, which is kept in Docker volumes.

| Want to… | Run |
|---|---|
| Use another port | `HOST_PORT=8010 docker compose up` |
| Start instantly on synthetic data | `DATA_SOURCE=sample docker compose up --build` |
| Refresh the data | `docker compose exec app python run_pipeline.py`, then restart the container |

**Option 2: Local Python.** Needs Python 3.11+, and Node 20+ to build the dashboard.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                   # pinned versions; also installs the project in editable mode
cp .env.example .env
python run_pipeline.py
cd web && npm install && npm run build && cd ..
sp500-web                                         # http://127.0.0.1:8000
```

For a looser install, use `pip install -e ".[web,llm,app,kaggle,dev]"` instead of the pinned requirements.

## Configure

Settings live in `.env` at the project root, which git ignores; start from `.env.example`. Real environment variables take precedence.

| Variable | Needed for | Notes |
|---|---|---|
| `SEC_USER_AGENT` | SEC fundamentals and filings | SEC refuses requests without a contact, e.g. `SP500-Research-Agent you@example.com`. Without it, the pipeline skips fundamentals. |
| `DEEPSEEK_API_KEY` | AI agent (DeepSeek) | Default model `deepseek-v4-pro`; set `DEEPSEEK_MODEL=deepseek-flash` for a faster, lighter one. |
| `ANTHROPIC_API_KEY` | AI agent (Claude) | Default model `claude-opus-5-5` at medium effort; override with `CLAUDE_AGENT_MODEL`, `CLAUDE_AGENT_EFFORT`. |
| `LLM_PROVIDER` | Choosing between the two | `deepseek` or `anthropic`; by default DeepSeek is used if both keys are set. |
| `FINNHUB_API_KEY` | Optional news source | Without it, headlines come from Yahoo Finance RSS. |
| `DEMO_MODE`, `CHAT_LIMIT_PER_HOUR`, `CHAT_LIMIT_PER_DAY`, `ADMIN_TOKEN` | Public deployments | See [DEPLOY.md](DEPLOY.md). |
| `SP500_HOME` | Installs outside the source tree | Folder that holds `data/`, `models/`, `reports/` and `web/dist`. |
| `SP500_HOLDOUT_START` | Moving the holdout | First date of the untouched holdout (default `2025-01-01`), or `none`. Same as `--holdout-start`. |
| `SENTIMENT_MODEL`, `LM_DICTIONARY_PATH` | Headline sentiment | `lexicon` (default), `lm` (the Loughran-McDonald dictionary CSV from sraf.nd.edu at `LM_DICTIONARY_PATH`) or `finbert` (`pip install -e ".[finbert]"`). |

Without any LLM key, everything still works except the AI agent, which falls back to a rule-based assistant.

## Data sources

| Source | Provides | Access | Refreshed after |
|---|---|---|---|
| Wikipedia | Today's S&P 500 list (sector, CIK) and every index change since 1976 | Public | 7 days |
| Yahoo Finance (`yfinance`) | Daily prices adjusted for splits and dividends; split history; SPY for the benchmark | No key, unofficial | 12 hours |
| SEC EDGAR | Every figure in 10-K and 10-Q filings, with filing dates; each company's industry (SIC) code and earnings-release dates | `SEC_USER_AGENT` | 7 days |
| SEC insider data sets | Insider purchases and sales (Forms 3, 4, 5), quarterly; only with `--insiders` | `SEC_USER_AGENT` | 30 days |
| FRED | VIX, 3-month T-bill and 10-year Treasury yields | Public | 12 hours |
| Kenneth French data library | Daily Fama-French five factors, momentum and short-term reversal | Public | 7 days |
| Yahoo Finance RSS or Finnhub | Recent headlines, fetched when the agent or a stock page asks | RSS: none; Finnhub: key | Per session |

- **Cache:** downloads are kept in `data/raw/live/`; `--refresh` ignores it.
- **Licensing:** Yahoo's terms don't allow redistributing its data, so `data/` is git-ignored. Keep it that way if you publish the repository.

**Other sources.** The original Kaggle dataset still works: `python run_pipeline.py --download-kaggle`, or `--from-zip ~/Downloads/archive.zip` with a ZIP downloaded in your browser. `--make-sample` builds a synthetic random-walk dataset that needs no network. Because no model can find an edge in a random walk, it's a good check that the evaluation isn't leaking.

## Run the research pipeline

```bash
python run_pipeline.py [options]
```

| Option | Default | Effect |
|---|---|---|
| `--source live\|kaggle\|sample` | `live` | Where the data comes from |
| `--start 2014-01-01` | 2014-01-01 | First date downloaded (live) |
| `--limit N` | all | Only the first N current members: a quick run of under a minute, with no survivorship handling |
| `--refresh` | off | Ignore the download cache |
| `--insiders` | off | Also download SEC insider-trading data (large; cached for 30 days) |
| `--retrain-every 63` | 63 | Refit the model every N sessions out of sample (0 = use `--folds` equal blocks) |
| `--folds 5` | 5 | Walk-forward folds when `--retrain-every 0` |
| `--no-tune` | off | Skip the hyperparameter search inside each training window (faster) |
| `--holdout-start 2025-01-01` | 2025-01-01 | First date of the untouched holdout; `none` researches on everything |
| `--evaluate-holdout` | off | Score the saved production model on the holdout, log the evaluation, and stop |
| `--cost-model spread\|flat` | `spread` | Per-stock costs scaled by each stock's estimated bid-ask spread (the median stock pays `--cost-bps`), or a flat `--cost-bps` for all |
| `--cost-bps 10` | 10 | Cost per unit of weight traded (for the median stock with `spread`) |
| `--borrow-bps 0` | 0 | Annual fee for borrowing shorted stock |
| `--quantile 0.2` | 0.2 | Share of stocks bought (and shorted) |
| `--exit-quantile 0.4` | 0.4 | Buffer: keep a holding until it leaves this top share (set equal to `--quantile` for none) |
| `--smooth-days 3` | 3 | Rank on each stock's score averaged over this many sessions |
| `--construction quantile\|optimizer` | `quantile` | The production portfolio: quantile ranks, or the cost-aware, factor-neutral optimiser |
| `--max-rows 150000` | 150,000 | Cap on training rows per model fit, for speed |
| `--bootstrap-reps 2000` | 2,000 | Resamples for the bootstrap confidence intervals |
| `--skip-experiments` | off | Don't compare the alternative research setups (faster) |
| `--make-sample`, `--download-kaggle`, `--from-zip PATH`, `--overwrite` | | Prepare the sample or Kaggle data, then run on it |

A full live run takes about 4–5 minutes the first time and 2–3 minutes from cache (`--skip-experiments` saves about a minute). It writes:

| Path | Contents |
|---|---|
| `data/processed/` | Feature table, fundamentals, macro data, index membership, benchmark prices, data-quality report |
| `models/return_direction_model.joblib` | The production model, with its validation metrics and research setup |
| `reports/research/` | Walk-forward predictions, model comparison, calibration, feature importance, IC breakdown and decay, backtest, portfolio constructions, staggered starts, cost sensitivity, bootstrap intervals, overfitting statistics, attribution (home-made and Fama-French), setups compared, and `research_report.md` |
| `reports/research/holdout_log.jsonl` | One line per holdout evaluation (`--evaluate-holdout`) |
| `reports/live_signals/` | Each run's live ranking, scored by later runs once returns are known (the forward test) |

What the model predicts is set in `src/sp500_agent/config.py` (`PRODUCTION_SPEC`, `EXPERIMENT_SPECS`). The default scores each stock's probability of beating a randomly chosen index member over the next 5 trading days, after scaling for volatility (a ranked target). [FINDINGS.md](FINDINGS.md) explains the earlier yes/no setups.

**The holdout, step by step.** Research never sees data from `--holdout-start` on. When a setup is final, run `python run_pipeline.py --evaluate-holdout` once: it walks the saved model (same hyperparameters, same portfolio) through the holdout and logs the result. Each further evaluation is logged too, and the command tells you when the holdout has already been looked at: from then on it is no longer untouched, and only the forward test in `reports/live_signals/` is.

## Use the dashboard

`sp500-web` serves the dashboard and its API. Options: `--port`, `--host`, and `--reload` (restart on code changes).

| Page | What it shows |
|---|---|
| **Overview** | Model quality at a glance, the highest- and lowest-ranked stocks, the backtest against the S&P 500, sector tilt, rates and VIX |
| **Screener** | Every ranked stock with probability, stance, returns, volatility and fundamentals; sort by any column, filter by sector, probability and volatility |
| **Stock** (click any ticker) | Price chart with volume and 1M–Max ranges, percentile against other index members, SEC fundamentals history, live news and SEC filings, and a button that asks the agent for a brief |
| **Research agent** | The AI chat (below) |
| **Backtest** | Growth of $1 and drawdowns against the S&P 500, returns by prediction quintile, rolling rank IC, statistics, factor attribution |
| **Model lab** | The research setups compared, model comparison, AUC by fold, calibration, feature importance, single-signal ICs, research report download |
| **Data** | Sources and fetch times, survivorship coverage, quality checks, VIX and yields, recent index changes |

Press ⌘K (Ctrl+K on Windows and Linux) to search for a ticker or company from anywhere. The theme switch is at the bottom of the sidebar.

After re-running the pipeline, the dashboard needs a restart (or `POST /api/reload`) to pick up the new data.

## Use the research agent

The agent answers questions by calling research tools and citing their results.

- **Data and model:** stock snapshots, rankings, screens, comparisons, price history, sector summaries, SEC fundamentals history, market conditions, index changes, and the model's and backtest's own evidence.
- **Live lookups:** current headlines and SEC filings.

In the dashboard's **Research agent** page each tool call appears as it happens; click one to see the raw result. From the terminal:

```bash
sp500-chat                                                     # interactive; type 'reset' or 'quit'
sp500-chat "Compare NVDA and AMD on growth, valuation and recent filings"
sp500-chat --provider anthropic "Should I trust this model?"
```

Questions that work well:
- *Which stocks does the model rank highest, and how much should I trust it?*
- *How have AAPL's revenue and margins changed over the last year?*
- *Find low-volatility stocks with improving revenue growth.*
- *What's the market backdrop: VIX, rates and the yield curve?*
- *Which stocks joined or left the S&P 500 recently?*
- *What does the backtest say after costs, and where do the returns come from?*

**What it costs.** A typical question uses 5,000–15,000 tokens, which is a fraction of a cent on DeepSeek. The agent is research output for education: it describes evidence and limits but doesn't give buy or sell instructions, and it can still be wrong, so check its figures in the dashboard.

## Other interfaces

- **Markdown brief, no LLM:** `sp500-brief --ticker AAPL` prints a brief and saves it to `reports/`.
- **Streamlit app:** `streamlit run streamlit_app.py` is a lighter UI that needs no Node build.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Address already in use` | Another program has the port: `sp500-web --port 8010`, or `HOST_PORT=8010` with Docker. |
| `fundamentals: skipped. SEC EDGAR needs a User-Agent…` | Set `SEC_USER_AGENT` in `.env` with a contact email. |
| Yahoo download errors or missing tickers | Usually transient: re-run with `--refresh`. Delisted stocks are often unavailable; the data-quality report lists what's missing. |
| `web/dist not found, serving the API only` | Build the dashboard: `cd web && npm install && npm run build`. |
| Dashboard says "Run python run_pipeline.py first" | No research data yet in `data/processed/`; run the pipeline. |
| Agent answers "The AI agent could not answer…" | Check the LLM key in `.env` and your provider balance. Without a key, the chat uses the rule-based assistant. |
| A stock is "left out of today's ranking" | It had a daily move over 40% in the last 5 sessions, often a split or spin-off Yahoo hasn't adjusted yet. Its history, news and filings are still available. |
| Kaggle download fails with an SSL error | Download the ZIP in a browser and use `--from-zip`. |

## Development

```bash
pytest                                  # ~140 offline tests, ~25 s
sp500-web --reload                      # API with auto-restart
cd web && npm run dev                   # dashboard on :5173 with hot reload, proxying /api to :8000
cd web && npm run typecheck
python scripts/capture_screenshots.py --url http://127.0.0.1:8000   # regenerate docs/media (needs Chrome + pip install playwright)
```

GitHub Actions (`.github/workflows/ci.yml`) runs on every push:
1. the tests;
2. the dashboard build;
3. a Docker image build that starts the container on sample data and checks the API.
