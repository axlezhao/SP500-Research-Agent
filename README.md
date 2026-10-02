# S&P 500 AI Research Agent

One project, three layers, built on S&P 500 prices, company data and news from [Kaggle](https://www.kaggle.com/datasets/sadiqguru/s-and-p-500-stock-data-along-with-financials-and-news):

| Layer | What it does | Where |
|---|---|---|
| **Data science** | Feature engineering, three models compared with walk-forward validation, calibration, permutation importance, an auto-generated research report | `features.py`, `model.py`, `validation.py`, `research.py` |
| **Quant research** | Out-of-sample portfolio backtest (long-only, long-short, benchmark) with costs and next-day execution, quintile spreads, information coefficients | `backtest.py` |
| **AI agent** | An LLM (DeepSeek or Claude) that answers research questions by calling 11 tools over the data, model and backtest | `research_tools.py`, `llm_agent.py` |

Everything is surfaced in a Streamlit app with four tabs: the agent chat, the backtest, the model lab, and a data screener.

This project is for education and research only. It is not financial advice.

## Quick start

```bash
pip install -r requirements.txt
python run_pipeline.py --make-sample      # synthetic data, no Kaggle account needed
cp .env.example .env                       # then add DEEPSEEK_API_KEY and/or ANTHROPIC_API_KEY
streamlit run streamlit_app.py
```

`requirements.txt` pins the versions the project was last tested with and installs the project in editable mode, so `import sp500_agent` works anywhere. For a looser install: `pip install -e ".[app,llm,kaggle,dev]"`.

The sample data is a random walk on purpose: no model should find an edge in it, which makes it a good test that the evaluation isn't leaking.

## Getting the real data

Set up Kaggle credentials (Kaggle account settings → Create API token → save `kaggle.json` to `~/.kaggle/kaggle.json`), then download into `data/raw/kaggle_sp500_dataset/` and run everything:

```bash
python run_pipeline.py --download-kaggle
```

If KaggleHub fails with an SSL error from `storage.googleapis.com`, download the ZIP in your browser instead:

```bash
python run_pipeline.py --from-zip ~/Downloads/archive.zip
```

If you imported the data with an older version of this project, the per-ticker price files were flattened into `data/raw/`. Remove them and re-import with `--overwrite`.

Pipeline options: `--folds` (walk-forward folds, default 5), `--cost-bps` (backtest cost per unit traded, default 10), `--quantile` (share held long and short, default 0.2), `--max-rows` (training-row cap per fit, for speed).

## 1. Data science

**Features** (`features.py`). Every feature uses only information available at the close of its date:

- Returns over 5 and 20 days, 60-day momentum, distance from the 50-day moving average, annualised 20-day volatility, volume relative to its 20-day average.
- Cross-sectional percentile ranks and the 20-day return relative to the sector median, so the model learns which stocks are relatively strong rather than whether the whole market rose.
- News: daily article count and sentiment. Timezone-aware timestamps are converted to New York time; news at or after 4pm counts toward the next session; weekend and holiday news moves to the next trading day. 20-day sentiment is averaged over articles only, so a quiet period reads as unknown rather than neutral.
- Fundamentals (market cap, P/E, margins) are a single current snapshot, so they are shown in briefs but **not** used for training: attaching today's values to past rows would leak the future.

**Target.** Whether the close is higher 5 trading days later. The latest 5 rows of each stock have no outcome yet; they are excluded from training and are exactly the rows used for current predictions.

**Walk-forward validation** (`validation.py`). Expanding-window folds: each trains only on dates before its test block, with a 5-session gap so no training label overlaps a test label. A random train/test split would put neighbouring days of the same stock on both sides; on the random-walk sample data it reported ROC AUC ≈ 0.69 where the true answer is 0.50.

**Model comparison.** Logistic regression, random forest and gradient boosting are scored on the same folds by AUC, accuracy against an always-predict-the-majority baseline, Brier score, and information coefficient (IC: daily rank correlation between prediction and realised return). IC t-stats use non-overlapping dates only, since daily ICs on 5-day returns share most of their window. The best model by mean AUC is refit on all labelled data and saved.

**Diagnostics.** A calibration table (do 60% predictions rise 60% of the time?), permutation importance on the last fold's unseen data, and the IC of every single feature next to the model's, which shows whether the model adds anything over its best input.

All of it is written to `reports/research/`, including a readable `research_report.md`.

## 2. Quant research

`backtest.py` turns the out-of-sample predictions into portfolios:

- Every 5 sessions, rank stocks by predicted probability. Enter one session after the signal (trading at the signal's own close would be optimistic) and hold 5 sessions, so holding periods never overlap.
- **Long-only** buys the top 20%, **long-short** also shorts the bottom 20%, and the **benchmark** holds every stock equally.
- Costs are charged on every unit of weight traded, from the drifted weights of the previous period.
- Reported: total return, CAGR, volatility, Sharpe, maximum drawdown, hit rate, turnover, the long-only information ratio against the benchmark, mean return by prediction quintile, and IC at each rebalance.

Caveats: the dataset contains today's index members only (survivorship bias flatters every long strategy); costs ignore market impact and borrow fees; three models were compared, so the winner's scores are slightly optimistic.

## 3. AI agent

`llm_agent.py` runs a tool-calling loop. The model decides which of these tools to call, often several at once, and answers from their results:

`dataset_overview`, `search_companies`, `stock_snapshot`, `rank_stocks`, `screen_stocks`, `compare_stocks`, `price_history`, `recent_news`, `sector_summary`, `model_performance`, `backtest_results`

The tools (`research_tools.py`) are plain Python returning JSON. They validate arguments and return errors the model can recover from, such as an unknown ticker with suggested alternatives. The system prompt tells the agent to ground every number in a tool result, state the data's as-of date, explain the model's weak edge honestly, and not give buy/sell instructions.

**Providers.** Put a key in `.env` (git-ignored; see `.env.example`):

| Provider | Key | Default model | Override |
|---|---|---|---|
| DeepSeek (OpenAI-compatible API) | `DEEPSEEK_API_KEY` | `deepseek-v4-pro` | `DEEPSEEK_MODEL=deepseek-flash` |
| Claude (Anthropic API) | `ANTHROPIC_API_KEY` | `claude-opus-5-5`, medium effort | `CLAUDE_AGENT_MODEL`, `CLAUDE_AGENT_EFFORT` |

With both keys set, DeepSeek is used unless `LLM_PROVIDER=anthropic`. The Claude backend caches the system prompt and tools across turns, and enables Anthropic's server-side fallback, which retries a request on a recommended fallback model if a safety classifier declines it. Conversation history is append-only in both backends.

Chat in the terminal:

```bash
sp500-chat                                   # interactive
sp500-chat "Which tech stocks does the model rank highest, and should I trust it?"
```

Or use the **Research agent** tab in the Streamlit app, which shows each tool call and its result and draws price charts. Without a key the app falls back to a rule-based assistant (ticker briefs and rankings).

A one-off markdown brief without an LLM: `sp500-brief --ticker AAPL` (written to `reports/`).

## Tests

```bash
pytest
```

The suite runs offline in about ten seconds:

- **Leakage check:** on random-walk data the walk-forward AUC must stay close to 0.5.
- **Folds:** the walk-forward folds must be ordered and embargoed.
- **Backtest:** checked against a perfect signal, costs and the benchmark arithmetic.
- **Tools:** every tool must return strict JSON.
- **Agent loop:** scripted fake clients test both providers (tool execution, error handling, append-only history, the Claude request shape), so no API key is needed.

## Project layout

```
run_pipeline.py            data → features → walk-forward research → backtest → model + report
streamlit_app.py           four-tab app: agent chat, backtest, model lab, data & screener
src/sp500_agent/
  data_loader.py           find and normalise the price, fundamentals and news files
  features.py              feature engineering (time-series and cross-sectional)
  model.py                 model zoo, final fit, scoring the latest date
  validation.py            walk-forward folds, model comparison, IC, calibration, importance
  backtest.py              portfolio backtest and performance statistics
  research.py              orchestrates a research run, saves artifacts, writes the report
  research_tools.py        the agent's tools
  llm_agent.py             DeepSeek and Claude tool-calling agents, terminal chat
  chat.py, agent.py        rule-based chat and markdown briefs (no LLM needed)
  charts.py                Altair charts for the app
tests/                     pytest suite
```
