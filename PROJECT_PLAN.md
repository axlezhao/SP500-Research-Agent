# Project Plan

## Phase 1: Baseline Agent

- Download dataset with KaggleHub.
- Load CSV tables.
- Build stock, financial, and news features.
- Train a baseline return-direction model.
- Generate ticker-level research briefs.

## Phase 2: Stronger Modeling

- Add walk-forward validation (a single time-ordered holdout with an embargo is in place).
- Add sector-relative features.
- Add backtesting.
- Add better sentiment with FinBERT or embeddings.

## Phase 3: Product Layer

- Add Streamlit dashboard. (done)
- Add conversational query interface. (rule-based version done; an LLM with tool calls is next)
- Add model explanations and portfolio simulation.

