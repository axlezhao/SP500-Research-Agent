from __future__ import annotations

import pandas as pd
import streamlit as st

from sp500_agent.chat import answer_message, ranking_answer
from sp500_agent.config import FEATURES_PATH, MODEL_PATH
from sp500_agent.features import load_features as read_features
from sp500_agent.formatting import fmt_pct, fmt_prob
from sp500_agent.model import load_model, score_latest


st.set_page_config(page_title="S&P 500 Research Agent", page_icon="📈", layout="wide")

GREETING = "Ask about a ticker like `NVDA`, compare rankings with `top 10 stocks`, or ask for the weakest names."


@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    return read_features() if FEATURES_PATH.exists() else pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_scoring_bundle():
    return load_model() if MODEL_PATH.exists() else None


@st.cache_data(show_spinner=False)
def load_scored_latest() -> pd.DataFrame:
    features = load_features()
    model = load_scoring_bundle()
    if features.empty or model is None:
        return pd.DataFrame()
    return score_latest(features, model)


def price_chart(ticker: str, days: int = 252) -> pd.DataFrame:
    features = load_features()
    chart = features[features["ticker"] == ticker].sort_values("date").tail(days)
    return chart[["date", "close"]].set_index("date")


def ask(question: str, scored: pd.DataFrame) -> None:
    answer = answer_message(question, scored)
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {"role": "assistant", "content": answer.text, "table": answer.table, "ticker": answer.ticker}
    )


def render_message(message: dict, scored: pd.DataFrame) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        table = message.get("table")
        if table is not None:
            st.dataframe(table, width="stretch", hide_index=True)
        ticker = message.get("ticker")
        match = scored[scored["ticker"] == ticker] if ticker else scored.iloc[0:0]
        if not match.empty:
            row = match.iloc[0]
            metric_cols = st.columns(4)
            metric_cols[0].metric("Model probability", fmt_prob(row["up_probability_5d"]))
            metric_cols[1].metric("Rank", f"{int(row['rank'])}/{len(scored)}")
            metric_cols[2].metric("20D return", fmt_pct(row.get("return_20d")))
            metric_cols[3].metric("Volatility (ann.)", fmt_pct(row.get("volatility_20d")))
            chart = price_chart(ticker)
            if not chart.empty:
                st.line_chart(chart, width="stretch")


st.title("S&P 500 Research Agent")
st.caption("Educational model output only. Not financial advice.")

features = load_features()
model = load_scoring_bundle()
scored = load_scored_latest()

if features.empty or model is None or scored.empty:
    st.error("Run `python run_pipeline.py` first so the app has features and a trained model.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": GREETING}]

with st.sidebar:
    st.subheader("Model Snapshot")
    st.metric("Stocks ranked", f"{len(scored):,}")
    st.metric("Feature rows", f"{len(features):,}")
    latest_date = scored["date"].max()
    st.metric("Latest price date", latest_date.date().isoformat() if pd.notna(latest_date) else "n/a")
    metrics = model.get("metrics", {})
    if "auc" in metrics:
        st.metric("Holdout ROC AUC", f"{metrics['auc']:.3f}", help="Time-ordered holdout. 0.500 means no skill.")
    st.divider()
    tickers = sorted(scored["ticker"].tolist())
    selected = st.selectbox("Ticker", tickers, index=tickers.index("NVDA") if "NVDA" in tickers else 0)
    if st.button("Analyze", width="stretch"):
        ask(selected, scored)

user_message = st.chat_input("Ask about a stock or ranking")
if user_message:
    ask(user_message, scored)

for message in st.session_state.messages:
    render_message(message, scored)

with st.expander("Current top-ranked stocks", expanded=False):
    st.dataframe(ranking_answer(scored, n=10).table, width="stretch", hide_index=True)
