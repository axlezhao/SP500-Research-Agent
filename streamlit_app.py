from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st

from src.sp500_agent.config import MODEL_DIR, PROCESSED_DIR
from src.sp500_agent.model import load_model, score_latest


st.set_page_config(page_title="S&P 500 Research Agent", page_icon="📈", layout="wide")

POPULAR_ALIASES = {
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "NVIDIA": "NVDA",
    "TESLA": "TSLA",
    "AMAZON": "AMZN",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "META": "META",
    "FACEBOOK": "META",
    "JPMORGAN": "JPM",
    "JP MORGAN": "JPM",
    "EXXON": "XOM",
}


def fmt_pct(value) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def fmt_num(value) -> str:
    if pd.isna(value):
        return "n/a"
    value = float(value)
    if abs(value) >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    return f"{value:.2f}"


@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "research_features.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["date"])


@st.cache_resource(show_spinner=False)
def load_scoring_bundle():
    path = MODEL_DIR / "return_direction_model.joblib"
    if not path.exists():
        return None
    return load_model()


@st.cache_data(show_spinner=False)
def load_scored_latest() -> pd.DataFrame:
    features = load_features()
    model = load_scoring_bundle()
    if features.empty or model is None:
        return pd.DataFrame()
    scored = score_latest(features, model).reset_index(drop=True)
    scored["rank"] = scored.index + 1
    return scored


def stance(probability: float) -> str:
    if probability >= 0.60:
        return "constructive"
    if probability <= 0.40:
        return "cautious"
    return "neutral"


def company_label(row: pd.Series) -> str:
    for col in ["displayname", "longname", "shortname", "company_name"]:
        if col in row and pd.notna(row[col]):
            return str(row[col])
    return str(row["ticker"])


def find_ticker(message: str, scored: pd.DataFrame) -> str | None:
    upper = message.upper()
    valid = set(scored["ticker"].astype(str).str.upper())

    # Exact ticker tokens win first. This prevents single-letter tickers like D
    # from matching inside tickers such as NVDA.
    for token in re.findall(r"\b[A-Z]{1,5}\b", upper):
        if token in valid:
            return token

    for name, ticker in POPULAR_ALIASES.items():
        if re.search(rf"\b{re.escape(name)}\b", upper) and ticker in valid:
            return ticker

    for _, row in scored.iterrows():
        ticker = str(row["ticker"]).upper()
        for col in ["displayname", "longname", "shortname", "company_name"]:
            if col in row and pd.notna(row[col]):
                company_name = str(row[col]).upper().strip()
                if len(company_name) >= 3 and re.search(rf"\b{re.escape(company_name)}\b", upper):
                    return ticker

    return None


def ranking_answer(scored: pd.DataFrame, n: int = 10, ascending: bool = False) -> tuple[str, pd.DataFrame]:
    ranked = scored.sort_values("up_probability_5d", ascending=ascending).head(n).copy()
    ranked["company"] = ranked.apply(company_label, axis=1)
    ranked["model_probability"] = ranked["up_probability_5d"].map(lambda x: f"{x:.1%}")
    ranked["5d_return"] = ranked["return_5d"].map(fmt_pct)
    ranked["20d_return"] = ranked["return_20d"].map(fmt_pct)
    ranked["20d_volatility"] = ranked["volatility_20d"].map(fmt_pct)
    table = ranked[["rank", "ticker", "company", "sector", "model_probability", "5d_return", "20d_return", "20d_volatility"]]
    direction = "lowest" if ascending else "highest"
    answer = (
        f"Here are the {n} stocks with the {direction} model-ranked 5-day upward-move probability. "
        "This is a research signal, not a buy or sell recommendation."
    )
    return answer, table


def ticker_answer(ticker: str, scored: pd.DataFrame) -> tuple[str, pd.Series]:
    row = scored[scored["ticker"] == ticker].iloc[0]
    view = stance(float(row["up_probability_5d"]))
    name = company_label(row)
    answer = f"""### {ticker}: {name}

The model view is **{view}**. {ticker} ranks **{int(row['rank'])} of {len(scored)}** by predicted 5-day upward-move probability.

**Model probability:** {row['up_probability_5d']:.1%}  
**Latest close:** {fmt_num(row.get('close'))}  
**5-day return:** {fmt_pct(row.get('return_5d'))}  
**20-day return:** {fmt_pct(row.get('return_20d'))}  
**20-day volatility:** {fmt_pct(row.get('volatility_20d'))}  
**20-day news sentiment:** {float(row.get('sentiment_20d', 0)):.2f}  
**Sector:** {row.get('sector', 'n/a')}

I would present this as an educational research brief. It should not be treated as financial advice or a direct instruction to buy or sell.
"""
    return answer, row


def answer_message(message: str, scored: pd.DataFrame):
    lower = message.lower()
    top_match = re.search(r"top\s+(\d+)", lower)
    n = min(max(int(top_match.group(1)), 1), 25) if top_match else 10

    if any(word in lower for word in ["top", "rank", "strongest", "best"]):
        return (*ranking_answer(scored, n=n, ascending=False), None)
    if any(word in lower for word in ["worst", "weakest", "lowest"]):
        return (*ranking_answer(scored, n=n, ascending=True), None)

    ticker = find_ticker(message, scored)
    if ticker:
        answer, row = ticker_answer(ticker, scored)
        return answer, None, row

    fallback = (
        "Ask me about a ticker, for example `NVDA`, `AAPL`, or `MSFT`, "
        "or ask for `top 10 stocks` / `weakest 10 stocks`."
    )
    return fallback, None, None


def price_chart(ticker: str, days: int = 252) -> pd.DataFrame:
    features = load_features()
    chart = features[features["ticker"] == ticker].sort_values("date").tail(days)
    return chart[["date", "close"]].set_index("date")


st.title("S&P 500 Research Agent")
st.caption("Educational model output only. Not financial advice.")

features = load_features()
model = load_scoring_bundle()
scored = load_scored_latest()

if features.empty or model is None or scored.empty:
    st.error("Run `python run_pipeline.py` first so the app has features and a trained model.")
    st.stop()

with st.sidebar:
    st.subheader("Model Snapshot")
    st.metric("Stocks", f"{scored['ticker'].nunique():,}")
    st.metric("Feature rows", f"{len(features):,}")
    latest_date = features["date"].max()
    st.metric("Latest price date", latest_date.date().isoformat() if pd.notna(latest_date) else "n/a")
    st.divider()
    selected = st.selectbox("Ticker", scored["ticker"].sort_values().tolist(), index=sorted(scored["ticker"].tolist()).index("NVDA") if "NVDA" in set(scored["ticker"]) else 0)
    if st.button("Analyze", use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": selected})

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask about a ticker like `NVDA`, compare rankings with `top 10 stocks`, or ask for the weakest names."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_message = st.chat_input("Ask about a stock or ranking")
if user_message:
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    answer, table, row = answer_message(user_message, scored)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
        if table is not None:
            st.dataframe(table, use_container_width=True, hide_index=True)
        if row is not None:
            ticker = str(row["ticker"])
            metric_cols = st.columns(4)
            metric_cols[0].metric("Model probability", f"{row['up_probability_5d']:.1%}")
            metric_cols[1].metric("Rank", f"{int(row['rank'])}/{len(scored)}")
            metric_cols[2].metric("20D return", fmt_pct(row.get("return_20d")))
            metric_cols[3].metric("20D volatility", fmt_pct(row.get("volatility_20d")))
            chart = price_chart(ticker)
            if not chart.empty:
                st.line_chart(chart, use_container_width=True)

with st.expander("Current top-ranked stocks", expanded=False):
    _, table = ranking_answer(scored, n=10, ascending=False)
    st.dataframe(table, use_container_width=True, hide_index=True)
