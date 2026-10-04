from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from sp500_agent import charts
from sp500_agent.chat import answer_message
from sp500_agent.config import FEATURES_PATH, MODEL_PATH
from sp500_agent.formatting import company_label, fmt_pct, fmt_prob
from sp500_agent.llm_agent import configured_providers, create_agent
from sp500_agent.research_tools import ResearchData, ResearchToolkit

st.set_page_config(page_title="S&P 500 Research Agent", page_icon="📈", layout="wide")

GREETING = (
    "Ask about a stock (`How does NVDA look?`), a comparison (`Compare AAPL, MSFT and GOOGL`), "
    "a screen (`Low-volatility stocks the model likes`), or the evidence (`Should I trust this model?`)."
)
AGENT_MODE, RULE_MODE = "AI agent", "Rule-based"
PROVIDER_LABELS = {"deepseek": "DeepSeek", "anthropic": "Claude"}


@st.cache_resource(show_spinner="Loading data and model...")
def load_research_data() -> ResearchData | None:
    if not FEATURES_PATH.exists() or not MODEL_PATH.exists():
        return None
    return ResearchData.load()


def chart_mode() -> str:
    theme = getattr(st.context, "theme", None)
    return "dark" if theme is not None and getattr(theme, "type", None) == "dark" else "light"


def price_history_frame(features: pd.DataFrame, ticker: str, days: int = 252) -> pd.DataFrame:
    return features.loc[features["ticker"] == ticker, ["date", "close"]].sort_values("date").tail(days)


# -- chat -----------------------------------------------------------------------------------------
def get_agent(provider: str):
    key = f"agent::{provider}"
    if key not in st.session_state:
        st.session_state[key] = create_agent(ResearchToolkit(data), provider=provider)
    return st.session_state[key]


def reset_conversation() -> None:
    st.session_state.messages = [{"role": "assistant", "content": GREETING}]
    for key in [k for k in st.session_state if str(k).startswith("agent::")]:
        del st.session_state[key]


def ask_rule_based(question: str) -> dict:
    answer = answer_message(question, data.scored)
    return {"role": "assistant", "content": answer.text, "table": answer.table, "ticker": answer.ticker}


def ask_agent(question: str, provider: str) -> dict:
    agent = get_agent(provider)
    with st.status(f"Researching with {PROVIDER_LABELS[provider]}...", expanded=False) as status:
        try:
            reply = agent.ask(question)
        except Exception as exc:  # network, auth or rate-limit problems
            status.update(label="The agent hit an error", state="error")
            return {"role": "assistant", "content": f"The AI agent could not answer: `{type(exc).__name__}: {exc}`"}
        status.update(label=f"Used {len(reply.tool_calls)} tool call(s)", state="complete")
    return {
        "role": "assistant",
        "content": reply.text or "(No answer.)",
        "tool_calls": [vars(call) for call in reply.tool_calls],
        "meta": f"{PROVIDER_LABELS[reply.provider]} · {reply.model} · {reply.usage.get('input_tokens', 0):,} in / {reply.usage.get('output_tokens', 0):,} out tokens",
    }


def render_tool_calls(calls: list[dict]) -> None:
    with st.expander(f"🔧 {len(calls)} tool call(s)"):
        for call in calls:
            args = ", ".join(f"{k}={v!r}" for k, v in call["arguments"].items())
            st.markdown(f"**`{call['name']}({args})`**" + (f" — error: {call['error']}" if call["error"] else ""))
            if call["result"] is not None:
                st.json(call["result"], expanded=False)
    for call in calls:
        result = call.get("result") or {}
        if call["name"] == "price_history" and result.get("series"):
            history = pd.DataFrame(result["series"]).assign(date=lambda d: pd.to_datetime(d["date"]))
            st.altair_chart(charts.price_chart(history, result["ticker"], chart_mode()), width="stretch")


def escape_dollars(text: str) -> str:
    # Streamlit renders text between two $ signs as LaTeX, which mangles prices like "$119 → $100".
    return re.sub(r"(?<!\\)\$", r"\\$", text)


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(escape_dollars(message["content"]))
        if message.get("tool_calls"):
            render_tool_calls(message["tool_calls"])
        table = message.get("table")
        if table is not None:
            st.dataframe(table, width="stretch", hide_index=True)
        ticker = message.get("ticker")
        match = data.scored[data.scored["ticker"] == ticker] if ticker else data.scored.iloc[0:0]
        if not match.empty:
            row = match.iloc[0]
            cols = st.columns(4)
            cols[0].metric("Model probability", fmt_prob(row["model_probability"]))
            cols[1].metric("Rank", f"{int(row['rank'])}/{len(data.scored)}")
            cols[2].metric("20D return", fmt_pct(row.get("return_20d")))
            cols[3].metric("Volatility (ann.)", fmt_pct(row.get("volatility_20d")))
            st.altair_chart(charts.price_chart(price_history_frame(data.features, ticker), ticker, chart_mode()), width="stretch")
        if message.get("meta"):
            st.caption(message["meta"])


# -- page -----------------------------------------------------------------------------------------
st.title("S&P 500 Research Agent")
st.caption("A data science, quant research and AI agent project. Educational output only, not financial advice.")

data = load_research_data()
if data is None:
    st.error("Run `python run_pipeline.py` first so the app has features, a trained model and research results.")
    st.stop()

artifacts = data.artifacts
metrics = data.bundle.get("metrics", {})
if "messages" not in st.session_state:
    reset_conversation()

providers = configured_providers()
with st.sidebar:
    st.subheader("Snapshot")
    st.metric("As of", data.scored["date"].max().date().isoformat())
    cols = st.columns(2)
    cols[0].metric("Stocks", f"{len(data.scored):,}")
    cols[1].metric("Walk-forward AUC", f"{metrics['auc_mean']:.3f}" if "auc_mean" in metrics else "n/a", help="Out-of-sample. 0.500 means no skill.")
    source_label = {"live": "Live public data", "kaggle": "Kaggle dataset", "sample": "Synthetic sample"}.get(data.source, data.source)
    st.caption(f"Data: {source_label} · Model: {data.bundle.get('model_name', 'n/a')}")
    st.divider()

    st.subheader("Assistant")
    mode = st.segmented_control("Mode", [AGENT_MODE, RULE_MODE], default=AGENT_MODE if providers else RULE_MODE, label_visibility="collapsed") or RULE_MODE
    provider = None
    if mode == AGENT_MODE:
        if not providers:
            st.info("Add `DEEPSEEK_API_KEY` or `ANTHROPIC_API_KEY` to `.env` (see `.env.example`) to enable the AI agent.")
            mode = RULE_MODE
        else:
            provider = st.selectbox("Provider", providers, format_func=lambda p: PROVIDER_LABELS.get(p, p)) if len(providers) > 1 else providers[0]
            st.caption(f"{PROVIDER_LABELS[provider]} · `{get_agent(provider).model}`")
    if st.button("New conversation", width="stretch"):
        reset_conversation()
    st.divider()
    tickers = sorted(data.scored["ticker"].tolist())
    selected = st.selectbox("Quick brief", tickers, index=tickers.index("NVDA") if "NVDA" in tickers else 0)
    analyze = st.button("Analyze", width="stretch")

chat_tab, backtest_tab, model_tab, data_tab = st.tabs(["💬 Research agent", "📊 Backtest", "🔬 Model lab", "🗂 Data & screener"])

with chat_tab:
    for message in st.session_state.messages:
        render_message(message)
    question = st.chat_input("Ask about a stock, a sector, the model or the backtest")
    if analyze:
        question = f"Give me a research brief on {selected}." if mode == AGENT_MODE else selected
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        render_message(st.session_state.messages[-1])
        reply = ask_agent(question, provider) if mode == AGENT_MODE else ask_rule_based(question)
        st.session_state.messages.append(reply)
        st.rerun()

with backtest_tab:
    summary = artifacts.get("backtest_summary")
    returns = artifacts.get("backtest_returns")
    if summary is None or returns is None:
        st.info("No backtest yet. Run `python run_pipeline.py`.")
    else:
        stats = summary.set_index("strategy")
        config = artifacts.get("backtest_config") or {}
        st.markdown(
            f"Out-of-sample predictions from walk-forward validation, {returns['date'].min().date()} to {returns['date'].max().date()} "
            f"({len(returns)} rebalances). Every {config.get('holding_days', 5)} sessions: rank stocks, enter one session later, hold "
            f"{config.get('holding_days', 5)} sessions. Costs {config.get('cost_bps', 10):g} bps per unit traded."
        )
        cols = st.columns(4)
        cols[0].metric("Long top 20%", fmt_pct(stats.loc["long_only", "total_return"]), f"{(stats.loc['long_only', 'total_return'] - stats.loc['benchmark', 'total_return']) * 100:+.1f} pts vs benchmark")
        cols[1].metric("Long-short", fmt_pct(stats.loc["long_short", "total_return"]))
        cols[2].metric("Long-short Sharpe", f"{stats.loc['long_short', 'sharpe']:.2f}")
        cols[3].metric("Rank IC (mean)", f"{stats.loc['long_short', 'ic_mean']:.3f}", help="Average rank correlation between prediction and realised return at each rebalance.")
        left, right = st.columns(2)
        with left:
            st.markdown("**Growth of $1, after costs**")
            st.altair_chart(charts.equity_curve(returns, chart_mode()), width="stretch")
        with right:
            st.markdown("**Drawdown from peak**")
            st.altair_chart(charts.drawdown_chart(returns, chart_mode()), width="stretch")
        left, right = st.columns(2)
        with left:
            st.markdown("**Return by prediction quintile**")
            quantiles = artifacts.get("quantile_returns")
            if quantiles is not None and not quantiles.empty:
                st.altair_chart(charts.quintile_chart(quantiles, chart_mode()), width="stretch")
        with right:
            st.markdown("**Strategy statistics**")
            headers = {"total_return": "Total return", "cagr": "CAGR", "ann_volatility": "Volatility", "sharpe": "Sharpe", "max_drawdown": "Max drawdown", "hit_rate": "Hit rate", "avg_turnover": "Turnover"}
            table = stats[list(headers)].rename(index=charts.STRATEGY_LABELS, columns=headers)
            percent = ["Total return", "CAGR", "Volatility", "Max drawdown", "Hit rate"]
            st.dataframe(table.style.format({c: "{:.1%}" for c in percent} | {"Sharpe": "{:.2f}", "Turnover": "{:.2f}"}), width="stretch")
        st.caption(
            "Caveats: the data holds today's index members only (survivorship bias flatters long strategies); "
            "costs are a flat charge with no market impact or short-borrow fees."
        )

with model_tab:
    comparison = artifacts.get("model_comparison")
    if comparison is None:
        st.info("No model research yet. Run `python run_pipeline.py`.")
    else:
        st.markdown(
            "Three models compared with expanding-window walk-forward validation: each fold trains on earlier dates only, "
            "with a 5-session gap so no training label overlaps the test period."
        )
        display = comparison.assign(model=comparison["model"].map(charts.MODEL_LABELS).fillna(comparison["model"]))
        st.dataframe(
            display[["model", "auc_mean", "auc_std", "accuracy", "baseline_accuracy", "brier", "ic_mean", "ic_tstat"]],
            width="stretch",
            hide_index=True,
            column_config={
                "model": "Model",
                "auc_mean": st.column_config.NumberColumn("AUC", format="%.3f", help="Mean over folds. 0.5 = no skill."),
                "auc_std": st.column_config.NumberColumn("AUC sd", format="%.3f"),
                "accuracy": st.column_config.NumberColumn("Accuracy", format="%.3f"),
                "baseline_accuracy": st.column_config.NumberColumn("Baseline", format="%.3f", help="Always predicting the more common direction."),
                "brier": st.column_config.NumberColumn("Brier", format="%.4f", help="Mean squared error of the probabilities; lower is better."),
                "ic_mean": st.column_config.NumberColumn("Mean IC", format="%.4f"),
                "ic_tstat": st.column_config.NumberColumn("IC t-stat", format="%.2f"),
            },
        )
        left, right = st.columns(2)
        with left:
            st.markdown("**AUC by fold** (dashed line = no skill)")
            folds = artifacts.get("fold_metrics")
            if folds is not None:
                st.altair_chart(charts.fold_auc_chart(folds.assign(test_start=pd.to_datetime(folds["test_start"]), test_end=pd.to_datetime(folds["test_end"])), chart_mode()), width="stretch")
        with right:
            st.markdown(f"**Calibration of {charts.MODEL_LABELS.get(data.bundle.get('model_name'), data.bundle.get('model_name'))}** (dashed = perfect)")
            calibration = artifacts.get("calibration")
            if calibration is not None:
                st.altair_chart(charts.calibration_chart(calibration, chart_mode()), width="stretch")
        left, right = st.columns(2)
        with left:
            st.markdown("**Permutation importance** (last fold, unseen data)")
            importance = artifacts.get("feature_importance")
            if importance is not None and not importance.empty:
                st.altair_chart(charts.importance_chart(importance, mode=chart_mode()), width="stretch")
        with right:
            st.markdown("**Single-signal information coefficients**")
            signals = artifacts.get("signal_ic")
            if signals is not None:
                st.dataframe(
                    signals[["signal", "ic_mean", "ic_tstat", "ic_positive_share"]],
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "signal": "Signal",
                        "ic_mean": st.column_config.NumberColumn("Mean IC", format="%.4f"),
                        "ic_tstat": st.column_config.NumberColumn("t-stat", format="%.2f", help="From non-overlapping dates. |t| > 2 is the usual bar."),
                        "ic_positive_share": st.column_config.NumberColumn("Days IC > 0", format="percent"),
                    },
                )
        if artifacts.get("report"):
            st.download_button("Download the full research report (Markdown)", artifacts["report"], file_name="research_report.md", mime="text/markdown")

with data_tab:
    quality = data.quality
    if quality:
        st.markdown("**Data sources and quality**")
        prices_q = quality.get("prices", {})
        sources = [{"Source": "Yahoo Finance", "Provides": "Daily prices", "Coverage": f"{prices_q.get('tickers', 'n/a')} stocks, {prices_q.get('start')} to {prices_q.get('end')}"}]
        if "membership" in quality:
            m = quality["membership"]
            sources.append({"Source": "Wikipedia", "Provides": "Index membership", "Coverage": f"{m['tickers_ever_in_index']} stocks ever in the index; {m['unresolved_changes']} of {m['changes_used']} changes unresolved"})
        if "fundamentals" in quality:
            f = quality["fundamentals"]
            sources.append({"Source": "SEC EDGAR", "Provides": "Point-in-time fundamentals", "Coverage": f"{f['tickers_with_data']} stocks ({fmt_pct(f['share_of_priced_tickers'])}); median {f['median_days_since_last_filing']} days since last filing"})
        if "macro" in quality:
            sources.append({"Source": "FRED", "Provides": "VIX, Treasury yields", "Coverage": "through " + max(quality["macro"].values())})
        fetched = quality.get("fetched_at", {})
        for row in sources:
            key = {"Yahoo Finance": "prices", "Wikipedia": "universe", "SEC EDGAR": "fundamentals", "FRED": "macro"}[row["Source"]]
            row["Last fetched (UTC)"] = (fetched.get(key) or "")[:16].replace("T", " ") or "n/a"
        st.dataframe(pd.DataFrame(sources), width="stretch", hide_index=True)
        notes = []
        if "survivorship" in quality:
            s = quality["survivorship"]
            notes.append(
                f"Survivorship: {s['former_members_with_prices']} of {s['former_members']} former index members have usable prices "
                f"({s.get('dropped_no_matching_history', 0)} dropped because the symbol's history doesn't match their membership, e.g. a reused ticker)."
            )
        if prices_q.get("extreme_daily_moves"):
            examples = ", ".join(f"{e['ticker']} {e['move']:+.0%} on {e['date']}" for e in prices_q.get("extreme_examples", [])[:3])
            notes.append(f"{prices_q['extreme_daily_moves']} daily moves above 40% (largest: {examples}). Most are real events; some are corporate actions the price source hasn't adjusted.")
        for note in notes:
            st.caption(note)

    if data.macro is not None and not data.macro.empty:
        st.markdown("**Market conditions** (FRED)")
        recent_macro = data.macro[data.macro["date"] >= data.macro["date"].max() - pd.Timedelta(days=5 * 365)]
        left, right = st.columns(2)
        with left:
            st.altair_chart(charts.series_chart(recent_macro, "vix", "VIX", ".1f", chart_mode()), width="stretch")
        with right:
            st.altair_chart(charts.series_chart(recent_macro, "term_spread", "10y minus 3m yield (pts)", ".2f", chart_mode(), reference=0.0), width="stretch")
        st.caption("Below the dashed line the yield curve is inverted, which has often preceded recessions.")

    eda = artifacts.get("eda")
    if eda:
        cols = st.columns(4)
        cols[0].metric("Rows", f"{eda['rows'] / 1e6:.2f}M" if eda["rows"] >= 1e6 else f"{eda['rows']:,}")
        cols[1].metric("Stocks", f"{eda['tickers']:,}")
        cols[2].metric("Trading days", f"{eda['trading_days']:,}")
        cols[3].metric("Target base rate", fmt_pct(eda.get("target_rate", eda["up_rate_5d"])), help=f"How often the predicted event happens: {eda.get('target', 'close higher in 5 days')}.")
        fr = eda["forward_return_5d"]
        st.caption(
            f"{eda['start']} to {eda['end']}. 5-day forward returns: mean {fmt_pct(fr['mean'])}, std {fmt_pct(fr['std'])}, skew {fr['skew']:.2f}, "
            f"excess kurtosis {fr['excess_kurtosis']:.2f}."
            + (f" Rows with recent news: {fmt_pct(eda['share_of_rows_with_recent_news'])}." if eda.get("share_of_rows_with_recent_news") else "")
        )
    st.markdown("**Screener** (latest date)")
    for ticker, reason in data.excluded.items():
        st.caption(f"{ticker} is left out of the ranking: {reason}.")
    screen = data.scored.copy()
    screen["company"] = screen.apply(company_label, axis=1)
    sectors = sorted(screen["sector"].dropna().unique()) if "sector" in screen else []
    chosen = st.multiselect("Sectors", sectors, placeholder="All sectors")
    if chosen:
        screen = screen[screen["sector"].isin(chosen)]
    columns = [c for c in ["rank", "ticker", "company", "sector", "model_probability", "return_5d", "return_20d", "momentum_60d", "volatility_20d", "sentiment_20d", "market_cap", "pe_ratio"] if c in screen.columns]
    st.dataframe(
        screen[columns],
        width="stretch",
        hide_index=True,
        column_config={
            "rank": "Rank",
            "ticker": "Ticker",
            "company": "Company",
            "sector": "Sector",
            "model_probability": st.column_config.ProgressColumn("Model probability", min_value=0.0, max_value=1.0, format="percent"),
            "return_5d": st.column_config.NumberColumn("5D return", format="percent"),
            "return_20d": st.column_config.NumberColumn("20D return", format="percent"),
            "momentum_60d": st.column_config.NumberColumn("60D momentum", format="percent"),
            "volatility_20d": st.column_config.NumberColumn("Volatility (ann.)", format="percent"),
            "sentiment_20d": st.column_config.NumberColumn("News sentiment", format="%.2f"),
            "market_cap": st.column_config.NumberColumn("Market cap", format="compact"),
            "pe_ratio": st.column_config.NumberColumn("P/E", format="%.1f"),
        },
    )
