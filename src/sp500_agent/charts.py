"""Altair chart builders for the Streamlit dashboard.

Colours come from a validated categorical palette (first three slots pass colour-blind checks
for every pair in light and dark mode); identity is never colour-alone: every multi-series chart
has a legend and tooltips, and the app shows the numbers as a table next to each chart.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

PALETTE = {
    "light": {"series": ["#2a78d6", "#eb6834", "#1baf7a"], "muted": "#8a8984"},
    "dark": {"series": ["#3987e5", "#d95926", "#199e70"], "muted": "#8f8e88"},
}
STRATEGY_LABELS = {"long_only": "Long top 20%", "long_short": "Long-short", "benchmark": "Benchmark"}
MODEL_LABELS = {"logistic_regression": "Logistic regression", "ridge": "Ridge regression", "random_forest": "Random forest", "gradient_boosting": "Gradient boosting", "composite": "Anomaly composite"}
HEIGHT = 300


def _colors(mode: str) -> dict:
    return PALETTE["dark" if mode == "dark" else "light"]


def _series_scale(domain: list[str], mode: str) -> alt.Scale:
    # Colour follows the entity: each name keeps its slot whatever else is shown.
    return alt.Scale(domain=domain, range=_colors(mode)["series"][: len(domain)])


def _legend() -> alt.Legend:
    return alt.Legend(title=None, orient="bottom", columns=2, labelLimit=160, symbolStrokeWidth=3)


def _multi_line(df: pd.DataFrame, x: str, y: str, series: str, domain: list[str], y_title: str, y_format: str, mode: str) -> alt.Chart:
    """Lines with a vertical crosshair and a tooltip listing every series at the hovered date."""
    hover = alt.selection_point(fields=[x], nearest=True, on="pointerover", empty=False, clear="pointerout")
    base = alt.Chart(df).encode(x=alt.X(f"{x}:T", title=None))
    lines = base.mark_line(strokeWidth=2).encode(
        y=alt.Y(f"{y}:Q", title=y_title, scale=alt.Scale(zero=False), axis=alt.Axis(format=y_format, grid=True)),
        color=alt.Color(f"{series}:N", scale=_series_scale(domain, mode), legend=_legend()),
    )
    wide = df.pivot_table(index=x, columns=series, values=y).reset_index()
    tooltip = [alt.Tooltip(f"{x}:T", title="Date")] + [alt.Tooltip(f"{name}:Q", title=name, format=y_format) for name in domain if name in wide.columns]
    rule = (
        alt.Chart(wide)
        .mark_rule(color=_colors(mode)["muted"], strokeWidth=1)
        .encode(x=f"{x}:T", opacity=alt.condition(hover, alt.value(1), alt.value(0)), tooltip=tooltip)
        .add_params(hover)
    )
    points = lines.mark_point(filled=True, size=64).encode(opacity=alt.condition(hover, alt.value(1), alt.value(0))).transform_filter(hover)
    return alt.layer(lines, rule, points).properties(height=HEIGHT)


def _strategy_frame(returns: pd.DataFrame) -> pd.DataFrame:
    long = returns.melt(id_vars="date", value_vars=list(STRATEGY_LABELS), var_name="strategy", value_name="ret")
    long["strategy"] = long["strategy"].map(STRATEGY_LABELS)
    long = long.sort_values(["strategy", "date"])
    long["growth"] = long.groupby("strategy")["ret"].transform(lambda r: (1 + r).cumprod())
    long["drawdown"] = long["growth"] / long.groupby("strategy")["growth"].cummax() - 1
    return long


def equity_curve(returns: pd.DataFrame, mode: str = "light") -> alt.Chart:
    """Growth of $1 for each strategy, after costs."""
    return _multi_line(_strategy_frame(returns), "date", "growth", "strategy", list(STRATEGY_LABELS.values()), "Growth of $1", "$.2f", mode)


def drawdown_chart(returns: pd.DataFrame, mode: str = "light") -> alt.Chart:
    return _multi_line(_strategy_frame(returns), "date", "drawdown", "strategy", list(STRATEGY_LABELS.values()), "Drawdown from peak", ".0%", mode)


def quintile_chart(quantiles: pd.DataFrame, mode: str = "light") -> alt.Chart:
    """Mean forward return by prediction quintile; a useful signal rises from Q1 to Q5."""
    df = quantiles.assign(label=lambda q: "Q" + q["quantile"].astype(str))
    bars = alt.Chart(df).mark_bar(color=_colors(mode)["series"][0], cornerRadiusEnd=4).encode(
        x=alt.X("label:N", title="Prediction quintile (Q5 = most likely to rise)", sort=None, scale=alt.Scale(paddingInner=0.4), axis=alt.Axis(labelAngle=0)),
        y=alt.Y("mean_return:Q", title="Mean 5-day return", axis=alt.Axis(format=".1%")),
        tooltip=[alt.Tooltip("label:N", title="Quintile"), alt.Tooltip("mean_return:Q", title="Mean return", format=".2%"), alt.Tooltip("observations:Q", title="Rebalances")],
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=_colors(mode)["muted"]).encode(y="y:Q")
    return alt.layer(bars, zero).properties(height=HEIGHT)


def fold_auc_chart(fold_metrics: pd.DataFrame, mode: str = "light") -> alt.Chart:
    """Out-of-sample AUC per walk-forward fold for each model, against the 0.5 no-skill line."""
    df = fold_metrics.assign(model=fold_metrics["model"].map(MODEL_LABELS).fillna(fold_metrics["model"]))
    domain = [MODEL_LABELS.get(m, m) for m in MODEL_LABELS if MODEL_LABELS[m] in set(df["model"])]
    lines = alt.Chart(df).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=64, filled=True)).encode(
        x=alt.X("fold:O", title="Fold (later = more recent)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("auc:Q", title="Out-of-sample AUC", scale=alt.Scale(zero=False)),
        color=alt.Color("model:N", scale=_series_scale(domain, mode), legend=_legend()),
        tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("fold:O", title="Fold"), alt.Tooltip("auc:Q", title="AUC", format=".3f"),
                 alt.Tooltip("test_start:T", title="Test from"), alt.Tooltip("test_end:T", title="Test to")],
    )
    no_skill = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(color=_colors(mode)["muted"], strokeDash=[4, 4]).encode(y="y:Q")
    return alt.layer(no_skill, lines).properties(height=HEIGHT)


def importance_chart(importance: pd.DataFrame, top: int = 12, mode: str = "light") -> alt.Chart:
    df = importance.head(top)
    return alt.Chart(df).mark_bar(color=_colors(mode)["series"][0], cornerRadiusEnd=4).encode(
        x=alt.X("importance_mean:Q", title="AUC drop when shuffled"),
        y=alt.Y("feature:N", sort="-x", title=None, scale=alt.Scale(paddingInner=0.35), axis=alt.Axis(labelLimit=220)),
        tooltip=[alt.Tooltip("feature:N", title="Feature"), alt.Tooltip("importance_mean:Q", title="Mean drop", format=".4f"), alt.Tooltip("importance_std:Q", title="Std", format=".4f")],
    ).properties(height=max(160, 24 * len(df)))


def calibration_chart(calibration: pd.DataFrame, mode: str = "light") -> alt.Chart:
    """Predicted probability vs. how often the event happened; perfect calibration sits on the diagonal."""
    low = float(min(calibration["mean_predicted"].min(), calibration["actual_rate"].min()))
    high = float(max(calibration["mean_predicted"].max(), calibration["actual_rate"].max()))
    pad = (high - low) * 0.1 or 0.05
    domain = [max(0.0, low - pad), min(1.0, high + pad)]
    diagonal = alt.Chart(pd.DataFrame({"x": domain, "y": domain})).mark_line(color=_colors(mode)["muted"], strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
    points = alt.Chart(calibration).mark_line(color=_colors(mode)["series"][0], strokeWidth=2, point=alt.OverlayMarkDef(size=64, filled=True)).encode(
        x=alt.X("mean_predicted:Q", title="Mean predicted probability", scale=alt.Scale(domain=domain), axis=alt.Axis(format=".0%")),
        y=alt.Y("actual_rate:Q", title="Share where it happened", scale=alt.Scale(domain=domain), axis=alt.Axis(format=".0%")),
        tooltip=[alt.Tooltip("bucket:O", title="Decile"), alt.Tooltip("mean_predicted:Q", title="Predicted", format=".1%"),
                 alt.Tooltip("actual_rate:Q", title="Actual", format=".1%"), alt.Tooltip("rows:Q", title="Rows")],
    )
    return alt.layer(diagonal, points).properties(height=HEIGHT)


def series_chart(frame: pd.DataFrame, column: str, title: str, value_format: str, mode: str = "light", reference: float | None = None, height: int = 220) -> alt.Chart:
    """One time series with a hover tooltip and an optional dashed reference line (e.g. zero)."""
    df = frame[["date", column]].dropna()
    hover = alt.selection_point(fields=["date"], nearest=True, on="pointerover", empty=False, clear="pointerout")
    base = alt.Chart(df).encode(x=alt.X("date:T", title=None))
    line = base.mark_line(color=_colors(mode)["series"][0], strokeWidth=2).encode(y=alt.Y(f"{column}:Q", title=title, scale=alt.Scale(zero=False)))
    rule = base.mark_rule(color=_colors(mode)["muted"]).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip(f"{column}:Q", title=title, format=value_format)],
    ).add_params(hover)
    layers = [line, rule]
    if reference is not None:
        layers.insert(0, alt.Chart(pd.DataFrame({"y": [reference]})).mark_rule(color=_colors(mode)["muted"], strokeDash=[4, 4]).encode(y="y:Q"))
    return alt.layer(*layers).properties(height=height)


def price_chart(history: pd.DataFrame, ticker: str, mode: str = "light") -> alt.Chart:
    hover = alt.selection_point(fields=["date"], nearest=True, on="pointerover", empty=False, clear="pointerout")
    base = alt.Chart(history).encode(x=alt.X("date:T", title=None))
    line = base.mark_line(color=_colors(mode)["series"][0], strokeWidth=2).encode(
        y=alt.Y("close:Q", title=f"{ticker} close", scale=alt.Scale(zero=False), axis=alt.Axis(format="$,.0f"))
    )
    rule = base.mark_rule(color=_colors(mode)["muted"]).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("close:Q", title="Close", format="$,.2f")],
    ).add_params(hover)
    point = line.mark_point(filled=True, size=64, color=_colors(mode)["series"][0]).encode(opacity=alt.condition(hover, alt.value(1), alt.value(0)))
    return alt.layer(line, rule, point).properties(height=260)
