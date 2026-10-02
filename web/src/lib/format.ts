import type { Num } from "../api";

const missing = (v: Num | undefined): v is null | undefined => v === null || v === undefined || Number.isNaN(v);

export const pct = (v: Num | undefined, digits = 1) => (missing(v) ? "—" : `${(v * 100).toFixed(digits)}%`);
export const signedPct = (v: Num | undefined, digits = 1) => (missing(v) ? "—" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v * 100).toFixed(digits)}%`);
export const num = (v: Num | undefined, digits = 2) => (missing(v) ? "—" : v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits }));
export const price = (v: Num | undefined) => (missing(v) ? "—" : `$${v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`);

export function money(v: Num | undefined): string {
  if (missing(v)) return "—";
  const abs = Math.abs(v);
  const sign = v < 0 ? "−" : "";
  if (abs >= 1e12) return `${sign}$${(abs / 1e12).toFixed(2)}T`;
  if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(1)}M`;
  return `${sign}$${abs.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
}

export function date(value: string | null | undefined, style: "short" | "long" = "short"): string {
  if (!value) return "—";
  const d = new Date(value.length === 10 ? `${value}T00:00:00` : value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleDateString("en-US", style === "long" ? { year: "numeric", month: "long", day: "numeric" } : { year: "numeric", month: "short", day: "numeric" });
}

export const ordinal = (n: number) => {
  const s = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return `${n}${s[(v - 20) % 10] || s[v] || s[0]}`;
};

export const tone = (v: Num | undefined) => (missing(v) || v === 0 ? "text-text" : v > 0 ? "text-pos" : "text-neg");

export const MODEL_LABELS: Record<string, string> = {
  logistic_regression: "Logistic regression",
  random_forest: "Random forest",
  gradient_boosting: "Gradient boosting",
};

export const FEATURE_LABELS: Record<string, string> = {
  return_5d: "5-day return",
  return_20d: "20-day return",
  momentum_60d: "60-day momentum",
  volatility_20d: "Volatility",
  ma_gap_50d: "Gap to 50-day average",
  volume_ratio_20d: "Volume vs. 20-day average",
  sentiment_20d: "News sentiment",
  news_count_20d: "News count",
  return_20d_xs_rank: "20-day return (rank)",
  momentum_60d_xs_rank: "60-day momentum (rank)",
  volatility_20d_xs_rank: "Volatility (rank)",
  return_20d_vs_sector: "Return vs. sector",
  earnings_yield_xs_rank: "Earnings yield (rank)",
  sales_yield_xs_rank: "Sales yield (rank)",
  book_to_market_xs_rank: "Book-to-market (rank)",
  profit_margin_xs_rank: "Profit margin (rank)",
  roe_xs_rank: "ROE (rank)",
  revenue_growth_yoy_xs_rank: "Revenue growth (rank)",
  market_cap_xs_rank: "Size (rank)",
  vix: "VIX",
  vix_change_20d: "VIX 20-day change",
  term_spread: "Yield-curve spread",
  sector: "Sector",
  model_probability: "Model",
};
export const featureLabel = (name: string) => FEATURE_LABELS[name] ?? name.replace(/^(cat|num)__/, "").replace(/_/g, " ");
