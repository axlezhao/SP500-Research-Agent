// Typed client for the FastAPI backend (src/sp500_agent/api/server.py).

export type Num = number | null;

export interface RankedStock {
  ticker: string;
  company: string;
  sector: string | null;
  rank: number;
  model_probability: number;
  return_5d: Num;
  return_20d: Num;
  momentum_60d: Num;
  volatility_20d: Num;
}

export interface ScreenerRow extends RankedStock {
  stance: Stance;
  close: Num;
  market_cap: Num;
  pe_ratio: Num;
  profit_margin: Num;
  revenue_growth_yoy: Num;
  sentiment_20d: Num;
}

export type Stance = "constructive" | "neutral" | "cautious";

export interface StrategyStats {
  periods: number;
  total_return: Num;
  cagr: Num;
  ann_volatility: Num;
  sharpe: Num;
  max_drawdown: Num;
  hit_rate: Num;
  avg_turnover: Num;
  information_ratio?: Num;
  ic_mean?: Num;
  ic_tstat?: Num;
}

export interface MacroSeries {
  label: string;
  latest: Num;
  one_year_ago: Num;
  percentile_10y: Num;
}

export interface Overview {
  as_of_date: string;
  stocks_ranked: number;
  history_start: string;
  history_end: string;
  sectors: unknown;
  model: string;
  target: string;
  horizon_days: number;
  spec: Spec;
  data_source: string;
  data_quality: Record<string, unknown>;
  left_out_of_ranking: Record<string, string>;
  live_lookups: { news: string; sec_filings: string };
  model_metrics: { auc_mean?: Num; auc_std?: Num; ic_mean?: Num; ic_tstat?: Num; accuracy?: Num; baseline_accuracy?: Num; oos_start?: string; oos_end?: string };
  backtest: Record<"long_only" | "long_short" | "benchmark", StrategyStats> & { sp500?: StrategyStats };
  top: RankedStock[];
  bottom: RankedStock[];
  sectors_summary?: never;
  macro: { as_of: string; series: Record<string, MacroSeries>; notes: string } | null;
}

export interface SectorRow {
  sector: string;
  stocks: number;
  mean_up_probability: number;
  mean_return_20d: Num;
  mean_volatility: Num;
  top_ranked: string;
}

export interface StockDetail {
  ticker: string;
  company: string;
  sector: string | null;
  sub_industry: string | null;
  excluded_reason: string | null;
  snapshot: {
    rank: number;
    out_of: number;
    model_probability: number;
    model_view: Stance;
    data_warning: string | null;
    signals: Record<string, Num>;
    fundamentals: Record<string, Num>;
    fundamentals_as_of: string | null;
    notes: string;
  } | null;
  latest: { date: string; close: Num; return_1d: Num; return_5d: Num; return_20d: Num; momentum_60d: Num; volatility_20d: Num };
  percentiles: { signal: string; label: string; percentile: number }[];
  membership: { currently_in_index: boolean; membership_periods: { from: string | null; to: string | null }[] } | null;
  live: { news: boolean; filings: boolean };
}

export interface PricePoint {
  date: string;
  close: number;
  volume?: Num;
}

export interface FundamentalRow {
  available_date: string;
  period_end: string | null;
  revenue_ttm: Num;
  net_income_ttm: Num;
  profit_margin: Num;
  revenue_growth_yoy: Num;
  equity: Num;
  liabilities: Num;
  shares_outstanding: Num;
}

export interface Headline {
  date: string | null;
  title: string;
  sentiment_score: Num;
  source: string | null;
  url?: string | null;
}

export interface Filing {
  form: string;
  filed: string;
  report_date: string | null;
  items: string | null;
  url: string;
}

export interface Attribution {
  strategy: string;
  alpha_annual: number;
  alpha_tstat: number;
  r_squared: number;
  periods: number;
  [beta: string]: number | string;
}

export interface Experiment {
  setup: string;
  horizon: number;
  relative_to: string;
  portfolio: string | null;
  production: boolean;
  auc: Num;
  ic_mean: Num;
  ic_tstat: Num;
  long_only_cagr: Num;
  long_only_sharpe: Num;
  reference: string | null;
  reference_cagr: Num;
  long_short_cagr: Num;
  long_short_sharpe: Num;
  long_short_alpha: Num;
  long_short_alpha_t: Num;
  long_only_alpha: Num;
  long_only_alpha_t: Num;
  turnover: Num;
  note?: string | null;
}

export interface Spec {
  horizon: number;
  relative_to: string;
  description: string;
  label: string;
}

type StrategySeries = { date: string; long_only: number; long_short: number; benchmark: number; sp500?: number };

export interface BacktestData {
  config: { holding_days: number; quantile: number; cost_bps: number; neutralize?: string } | null;
  labels: Record<string, string>;
  summary: (StrategyStats & { strategy: string; information_ratio_vs_sp500?: Num })[];
  growth: StrategySeries[];
  drawdown: StrategySeries[];
  attribution: Attribution[];
  quintiles: { quantile: number; mean_return: number; observations: number }[];
  ic: { date: string; ic: Num; ic_rolling: Num }[];
}

export interface ModelData {
  selected: string;
  target: string;
  spec: Spec;
  experiments: Experiment[];
  features: string[];
  comparison: { model: string; auc_mean: number; auc_std: number; accuracy: number; baseline_accuracy: number; brier: number; ic_mean: Num; ic_tstat: Num }[];
  folds: { model: string; fold: number; auc: number; test_start: string; test_end: string }[];
  calibration: { bucket: number; mean_predicted: number; actual_rate: number; rows: number }[];
  importance: { feature: string; importance_mean: number; importance_std: number }[];
  signals: { signal: string; ic_mean: Num; ic_tstat: Num; ic_positive_share: Num }[];
  report_available: boolean;
}

export interface DataPageData {
  source: string;
  quality: Record<string, any>;
  eda: Record<string, any> | null;
  macro: { date: string; tbill_3m: Num; treasury_10y: Num; vix: Num; term_spread: Num }[];
  index_changes: { date: string; added: string | null; added_name: string | null; removed: string | null; removed_name: string | null; reason: string | null }[];
  excluded: Record<string, string>;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function get<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new ApiError(response.status, detail);
  }
  return response.json() as Promise<T>;
}

const enc = encodeURIComponent;

export const api = {
  overview: () => get<Overview & { sectors: SectorRow[] }>("/api/overview"),
  stocks: () => get<{ as_of: string; target: string; excluded: Record<string, string>; stocks: ScreenerRow[] }>("/api/stocks"),
  search: (q: string) => get<{ matches: { ticker: string; company: string; sector: string | null }[] }>(`/api/search?q=${enc(q)}`),
  stock: (t: string) => get<StockDetail>(`/api/stocks/${enc(t)}`),
  prices: (t: string, days = 5000) => get<{ prices: PricePoint[] }>(`/api/stocks/${enc(t)}/prices?days=${days}`),
  fundamentals: (t: string) => get<{ history: FundamentalRow[]; note?: string }>(`/api/stocks/${enc(t)}/fundamentals`),
  news: (t: string) => get<{ headlines: Headline[]; fetched_from?: string; note?: string }>(`/api/stocks/${enc(t)}/news`),
  filings: (t: string) => get<{ filings: Filing[]; note?: string; notes?: string }>(`/api/stocks/${enc(t)}/filings`),
  backtest: () => get<BacktestData>("/api/backtest"),
  model: () => get<ModelData>("/api/model"),
  data: () => get<DataPageData>("/api/data"),
  agentStatus: () => get<{ providers: string[]; mode: "llm" | "rule-based"; demo: boolean; limits: { per_hour: number | null; per_day: number | null } }>("/api/agent/status"),
};

export type AgentEvent =
  | { type: "tool_start"; name: string; arguments: Record<string, unknown> }
  | { type: "tool_end"; name: string; arguments: Record<string, unknown>; error: string | null; result: unknown }
  | { type: "answer"; text: string; provider: string; model: string | null; usage: Record<string, number> }
  | { type: "error"; message: string };

/** POST a chat message and call onEvent for each server-sent event until the stream ends. */
export async function streamChat(sessionId: string, message: string, onEvent: (event: AgentEvent) => void, signal?: AbortSignal): Promise<void> {
  const response = await fetch("/api/agent/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
    signal,
  });
  if (!response.ok || !response.body) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new ApiError(response.status, detail);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const chunk = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      for (const line of chunk.split("\n")) {
        if (line.startsWith("data: ")) onEvent(JSON.parse(line.slice(6)) as AgentEvent);
      }
    }
  }
}

export async function resetChat(sessionId: string): Promise<void> {
  await fetch("/api/agent/reset", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: sessionId }) });
}
