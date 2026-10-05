import { useQuery } from "@tanstack/react-query";
import { api, type Attribution, type BacktestData, type Construction, type StrategyStats } from "../api";
import { ColumnChart, MultiLineChart } from "../components/charts";
import { Callout, Card, DataTable, ErrorState, LoadingGrid, PageHeader, Pill, Stat, type Column } from "../components/ui";
import { date, num, pct, signedPct, tone } from "../lib/format";

type Row = StrategyStats & { strategy: string };

const FACTORS = [
  { key: "MKT", label: "Market" },
  { key: "SMB", label: "Size" },
  { key: "HML", label: "Value" },
  { key: "RMW", label: "Profitability" },
  { key: "CMA", label: "Investment" },
  { key: "MOM", label: "Momentum" },
  { key: "STREV", label: "Reversal" },
];

type Staggered = NonNullable<BacktestData["staggered"]>[number];
type Interval = NonNullable<BacktestData["bootstrap"]>[number];
type CostRow = { strategy: string } & Record<string, number | string | null>;

function costTable(rows: NonNullable<BacktestData["cost_sensitivity"]>): { rows: CostRow[]; levels: number[] } {
  const levels = [...new Set(rows.map((r) => r.cost_bps))].sort((a, b) => a - b);
  const byStrategy = new Map<string, CostRow>();
  for (const r of rows) {
    const row: CostRow = byStrategy.get(r.strategy) ?? { strategy: r.strategy };
    row[`c${r.cost_bps}`] = r.cagr;
    byStrategy.set(r.strategy, row);
  }
  return { rows: [...byStrategy.values()], levels };
}

function significance(t: number | null | undefined) {
  if (t === null || t === undefined || Number.isNaN(t)) return "";
  return Math.abs(t) >= 2 ? "font-semibold" : "text-text-2";
}

export default function Backtest() {
  const { data, isPending, isError, error } = useQuery({ queryKey: ["backtest"], queryFn: api.backtest });
  if (isPending) return <LoadingGrid />;
  if (isError) return <ErrorState error={error} />;
  const stats = Object.fromEntries(data.summary.map((r) => [r.strategy, r])) as Record<string, Row>;
  const cfg = data.config;
  const series = Object.entries(data.labels).map(([key, label]) => ({ key, label }));
  const hasIndex = "sp500" in stats;
  const reference = hasIndex ? stats.sp500 : stats.benchmark;
  const referenceLabel = hasIndex ? "S&P 500" : "Benchmark";
  const horizon = cfg?.holding_days ?? 5;
  const attributionColumns = (rows: Attribution[]): Column<Attribution>[] => [
    { key: "strategy", label: "Strategy", render: (r) => <span className="font-medium">{data.labels[r.strategy] ?? r.strategy}</span> },
    { key: "alpha", label: "Alpha / year", align: "right", render: (r) => <span className={significance(r.alpha_tstat)}>{signedPct(r.alpha_annual)}</span> },
    { key: "alpha_t", label: "t-stat", align: "right", render: (r) => <span className={significance(r.alpha_tstat)}>{num(r.alpha_tstat, 2)}</span> },
    ...FACTORS.filter((f) => rows.some((r) => r[`beta_${f.key}`] !== undefined && r[`beta_${f.key}`] !== null)).map((f) => ({
      key: f.key,
      label: `${f.label} β`,
      align: "right" as const,
      render: (r: Attribution) => <span className={significance(r[`tstat_${f.key}`] as number)}>{num(r[`beta_${f.key}`] as number, 2)}</span>,
    })),
    { key: "r2", label: "R²", align: "right", render: (r) => num(r.r_squared, 2) },
  ];
  const constructionColumns: Column<Construction>[] = [
    { key: "name", label: "Construction", render: (r) => <span className="font-medium">{r.construction} {r.production && <Pill className="ml-1">production</Pill>}</span> },
    { key: "lo", label: "Long-only CAGR", align: "right", render: (r) => signedPct(r.long_only_cagr) },
    { key: "ex", label: "vs. equal-weight", align: "right", render: (r) => <span className={tone(r.excess_vs_equal_weight)}>{signedPct(r.excess_vs_equal_weight)}</span>, sortValue: (r) => r.excess_vs_equal_weight },
    { key: "lod", label: "Cost drag", align: "right", render: (r) => <span className="text-neg">{pct(r.long_only_cost_drag)}</span> },
    { key: "lot", label: "Turnover", align: "right", render: (r) => num(r.long_only_turnover, 2) },
    { key: "ls", label: "Long-short CAGR", align: "right", render: (r) => <span className={tone(r.long_short_cagr)}>{signedPct(r.long_short_cagr)}</span>, sortValue: (r) => r.long_short_cagr },
    { key: "lss", label: "L/S Sharpe", align: "right", render: (r) => num(r.long_short_sharpe, 2) },
    { key: "lsd", label: "L/S cost drag", align: "right", render: (r) => <span className="text-neg">{pct(r.long_short_cost_drag)}</span> },
  ];
  const staggeredColumns: Column<Staggered>[] = [
    { key: "strategy", label: "Strategy", render: (r) => <span className="font-medium">{data.labels[r.strategy] ?? r.strategy}</span> },
    { key: "cagr", label: "CAGR: mean (range)", align: "right", render: (r) => `${signedPct(r.cagr_mean)} (${signedPct(r.cagr_min)} to ${signedPct(r.cagr_max)})` },
    { key: "sharpe", label: "Sharpe: mean (range)", align: "right", render: (r) => `${num(r.sharpe_mean, 2)} (${num(r.sharpe_min, 2)} to ${num(r.sharpe_max, 2)})` },
  ];
  const intervalColumns: Column<Interval>[] = [
    { key: "strategy", label: "Strategy", render: (r) => <span className="font-medium">{data.labels[r.strategy] ?? r.strategy}</span> },
    { key: "cagr", label: "CAGR, 95% interval", align: "right", render: (r) => `${signedPct(r.cagr_low)} to ${signedPct(r.cagr_high)}` },
    { key: "sharpe", label: "Sharpe, 95% interval", align: "right", render: (r) => `${num(r.sharpe_low, 2)} to ${num(r.sharpe_high, 2)}` },
    { key: "p", label: "P(Sharpe ≤ 0)", align: "right", render: (r) => pct(r.prob_sharpe_not_positive, 0) },
  ];
  const costs = costTable(data.cost_sensitivity ?? []);
  const costColumns: Column<CostRow>[] = [
    { key: "strategy", label: "Strategy", render: (r) => <span className="font-medium">{data.labels[r.strategy] ?? r.strategy}</span> },
    ...costs.levels.map((level) => ({
      key: `c${level}`,
      label: `${level} bps`,
      align: "right" as const,
      render: (r: CostRow) => <span className={tone(r[`c${level}`] as number | null)}>{signedPct(r[`c${level}`] as number | null)}</span>,
    })),
  ];
  const robust = data.robustness ?? {};
  const deflated = robust.long_short?.deflated_sharpe;
  const pbo = robust.pbo?.pbo;
  const breakeven = robust.breakeven_cost ?? {};
  const costText =
    cfg?.cost_model === "spread"
      ? `Costs: ${cfg.cost_bps} bps per unit traded for the median stock, scaled per stock by its estimated bid-ask spread (Corwin-Schultz, from daily highs and lows)`
      : `Costs: ${cfg?.cost_bps} bps per unit of weight traded`;
  const buffer = cfg?.exit_quantile ? ` Holdings are kept until they leave the top ${Math.round((cfg.exit_quantile ?? 0) * 100)}%.` : "";
  const smoothing = cfg?.smooth_days && cfg.smooth_days > 1 ? ` Scores are averaged over ${cfg.smooth_days} sessions.` : "";
  const columns: Column<Row>[] = [
    { key: "strategy", label: "Strategy", render: (r) => <span className="font-medium">{data.labels[r.strategy] ?? r.strategy}</span> },
    { key: "total", label: "Total return", align: "right", render: (r) => <span className={tone(r.total_return)}>{signedPct(r.total_return)}</span> },
    { key: "cagr", label: "CAGR", align: "right", render: (r) => signedPct(r.cagr) },
    { key: "vol", label: "Volatility", align: "right", render: (r) => pct(r.ann_volatility) },
    { key: "sharpe", label: "Sharpe", align: "right", render: (r) => num(r.sharpe, 2) },
    { key: "dd", label: "Max drawdown", align: "right", render: (r) => <span className="text-neg">{pct(r.max_drawdown)}</span> },
    { key: "hit", label: "Hit rate", align: "right", render: (r) => pct(r.hit_rate, 0) },
    { key: "turn", label: "Turnover", align: "right", render: (r) => num(r.avg_turnover, 2) },
  ];
  const first = data.growth[0]?.date;
  const last = data.growth[data.growth.length - 1]?.date;
  const beat = (stats.long_only?.total_return ?? 0) > (reference?.total_return ?? 0);

  return (
    <div className="space-y-5">
      <PageHeader
        title="Backtest"
        description={
          cfg
            ? `Out-of-sample predictions from walk-forward validation, ${date(first)} to ${date(last)}. Every ${cfg.holding_days} sessions the index members are ranked; positions are entered one session later and held ${cfg.holding_days} sessions.${buffer}${smoothing} ${costText}. Stocks that stop trading stay in until their last price.`
            : undefined
        }
      />
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Long top 20%" value={signedPct(stats.long_only?.total_return, 0)} tone={tone(stats.long_only?.total_return)} hint={`CAGR ${signedPct(stats.long_only?.cagr)} · Sharpe ${num(stats.long_only?.sharpe, 2)}`} />
        <Stat label={hasIndex ? "S&P 500 (SPY)" : "Benchmark"} value={signedPct(reference?.total_return, 0)} tone={tone(reference?.total_return)} hint={`CAGR ${signedPct(reference?.cagr)} · Sharpe ${num(reference?.sharpe, 2)}`} />
        <Stat label="Long-short" value={signedPct(stats.long_short?.total_return, 0)} tone={tone(stats.long_short?.total_return)} hint={`Sharpe ${num(stats.long_short?.sharpe, 2)}`} />
        <Stat label="Rank IC at rebalances" value={num(stats.long_short?.ic_mean, 3)} hint={`t-stat ${num(stats.long_short?.ic_tstat, 2)}`} help="Rank correlation between the prediction and the realised return at each rebalance." />
      </div>
      {(deflated !== undefined || pbo !== undefined) && (
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <Stat label="vs. equal-weight members" value={signedPct(stats.long_only?.excess_cagr_vs_equal_weight)} tone={tone(stats.long_only?.excess_cagr_vs_equal_weight)} hint={`Information ratio ${num(stats.long_only?.information_ratio, 2)}`} help="The like-for-like benchmark for an equal-weighted portfolio of index members." />
          <Stat label="Deflated Sharpe (long-short)" value={pct(deflated, 0)} hint={`After ${robust.trials_counted ?? "?"} configurations; 95%+ is convincing`} help="The chance the true Sharpe is above what the best of this many unskilled backtests would reach by luck (Bailey and López de Prado)." />
          <Stat label="Backtest overfitting" value={pct(pbo, 0)} hint="Near 50% = picking the best backtest is a coin flip" help="Probability of backtest overfitting from combinatorially symmetric cross-validation across every backtest at this horizon." />
          <Stat label="Break-even cost" value={breakeven.long_short_bps === undefined || breakeven.long_short_bps === null ? "—" : `${num(breakeven.long_short_bps, 1)} bps`} hint={breakeven.long_only_vs_equal_weight_bps === undefined || breakeven.long_only_vs_equal_weight_bps === null ? "Long-short" : `Long-short · long-only vs. EW ${num(breakeven.long_only_vs_equal_weight_bps, 1)} bps`} help="Cost per unit of weight traded at which the gross edge is used up." />
        </div>
      )}
      <Callout tone={beat ? "info" : "warn"}>
        {beat ? `The model's top fifth beat the ${referenceLabel} over this period.` : `The model's top fifth did not beat the ${hasIndex ? "S&P 500" : "equal-weight benchmark"}.`} Sharpe ratios for long-only portfolios and benchmarks are in excess of 3-month T-bills; long-short is self-financing.
      </Callout>
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Growth of $1" subtitle="After trading costs">
          <MultiLineChart data={data.growth} series={series} yFormat={(v) => `$${v.toFixed(2)}`} reference={1} />
        </Card>
        <Card title="Drawdown from peak">
          <MultiLineChart data={data.drawdown} series={series} yFormat={(v) => `${(v * 100).toFixed(0)}%`} reference={0} />
        </Card>
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Return by prediction quintile" subtitle={`Mean ${horizon}-day return; a working signal rises from Q1 to Q5`}>
          <ColumnChart data={data.quintiles} x="quantile" y="mean_return" xFormat={(v) => `Q${v}`} yFormat={(v) => `${(v * 100).toFixed(2)}%`} signed tooltipRows={(r) => [{ label: "Mean return", value: `${(r.mean_return * 100).toFixed(2)}%` }, { label: "Rebalances", value: String(r.observations) }]} />
        </Card>
        <Card title="Rank IC over time" subtitle="12-rebalance rolling mean; above zero means the ranking worked">
          <MultiLineChart data={data.ic.filter((r) => r.ic_rolling !== null) as { date: string; ic_rolling: number }[]} series={[{ key: "ic_rolling", label: "Rolling IC" }]} yFormat={(v) => v.toFixed(2)} reference={0} />
        </Card>
      </div>
      <Card title="Statistics" bodyClassName="p-0">
        <DataTable rows={data.summary} columns={columns} rowKey={(r) => r.strategy} />
      </Card>
      {data.attribution.length > 0 && (
        <Card
          title="Where the returns come from"
          subtitle="Each strategy's returns regressed on market, size (small minus big), value (cheap minus expensive) and momentum factors built from the same members and dates. Alpha is what the factors don't explain; bold = |t| ≥ 2."
          bodyClassName="p-0"
        >
          <DataTable rows={data.attribution} columns={attributionColumns(data.attribution)} rowKey={(r) => r.strategy} />
        </Card>
      )}
      {data.french_attribution && data.french_attribution.length > 0 && (
        <Card title="Cross-check: Fama-French factors" subtitle="The same regression on Kenneth French's published factors for the whole US market, adding profitability, investment and short-term reversal. Newey-West t-stats; bold = |t| ≥ 2." bodyClassName="p-0">
          <DataTable rows={data.french_attribution} columns={attributionColumns(data.french_attribution)} rowKey={(r) => r.strategy} />
        </Card>
      )}
      {data.constructions && data.constructions.length > 0 && (
        <Card title="Portfolio construction: where the costs go" subtitle="The same predictions traded four ways. Buffers and smoothing cut turnover; the optimiser trades only when the expected gain beats the cost, and its long-short is neutral to beta, size and value" bodyClassName="p-0">
          <DataTable rows={data.constructions} columns={constructionColumns} rowKey={(r) => r.construction} />
        </Card>
      )}
      <div className="grid gap-4 xl:grid-cols-2">
        {costs.rows.length > 0 && (
          <Card title="Net CAGR at different costs" subtitle="Flat cost per unit of weight traded, same trades" bodyClassName="p-0">
            <DataTable rows={costs.rows} columns={costColumns} rowKey={(r) => r.strategy} />
          </Card>
        )}
        {data.staggered && data.staggered.length > 0 && (
          <Card title="Does the rebalance day matter?" subtitle={`The backtest started on each of the ${data.staggered[0]?.offsets ?? horizon} sessions of the holding period; the range is calendar luck`} bodyClassName="p-0">
            <DataTable rows={data.staggered} columns={staggeredColumns} rowKey={(r) => r.strategy} />
          </Card>
        )}
      </div>
      {data.bootstrap && data.bootstrap.length > 0 && (
        <Card title="How sure can we be?" subtitle="Stationary-bootstrap 95% intervals: blocks of consecutive periods resampled, so volatility clusters are kept" bodyClassName="p-0">
          <DataTable rows={data.bootstrap} columns={intervalColumns} rowKey={(r) => r.strategy} />
        </Card>
      )}
      <Card title="Caveats">
        <ul className="list-disc space-y-1 pl-5 text-[13px] text-text-2">
          <li>Survivorship bias is reduced, not removed: former members delisted after mergers or failures often have no price data. Stocks that stop trading are kept to their last price, with a −30% delisting return for apparent failures.</li>
          <li>Cost levels are assumed (the median stock pays the configured bps); only each stock's relative cost comes from spread estimates. There is no market-impact model.</li>
          <li>Several models, portfolio constructions and research setups were compared; the deflated Sharpe and overfitting probability above account for that, but only the untouched holdout gives a clean read.</li>
        </ul>
      </Card>
    </div>
  );
}
