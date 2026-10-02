import { useQuery } from "@tanstack/react-query";
import { api, type StrategyStats } from "../api";
import { ColumnChart, MultiLineChart } from "../components/charts";
import { Callout, Card, DataTable, ErrorState, LoadingGrid, PageHeader, Stat, type Column } from "../components/ui";
import { date, num, pct, signedPct, tone } from "../lib/format";

const SERIES = [
  { key: "long_only", label: "Long top 20%" },
  { key: "long_short", label: "Long-short" },
  { key: "benchmark", label: "Equal-weight benchmark" },
];

type Row = StrategyStats & { strategy: string };

export default function Backtest() {
  const { data, isPending, isError, error } = useQuery({ queryKey: ["backtest"], queryFn: api.backtest });
  if (isPending) return <LoadingGrid />;
  if (isError) return <ErrorState error={error} />;
  const stats = Object.fromEntries(data.summary.map((r) => [r.strategy, r])) as Record<string, Row>;
  const cfg = data.config;
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
  const beat = (stats.long_only?.total_return ?? 0) > (stats.benchmark?.total_return ?? 0);

  return (
    <div className="space-y-5">
      <PageHeader
        title="Backtest"
        description={
          cfg
            ? `Out-of-sample predictions from walk-forward validation, ${date(first)} to ${date(last)}. Every ${cfg.holding_days} sessions the index members are ranked; positions are entered one session later and held ${cfg.holding_days} sessions. Costs: ${cfg.cost_bps} bps per unit of weight traded.`
            : undefined
        }
      />
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Long top 20%" value={signedPct(stats.long_only?.total_return, 0)} tone={tone(stats.long_only?.total_return)} hint={`CAGR ${signedPct(stats.long_only?.cagr)} · Sharpe ${num(stats.long_only?.sharpe, 2)}`} />
        <Stat label="Benchmark" value={signedPct(stats.benchmark?.total_return, 0)} tone={tone(stats.benchmark?.total_return)} hint={`CAGR ${signedPct(stats.benchmark?.cagr)} · Sharpe ${num(stats.benchmark?.sharpe, 2)}`} />
        <Stat label="Long-short" value={signedPct(stats.long_short?.total_return, 0)} tone={tone(stats.long_short?.total_return)} hint={`Sharpe ${num(stats.long_short?.sharpe, 2)}`} />
        <Stat label="Rank IC at rebalances" value={num(stats.long_short?.ic_mean, 3)} hint={`t-stat ${num(stats.long_short?.ic_tstat, 2)}`} help="Rank correlation between the prediction and the realised return at each rebalance." />
      </div>
      <Callout tone={beat ? "info" : "warn"}>
        {beat ? "The model's top fifth beat the equal-weight benchmark over this period." : "The model's top fifth did not beat simply holding every member equally."} Sharpe ratios for long-only and the benchmark are in excess of 3-month T-bills; long-short is self-financing.
      </Callout>
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Growth of $1" subtitle="After trading costs">
          <MultiLineChart data={data.growth} series={SERIES} yFormat={(v) => `$${v.toFixed(2)}`} reference={1} />
        </Card>
        <Card title="Drawdown from peak">
          <MultiLineChart data={data.drawdown} series={SERIES} yFormat={(v) => `${(v * 100).toFixed(0)}%`} reference={0} />
        </Card>
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Return by prediction quintile" subtitle="Mean 5-day return; a working signal rises from Q1 to Q5">
          <ColumnChart data={data.quintiles} x="quantile" y="mean_return" xFormat={(v) => `Q${v}`} yFormat={(v) => `${(v * 100).toFixed(2)}%`} signed tooltipRows={(r) => [{ label: "Mean return", value: `${(r.mean_return * 100).toFixed(2)}%` }, { label: "Rebalances", value: String(r.observations) }]} />
        </Card>
        <Card title="Rank IC over time" subtitle="12-rebalance rolling mean; above zero means the ranking worked">
          <MultiLineChart data={data.ic.filter((r) => r.ic_rolling !== null) as { date: string; ic_rolling: number }[]} series={[{ key: "ic_rolling", label: "Rolling IC" }]} yFormat={(v) => v.toFixed(2)} reference={0} />
        </Card>
      </div>
      <Card title="Statistics" bodyClassName="p-0">
        <DataTable rows={data.summary} columns={columns} rowKey={(r) => r.strategy} />
      </Card>
      <Card title="Caveats">
        <ul className="list-disc space-y-1 pl-5 text-[13px] text-text-2">
          <li>Survivorship bias is reduced, not removed: former members delisted after mergers or failures often have no price data.</li>
          <li>Costs are a flat charge per unit traded, with no market impact or short-borrow fees.</li>
          <li>Three models were compared; the selected one's results are slightly optimistic for that reason alone.</li>
        </ul>
      </Card>
    </div>
  );
}
