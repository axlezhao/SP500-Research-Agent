import { useQuery } from "@tanstack/react-query";
import { ArrowRight } from "lucide-react";
import { Link, useNavigate } from "react-router";
import { api, type RankedStock } from "../api";
import { HorizontalBars, MultiLineChart } from "../components/charts";
import { Callout, Card, DataTable, ErrorState, LoadingGrid, PageHeader, Stat, type Column } from "../components/ui";
import { date, num, pct, signedPct, tone } from "../lib/format";

const rankedColumns: Column<RankedStock>[] = [
  { key: "rank", label: "#", align: "right", render: (r) => <span className="text-muted">{r.rank}</span> },
  {
    key: "ticker",
    label: "Stock",
    render: (r) => (
      <div className="flex min-w-0 items-baseline gap-2">
        <span className="num font-semibold">{r.ticker}</span>
        <span className="max-w-44 truncate text-xs text-muted">{r.company}</span>
      </div>
    ),
  },
  { key: "p", label: "Probability", align: "right", render: (r) => pct(r.model_probability) },
  { key: "r20", label: "20D", align: "right", render: (r) => <span className={tone(r.return_20d)}>{signedPct(r.return_20d)}</span> },
  { key: "vol", label: "Vol.", align: "right", render: (r) => pct(r.volatility_20d, 0) },
];

export default function Dashboard() {
  const navigate = useNavigate();
  const overview = useQuery({ queryKey: ["overview"], queryFn: api.overview });
  const backtest = useQuery({ queryKey: ["backtest"], queryFn: api.backtest });
  if (overview.isPending) return <LoadingGrid />;
  if (overview.isError) return <ErrorState error={overview.error} />;
  const o = overview.data;
  const m = o.model_metrics;
  const bt = o.backtest;
  const vix = o.macro?.series.vix;
  const spread = o.macro?.series.term_spread;
  const edge = (m.ic_tstat ?? 0) > 2;
  const open = (r: RankedStock) => navigate(`/stock/${r.ticker}`);

  return (
    <div className="space-y-5">
      <PageHeader
        title="Market overview"
        description={
          <>
            {o.stocks_ranked} S&amp;P 500 members ranked by the model's probability to <strong>{o.target}</strong>, as of {date(o.as_of_date, "long")}.
          </>
        }
      />

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <Stat label="Stocks ranked" value={o.stocks_ranked} hint={Object.keys(o.left_out_of_ranking).length ? `${Object.keys(o.left_out_of_ranking).length} left out (data warnings)` : "Index members today"} />
        <Stat label="Walk-forward AUC" value={num(m.auc_mean, 3)} hint="0.500 = no skill" help="Area under the ROC curve on out-of-sample folds." />
        <Stat label="Rank IC" value={num(m.ic_mean, 3)} hint={`t-stat ${num(m.ic_tstat, 2)}`} help="Average daily rank correlation between the model's score and the realised return; Newey-West t-stat, corrected for overlapping return windows." />
        <Stat label="Long top 20%" value={signedPct(bt.long_only?.total_return, 0)} tone={tone(bt.long_only?.total_return)} hint={bt.sp500 ? `S&P 500 ${signedPct(bt.sp500.total_return, 0)}` : `Benchmark ${signedPct(bt.benchmark?.total_return, 0)}`} />
        <Stat label="Long-short Sharpe" value={num(bt.long_short?.sharpe, 2)} tone={tone(bt.long_short?.sharpe)} hint="After costs, out of sample" />
        <Stat label="VIX" value={num(vix?.latest, 1)} hint={vix?.percentile_10y != null ? `${Math.round(vix.percentile_10y * 100)}th pct. of 10 years` : undefined} />
      </div>

      <Callout tone={edge ? "info" : "warn"}>
        <span className="font-medium">{edge ? "The model shows a small out-of-sample edge." : "The out-of-sample evidence for an edge is weak."}</span>{" "}
        <span className="text-text-2">
          Rankings below are research output from a model with AUC {num(m.auc_mean, 3)} and rank IC t-stat {num(m.ic_tstat, 2)}. Treat them as one input, and see the{" "}
          <Link to="/backtest" className="text-accent underline">
            backtest
          </Link>{" "}
          and{" "}
          <Link to="/model" className="text-accent underline">
            model lab
          </Link>{" "}
          for the evidence.
        </span>
      </Callout>

      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Highest ranked" subtitle={`Most likely to ${o.target}`} action={<Link to="/screener" className="inline-flex items-center gap-1 text-xs font-medium text-accent">Screener <ArrowRight size={12} /></Link>} bodyClassName="p-0">
          <DataTable rows={o.top} columns={rankedColumns} rowKey={(r) => r.ticker} onRowClick={open} dense />
        </Card>
        <Card title="Lowest ranked" subtitle={`Least likely to ${o.target}`} bodyClassName="p-0">
          <DataTable rows={o.bottom} columns={rankedColumns} rowKey={(r) => r.ticker} onRowClick={open} dense />
        </Card>
      </div>

      <div className="grid gap-4 xl:grid-cols-5">
        <Card title="Backtest: growth of $1" subtitle={backtest.data ? `Out of sample, after costs · ${backtest.data.growth.length} rebalances` : undefined} className="xl:col-span-3">
          {backtest.data ? (
            <MultiLineChart data={backtest.data.growth} series={Object.entries(backtest.data.labels).map(([key, label]) => ({ key, label }))} yFormat={(v) => `$${v.toFixed(2)}`} height={260} reference={1} />
          ) : (
            <div className="h-64" />
          )}
        </Card>
        <Card title="Sector tilt" subtitle="Mean model probability minus 50%, in percentage points" className="xl:col-span-2">
          <HorizontalBars
            data={o.sectors.map((r) => ({ sector: r.sector, tilt: (r.mean_up_probability - 0.5) * 100 })).sort((a, b) => b.tilt - a.tilt)}
            label={(r) => r.sector}
            value="tilt"
            valueFormat={(v) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(1)}`}
          />
        </Card>
      </div>

      {o.macro && (
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          {(["tbill_3m", "treasury_10y", "term_spread", "vix"] as const).map((key) => {
            const s = o.macro!.series[key];
            if (!s) return null;
            const unit = key === "vix" ? "" : key === "term_spread" ? " pts" : "%";
            return <Stat key={key} label={s.label.replace(", %", "").replace(", percentage points", "")} value={`${num(s.latest, 2)}${unit}`} hint={`A year ago ${num(s.one_year_ago, 2)}${unit}`} />;
          })}
          {spread && (spread.latest ?? 0) < 0 && <p className="col-span-full text-xs text-warn">The yield curve is inverted.</p>}
        </div>
      )}
    </div>
  );
}
