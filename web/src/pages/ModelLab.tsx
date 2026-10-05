import { useQuery } from "@tanstack/react-query";
import { Download } from "lucide-react";
import { useMemo } from "react";
import { api, type Experiment, type ModelData } from "../api";
import { CalibrationChart, HorizontalBars, MultiLineChart } from "../components/charts";
import { Card, DataTable, ErrorState, LoadingGrid, PageHeader, Pill, type Column } from "../components/ui";
import { featureLabel, MODEL_LABELS, num, pct, signedPct, tone } from "../lib/format";

type Comparison = ModelData["comparison"][number];
type Signal = ModelData["signals"][number];
type Breakdown = NonNullable<ModelData["ic_breakdown"]>[number];
type DecayRow = { signal: string } & Record<string, number | string | null>;

const SLICE_LABELS: Record<string, string> = { year: "Year", sector: "Sector", size: "Size" };

export default function ModelLab() {
  const { data, isPending, isError, error } = useQuery({ queryKey: ["model"], queryFn: api.model });
  const folds = useMemo(() => {
    if (!data) return [];
    const byFold = new Map<number, Record<string, number | string>>();
    for (const f of data.folds) {
      const row = byFold.get(f.fold) ?? { date: f.test_start };
      row[f.model] = (f.ic_mean ?? f.auc) as number;
      byFold.set(f.fold, row);
    }
    return [...byFold.values()] as { date: string }[];
  }, [data]);
  const decay = useMemo(() => {
    if (!data?.ic_decay) return { rows: [] as DecayRow[], horizons: [] as number[] };
    const horizons = [...new Set(data.ic_decay.map((d) => d.horizon))].sort((a, b) => a - b);
    const bySignal = new Map<string, DecayRow>();
    for (const d of data.ic_decay) {
      const row: DecayRow = bySignal.get(d.signal) ?? { signal: d.signal };
      row[`h${d.horizon}`] = d.ic_mean;
      bySignal.set(d.signal, row);
    }
    return { rows: [...bySignal.values()], horizons };
  }, [data]);
  if (isPending) return <LoadingGrid />;
  if (isError) return <ErrorState error={error} />;

  const comparisonColumns: Column<Comparison>[] = [
    { key: "model", label: "Model", render: (r) => <span className="font-medium">{MODEL_LABELS[r.model] ?? r.model} {r.model === data.selected && <Pill className="ml-1">selected</Pill>}</span> },
    { key: "ic", label: "Rank IC", align: "right", render: (r) => num(r.ic_mean, 4), sortValue: (r) => r.ic_mean },
    { key: "t", label: "IC t (Newey-West)", align: "right", render: (r) => <span className={Math.abs(r.ic_tstat ?? 0) >= 2 ? "font-semibold" : ""}>{num(r.ic_tstat, 2)}</span>, sortValue: (r) => r.ic_tstat },
    { key: "t2", label: "IC t (non-overlapping)", align: "right", render: (r) => <span className="text-text-2">{num(r.ic_tstat_nonoverlap, 2)}</span> },
    { key: "auc", label: "AUC", align: "right", render: (r) => `${num(r.auc_mean, 3)} ± ${num(r.auc_std, 3)}`, sortValue: (r) => r.auc_mean },
    { key: "brier", label: "Brier", align: "right", render: (r) => num(r.brier, 4) },
  ];
  const breakdownColumns: Column<Breakdown>[] = [
    { key: "slice", label: "Slice", render: (r) => <span className="text-text-2">{SLICE_LABELS[r.slice] ?? r.slice}</span> },
    { key: "group", label: "Group", render: (r) => <span className="font-medium">{r.group}</span> },
    { key: "ic", label: "Rank IC", align: "right", render: (r) => <span className={tone(r.ic_mean)}>{num(r.ic_mean, 4)}</span>, sortValue: (r) => r.ic_mean },
    { key: "t", label: "t-stat", align: "right", render: (r) => <span className={Math.abs(r.ic_tstat ?? 0) >= 2 ? "font-semibold" : ""}>{num(r.ic_tstat, 2)}</span>, sortValue: (r) => r.ic_tstat },
    { key: "days", label: "Days", align: "right", render: (r) => r.ic_days.toLocaleString("en-US") },
  ];
  const decayColumns: Column<DecayRow>[] = [
    { key: "signal", label: "Signal", render: (r) => <span className={r.signal === "model_probability" ? "font-semibold" : ""}>{featureLabel(r.signal)}</span> },
    ...decay.horizons.map((h) => ({
      key: `h${h}`,
      label: `${h}d`,
      align: "right" as const,
      render: (r: DecayRow) => <span className={tone(r[`h${h}`] as number | null)}>{num(r[`h${h}`] as number | null, 4)}</span>,
    })),
  ];
  const signalColumns: Column<Signal>[] = [
    { key: "signal", label: "Signal", render: (r) => <span className={r.signal === "model_probability" ? "font-semibold" : ""}>{featureLabel(r.signal)}</span> },
    { key: "ic", label: "Mean IC", align: "right", render: (r) => num(r.ic_mean, 4), sortValue: (r) => (r.ic_mean === null ? null : Math.abs(r.ic_mean)) },
    { key: "t", label: "t-stat", align: "right", render: (r) => num(r.ic_tstat, 2), sortValue: (r) => r.ic_tstat },
    { key: "pos", label: "Days IC > 0", align: "right", render: (r) => pct(r.ic_positive_share, 0) },
  ];
  const models = Object.keys(MODEL_LABELS).filter((m) => data.comparison.some((c) => c.model === m));
  const reference = data.experiments.find((e) => e.reference)?.reference === "sp500" ? "S&P 500" : "Equal-weight";
  const experimentColumns: Column<Experiment>[] = [
    { key: "setup", label: "Setup", render: (r) => <span className="font-medium">{r.setup} {r.production && <Pill className="ml-1">production</Pill>}</span> },
    { key: "portfolio", label: "Portfolio", render: (r) => <span className="text-text-2">{r.portfolio ?? "—"}</span> },
    { key: "ic", label: "Rank IC", align: "right", render: (r) => num(r.ic_mean, 4), sortValue: (r) => r.ic_mean },
    { key: "t", label: "IC t-stat", align: "right", render: (r) => <span className={Math.abs(r.ic_tstat ?? 0) >= 2 ? "font-semibold" : ""}>{num(r.ic_tstat, 2)}</span>, sortValue: (r) => r.ic_tstat },
    { key: "lo", label: "Long-only CAGR", align: "right", render: (r) => signedPct(r.long_only_cagr) },
    { key: "ref", label: `${reference} CAGR`, align: "right", render: (r) => <span className="text-text-2">{signedPct(r.reference_cagr)}</span> },
    { key: "ls", label: "Long-short CAGR", align: "right", render: (r) => <span className={tone(r.long_short_cagr)}>{signedPct(r.long_short_cagr)}</span>, sortValue: (r) => r.long_short_cagr },
    {
      key: "alpha",
      label: "L/S alpha (t)",
      align: "right",
      render: (r) => (r.long_short_alpha === null ? "—" : <span className={Math.abs(r.long_short_alpha_t ?? 0) >= 2 ? "font-semibold" : ""}>{signedPct(r.long_short_alpha)} ({num(r.long_short_alpha_t, 2)})</span>),
    },
    { key: "turnover", label: "Turnover", align: "right", render: (r) => num(r.turnover, 2) },
  ];

  return (
    <div className="space-y-5">
      <PageHeader
        title="Model lab"
        description={`${data.comparison.length} models scoring the probability that a stock will ${data.target}, compared walk-forward: refit every quarter on earlier dates only, with hyperparameters chosen inside each training window and a ${data.spec.horizon}-session gap so no training label overlaps the test period. Models are ranked by the rank IC's Newey-West t-stat; the anomaly composite is a fixed-sign baseline a fitted model has to beat.`}
        actions={
          data.report_available && (
            <a href="/api/model/report" className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-surface px-3 py-1.5 text-[13px] font-medium hover:border-accent">
              <Download size={14} /> Research report
            </a>
          )
        }
      />
      {data.experiments.length > 0 && (
        <Card
          title="Research setups tried"
          subtitle={`The matching linear model (logistic for yes/no targets, ridge for ranked ones) under each horizon, peer group and target, walked forward and backtested after costs. Reference: the equal-weight universe. Bold = |t| ≥ 2.${data.robustness?.setup_with_highest_ic_tstat ? ` Highest IC t-stat: ${data.robustness.setup_with_highest_ic_tstat}.` : ""}`}
          bodyClassName="p-0"
        >
          <DataTable rows={data.experiments.filter((e) => e.auc !== null)} columns={experimentColumns} rowKey={(r) => r.setup} />
        </Card>
      )}
      <Card title={`Model comparison: ${data.spec.label}`} subtitle="Out of sample. Newey-West t-stats use every date and correct for overlapping return windows (|t| ≥ 2 in bold); AUC scores the split at the median (0.5 = no skill)" bodyClassName="p-0">
        <DataTable rows={data.comparison} columns={comparisonColumns} rowKey={(r) => r.model} />
      </Card>
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Rank IC by refit" subtitle="Each point is one quarter's out-of-sample IC; dashed line: no skill">
          <MultiLineChart data={folds} series={models.map((m) => ({ key: m, label: MODEL_LABELS[m] }))} yFormat={(v) => v.toFixed(3)} reference={data.folds.some((f) => f.ic_mean !== undefined) ? 0 : 0.5} points />
        </Card>
        <Card title="Calibration" subtitle="Predicted score vs. realised outcome (event rate, or average percentile for ranked targets), by decile. Dashed: perfect calibration">
          <CalibrationChart data={data.calibration} />
        </Card>
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="What drives the model" subtitle="Permutation importance on the last refit's unseen data: drop in ranking quality when a feature is shuffled">
          <HorizontalBars data={data.importance.slice(0, 12)} label={(r) => featureLabel(r.feature)} value="importance_mean" valueFormat={(v) => v.toFixed(4)} />
        </Card>
        <Card title="Single-signal benchmark" subtitle="IC of each input on its own; a model that can't beat its best input adds little" bodyClassName="p-0">
          <DataTable rows={data.signals.filter((s) => s.ic_mean !== null)} columns={signalColumns} rowKey={(r) => r.signal} maxHeight={420} dense />
        </Card>
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        {data.ic_breakdown && data.ic_breakdown.length > 0 && (
          <Card title="Where the ranking power lives" subtitle="The model's rank IC by year, within each sector, and within each third of the universe by size" bodyClassName="p-0">
            <DataTable rows={data.ic_breakdown} columns={breakdownColumns} rowKey={(r) => `${r.slice}-${r.group}`} maxHeight={420} dense />
          </Card>
        )}
        {decay.rows.length > 0 && (
          <Card title="How long the signals last" subtitle="Rank IC against forward returns of 1 to 63 sessions. A signal that fades within days can't pay for frequent trading" bodyClassName="p-0">
            <DataTable rows={decay.rows} columns={decayColumns} rowKey={(r) => r.signal} maxHeight={420} dense />
          </Card>
        )}
      </div>
      <Card title={`Model inputs (${data.features.length})`} subtitle="Each ranked within its date and mapped to a normal score">
        <div className="flex flex-wrap gap-1.5">
          {data.features.map((f) => (
            <Pill key={f}>{featureLabel(f)}</Pill>
          ))}
        </div>
      </Card>
    </div>
  );
}
