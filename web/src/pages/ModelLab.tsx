import { useQuery } from "@tanstack/react-query";
import { Download } from "lucide-react";
import { useMemo } from "react";
import { api, type ModelData } from "../api";
import { CalibrationChart, HorizontalBars, MultiLineChart } from "../components/charts";
import { Card, DataTable, ErrorState, LoadingGrid, PageHeader, Pill, type Column } from "../components/ui";
import { featureLabel, MODEL_LABELS, num, pct } from "../lib/format";

type Comparison = ModelData["comparison"][number];
type Signal = ModelData["signals"][number];

export default function ModelLab() {
  const { data, isPending, isError, error } = useQuery({ queryKey: ["model"], queryFn: api.model });
  const folds = useMemo(() => {
    if (!data) return [];
    const byFold = new Map<number, Record<string, number | string>>();
    for (const f of data.folds) {
      const row = byFold.get(f.fold) ?? { date: f.test_start };
      row[f.model] = f.auc;
      byFold.set(f.fold, row);
    }
    return [...byFold.values()] as { date: string }[];
  }, [data]);
  if (isPending) return <LoadingGrid />;
  if (isError) return <ErrorState error={error} />;

  const comparisonColumns: Column<Comparison>[] = [
    { key: "model", label: "Model", render: (r) => <span className="font-medium">{MODEL_LABELS[r.model] ?? r.model} {r.model === data.selected && <Pill className="ml-1">selected</Pill>}</span> },
    { key: "auc", label: "AUC", align: "right", render: (r) => `${num(r.auc_mean, 3)} ± ${num(r.auc_std, 3)}`, sortValue: (r) => r.auc_mean },
    { key: "acc", label: "Accuracy", align: "right", render: (r) => pct(r.accuracy), sortValue: (r) => r.accuracy },
    { key: "base", label: "Baseline", align: "right", render: (r) => pct(r.baseline_accuracy) },
    { key: "brier", label: "Brier", align: "right", render: (r) => num(r.brier, 4) },
    { key: "ic", label: "Rank IC", align: "right", render: (r) => num(r.ic_mean, 4), sortValue: (r) => r.ic_mean },
    { key: "t", label: "IC t-stat", align: "right", render: (r) => <span className={Math.abs(r.ic_tstat ?? 0) >= 2 ? "font-semibold" : ""}>{num(r.ic_tstat, 2)}</span> },
  ];
  const signalColumns: Column<Signal>[] = [
    { key: "signal", label: "Signal", render: (r) => <span className={r.signal === "model_probability" ? "font-semibold" : ""}>{featureLabel(r.signal)}</span> },
    { key: "ic", label: "Mean IC", align: "right", render: (r) => num(r.ic_mean, 4), sortValue: (r) => (r.ic_mean === null ? null : Math.abs(r.ic_mean)) },
    { key: "t", label: "t-stat", align: "right", render: (r) => num(r.ic_tstat, 2), sortValue: (r) => r.ic_tstat },
    { key: "pos", label: "Days IC > 0", align: "right", render: (r) => pct(r.ic_positive_share, 0) },
  ];
  const models = Object.keys(MODEL_LABELS).filter((m) => data.comparison.some((c) => c.model === m));

  return (
    <div className="space-y-5">
      <PageHeader
        title="Model lab"
        description={`Three models predicting whether a stock will ${data.target}, compared with expanding-window walk-forward validation: each fold trains on earlier dates only, with a 5-session gap so no training label overlaps the test period.`}
        actions={
          data.report_available && (
            <a href="/api/model/report" className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-surface px-3 py-1.5 text-[13px] font-medium hover:border-accent">
              <Download size={14} /> Research report
            </a>
          )
        }
      />
      <Card title="Model comparison" subtitle="Out of sample. AUC 0.5 = no skill; IC t-stat from non-overlapping dates (|t| ≥ 2 shown in bold)" bodyClassName="p-0">
        <DataTable rows={data.comparison} columns={comparisonColumns} rowKey={(r) => r.model} />
      </Card>
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="AUC by fold" subtitle="Dashed line: no skill. Later folds test later dates">
          <MultiLineChart data={folds} series={models.map((m) => ({ key: m, label: MODEL_LABELS[m] }))} yFormat={(v) => v.toFixed(3)} reference={0.5} points />
        </Card>
        <Card title="Calibration" subtitle="Predicted probability vs. how often it happened, by decile. Dashed: perfect calibration">
          <CalibrationChart data={data.calibration} />
        </Card>
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="What drives the model" subtitle="Permutation importance on the last fold's unseen data: drop in AUC when a feature is shuffled">
          <HorizontalBars data={data.importance.slice(0, 12)} label={(r) => featureLabel(r.feature)} value="importance_mean" valueFormat={(v) => v.toFixed(4)} />
        </Card>
        <Card title="Single-signal benchmark" subtitle="IC of each input on its own; a model that can't beat its best input adds little" bodyClassName="p-0">
          <DataTable rows={data.signals.filter((s) => s.ic_mean !== null)} columns={signalColumns} rowKey={(r) => r.signal} maxHeight={420} dense />
        </Card>
      </div>
      <Card title={`Model inputs (${data.features.length})`}>
        <div className="flex flex-wrap gap-1.5">
          {data.features.map((f) => (
            <Pill key={f}>{featureLabel(f)}</Pill>
          ))}
        </div>
      </Card>
    </div>
  );
}
