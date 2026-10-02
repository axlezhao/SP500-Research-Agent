import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import { useChartColors } from "../lib/theme";

type Fmt = (v: number) => string;
const shortDate = (v: string) => {
  const d = new Date(`${v}T00:00:00`);
  return `${d.toLocaleDateString("en-US", { month: "short" })} '${String(d.getFullYear()).slice(2)}`;
};
const longDate = (v: string) => new Date(`${v}T00:00:00`).toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" });

export interface SeriesSpec {
  key: string;
  label: string;
}

/** Legend in text colours with a coloured swatch (identity never relies on colour alone: tooltips name every series). */
export function Legend({ series }: { series: SeriesSpec[] }) {
  const colors = useChartColors();
  if (series.length < 2) return null;
  return (
    <ul className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-text-2">
      {series.map((s, i) => (
        <li key={s.key} className="flex items-center gap-1.5">
          <span className="inline-block h-0.5 w-3.5 rounded-full" style={{ background: colors.series[i] }} aria-hidden />
          {s.label}
        </li>
      ))}
    </ul>
  );
}

function TooltipBox({ title, rows }: { title: string; rows: { label: string; value: string; color?: string }[] }) {
  return (
    <div className="card min-w-40 px-3 py-2 text-xs shadow-lg">
      <div className="mb-1 font-medium text-text">{title}</div>
      {rows.map((r) => (
        <div key={r.label} className="flex items-center justify-between gap-4">
          <span className="flex items-center gap-1.5 text-text-2">
            {r.color && <span className="inline-block h-2 w-2 rounded-full" style={{ background: r.color }} aria-hidden />}
            {r.label}
          </span>
          <span className="num font-medium text-text">{r.value}</span>
        </div>
      ))}
    </div>
  );
}

/** Lines over time. `points` draws straight segments with markers, for short series (e.g. 5 folds) where smoothing would invent values. */
export function MultiLineChart<T extends { date: string }>({ data, series, yFormat, height = 280, reference, points = false }: { data: T[]; series: SeriesSpec[]; yFormat: Fmt; height?: number; reference?: number; points?: boolean }) {
  const colors = useChartColors();
  return (
    <div>
      <Legend series={series} />
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke={colors.grid} vertical={false} />
          <XAxis dataKey="date" tickFormatter={shortDate} tick={{ fill: colors.muted, fontSize: 11 }} axisLine={{ stroke: colors.border }} tickLine={false} minTickGap={40} />
          <YAxis tickFormatter={yFormat} tick={{ fill: colors.muted, fontSize: 11 }} axisLine={false} tickLine={false} width={56} domain={["auto", "auto"]} />
          {reference !== undefined && <ReferenceLine y={reference} stroke={colors.muted} strokeDasharray="4 4" />}
          <Tooltip
            cursor={{ stroke: colors.muted, strokeWidth: 1 }}
            content={({ active, payload, label }) =>
              active && payload?.length ? (
                <TooltipBox title={longDate(String(label))} rows={series.map((s, i) => ({ label: s.label, color: colors.series[i], value: yFormat(Number((payload[0].payload as Record<string, number>)[s.key])) }))} />
              ) : null
            }
          />
          {series.map((s, i) => (
            <Line key={s.key} type={points ? "linear" : "monotone"} dataKey={s.key} name={s.label} stroke={colors.series[i]} strokeWidth={2} dot={points ? { r: 4, strokeWidth: 2, stroke: colors.surface, fill: colors.series[i] } : false} activeDot={{ r: 5, strokeWidth: 2, stroke: colors.surface }} isAnimationActive={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ColumnChart<T>({ data, x, y, xFormat, yFormat, height = 260, signed = false, tooltipRows }: { data: T[]; x: keyof T & string; y: keyof T & string; xFormat?: (v: string) => string; yFormat: Fmt; height?: number; signed?: boolean; tooltipRows?: (row: T) => { label: string; value: string }[] }) {
  const colors = useChartColors();
  const rows = data as unknown as Record<string, unknown>[];
  const xKey: string = x;
  const yKey: string = y;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: 0 }} barCategoryGap="30%">
        <CartesianGrid stroke={colors.grid} vertical={false} />
        <XAxis dataKey={xKey} tickFormatter={xFormat} tick={{ fill: colors.muted, fontSize: 11 }} axisLine={{ stroke: colors.border }} tickLine={false} />
        <YAxis tickFormatter={yFormat} tick={{ fill: colors.muted, fontSize: 11 }} axisLine={false} tickLine={false} width={56} />
        <ReferenceLine y={0} stroke={colors.muted} />
        <Tooltip
          cursor={{ fill: colors.grid }}
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const row = payload[0].payload as T;
            const title = xFormat ? xFormat(String(row[x])) : String(row[x]);
            return <TooltipBox title={title} rows={tooltipRows ? tooltipRows(row) : [{ label: "Value", value: yFormat(Number(row[y])) }]} />;
          }}
        />
        <Bar dataKey={yKey} radius={[4, 4, 0, 0]} isAnimationActive={false}>
          {data.map((row, i) => (
            <Cell key={i} fill={signed && Number(row[y]) < 0 ? colors.series[1] : colors.series[0]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function HorizontalBars<T>({ data, label, value, valueFormat, height, domain }: { data: T[]; label: (row: T) => string; value: keyof T & string; valueFormat: Fmt; height?: number; domain?: [number, number] }) {
  const colors = useChartColors();
  const rows = data.map((row) => ({ name: label(row), value: Number(row[value]) }));
  return (
    <ResponsiveContainer width="100%" height={height ?? Math.max(140, rows.length * 28 + 24)}>
      <BarChart data={rows} layout="vertical" margin={{ top: 0, right: 16, bottom: 0, left: 0 }} barCategoryGap="28%">
        <CartesianGrid stroke={colors.grid} horizontal={false} />
        <XAxis type="number" tickFormatter={valueFormat} tick={{ fill: colors.muted, fontSize: 11 }} axisLine={false} tickLine={false} domain={domain ?? ["auto", "auto"]} />
        <YAxis type="category" dataKey="name" width={170} tick={{ fill: colors.text2, fontSize: 12 }} axisLine={false} tickLine={false} />
        <ReferenceLine x={0} stroke={colors.muted} />
        <Tooltip cursor={{ fill: colors.grid }} content={({ active, payload }) => (active && payload?.length ? <TooltipBox title={String(payload[0].payload.name)} rows={[{ label: "Value", value: valueFormat(Number(payload[0].payload.value)) }]} /> : null)} />
        <Bar dataKey="value" fill={colors.series[0]} radius={[0, 4, 4, 0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function CalibrationChart({ data, height = 280 }: { data: { bucket: number; mean_predicted: number; actual_rate: number; rows: number }[]; height?: number }) {
  const colors = useChartColors();
  const values = data.flatMap((d) => [d.mean_predicted, d.actual_rate]);
  const pad = 0.02;
  const lo = Math.max(0, Math.floor((Math.min(...values) - pad) * 50) / 50);
  const hi = Math.min(1, Math.ceil((Math.max(...values) + pad) * 50) / 50);
  const fmt = (v: number) => `${(v * 100).toFixed(0)}%`;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid stroke={colors.grid} />
        <XAxis type="number" dataKey="mean_predicted" name="Predicted" domain={[lo, hi]} tickFormatter={fmt} tick={{ fill: colors.muted, fontSize: 11 }} axisLine={{ stroke: colors.border }} tickLine={false} label={{ value: "Mean predicted probability", position: "insideBottom", offset: -4, fill: colors.muted, fontSize: 11 }} />
        <YAxis type="number" dataKey="actual_rate" name="Actual" domain={[lo, hi]} tickFormatter={fmt} tick={{ fill: colors.muted, fontSize: 11 }} axisLine={false} tickLine={false} width={48} />
        <ZAxis range={[70, 70]} />
        <ReferenceLine segment={[{ x: lo, y: lo }, { x: hi, y: hi }]} stroke={colors.muted} strokeDasharray="4 4" />
        <Tooltip
          cursor={false}
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const d = payload[0].payload as (typeof data)[number];
            return <TooltipBox title={`Decile ${d.bucket}`} rows={[{ label: "Predicted", value: `${(d.mean_predicted * 100).toFixed(1)}%` }, { label: "Actual", value: `${(d.actual_rate * 100).toFixed(1)}%` }, { label: "Rows", value: d.rows.toLocaleString() }]} />;
          }}
        />
        <Scatter data={data} fill={colors.series[0]} line={{ stroke: colors.series[0], strokeWidth: 2 }} isAnimationActive={false} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}
