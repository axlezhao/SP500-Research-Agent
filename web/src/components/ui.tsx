import clsx from "clsx";
import { AlertTriangle, ArrowDown, ArrowUp, Info } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";
import type { Stance } from "../api";

export function Card({ title, subtitle, action, children, className, bodyClassName }: { title?: ReactNode; subtitle?: ReactNode; action?: ReactNode; children: ReactNode; className?: string; bodyClassName?: string }) {
  return (
    <section className={clsx("card min-w-0", className)}>
      {(title || action) && (
        <header className="flex items-start justify-between gap-3 border-b border-border px-4 py-3">
          <div className="min-w-0">
            {title && <h2 className="text-[13px] font-semibold tracking-tight text-text">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-muted">{subtitle}</p>}
          </div>
          {action}
        </header>
      )}
      <div className={clsx("p-4", bodyClassName)}>{children}</div>
    </section>
  );
}

export function Stat({ label, value, hint, tone, help }: { label: string; value: ReactNode; hint?: ReactNode; tone?: string; help?: string }) {
  return (
    <div className="card px-4 py-3">
      <div className="flex items-center gap-1 text-xs font-medium text-muted">
        {label}
        {help && (
          <span title={help} className="cursor-help">
            <Info size={12} aria-label={help} />
          </span>
        )}
      </div>
      <div className={clsx("num mt-1 text-xl font-semibold", tone ?? "text-text")}>{value}</div>
      {hint && <div className="mt-0.5 text-xs text-text-2">{hint}</div>}
    </div>
  );
}

const STANCE_STYLE: Record<Stance, string> = {
  constructive: "bg-accent-soft text-accent",
  neutral: "bg-surface-2 text-text-2",
  cautious: "bg-warn-soft text-warn",
};

export function StanceBadge({ stance }: { stance: Stance }) {
  const Icon = stance === "constructive" ? ArrowUp : stance === "cautious" ? ArrowDown : null;
  return (
    <span className={clsx("inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium capitalize", STANCE_STYLE[stance])}>
      {Icon && <Icon size={11} aria-hidden />}
      {stance}
    </span>
  );
}

export function Pill({ children, className }: { children: ReactNode; className?: string }) {
  return <span className={clsx("inline-flex items-center rounded-md border border-border bg-surface-2 px-1.5 py-0.5 text-[11px] font-medium text-text-2", className)}>{children}</span>;
}

export function Callout({ children, tone = "info" }: { children: ReactNode; tone?: "info" | "warn" }) {
  return (
    <div className={clsx("flex gap-2.5 rounded-lg border px-3 py-2.5 text-[13px]", tone === "warn" ? "border-warn/30 bg-warn-soft text-text" : "border-accent/25 bg-accent-soft text-text")}>
      {tone === "warn" ? <AlertTriangle size={16} className="mt-0.5 shrink-0 text-warn" /> : <Info size={16} className="mt-0.5 shrink-0 text-accent" />}
      <div className="min-w-0">{children}</div>
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={clsx("animate-pulse rounded-md bg-surface-2", className)} />;
}

export function LoadingGrid() {
  return (
    <div className="space-y-4" aria-busy="true" aria-label="Loading">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {Array.from({ length: 4 }, (_, i) => (
          <Skeleton key={i} className="h-20" />
        ))}
      </div>
      <Skeleton className="h-72" />
    </div>
  );
}

export function ErrorState({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <Callout tone="warn">
      <p className="font-medium">Couldn't load this view.</p>
      <p className="mt-0.5 text-text-2">{message}</p>
    </Callout>
  );
}

export function PageHeader({ title, description, actions }: { title: ReactNode; description?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
      <div className="min-w-0">
        <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
        {description && <p className="mt-1 max-w-3xl text-[13px] text-text-2">{description}</p>}
      </div>
      {actions}
    </div>
  );
}

export interface Column<T> {
  key: string;
  label: string;
  align?: "left" | "right";
  render: (row: T) => ReactNode;
  sortValue?: (row: T) => number | string | null | undefined;
  className?: string;
}

/** Sortable table; click a header to sort. Rows can link via onRowClick. */
export function DataTable<T>({ rows, columns, initialSort, onRowClick, rowKey, maxHeight, dense }: { rows: T[]; columns: Column<T>[]; initialSort?: { key: string; desc: boolean }; onRowClick?: (row: T) => void; rowKey: (row: T) => string; maxHeight?: number; dense?: boolean }) {
  const [sort, setSort] = useState(initialSort);
  const sorted = useMemo(() => {
    const column = columns.find((c) => c.key === sort?.key);
    if (!sort || !column?.sortValue) return rows;
    const value = column.sortValue;
    return [...rows].sort((a, b) => {
      const va = value(a);
      const vb = value(b);
      if (va === null || va === undefined) return 1;
      if (vb === null || vb === undefined) return -1;
      const cmp = typeof va === "string" ? va.localeCompare(String(vb)) : (va as number) - (vb as number);
      return sort.desc ? -cmp : cmp;
    });
  }, [rows, columns, sort]);

  return (
    <div className="overflow-auto" style={maxHeight ? { maxHeight } : undefined}>
      <table className="w-full border-collapse text-[13px]">
        <thead className="sticky top-0 z-10 bg-surface">
          <tr>
            {columns.map((c) => {
              const active = sort?.key === c.key;
              return (
                <th key={c.key} scope="col" aria-sort={active ? (sort!.desc ? "descending" : "ascending") : undefined} className={clsx("border-b border-border px-3 py-2 text-xs font-medium whitespace-nowrap text-muted", c.align === "right" ? "text-right" : "text-left")}>
                  {c.sortValue ? (
                    <button type="button" className={clsx("inline-flex items-center gap-1 hover:text-text", active && "text-text")} onClick={() => setSort({ key: c.key, desc: active ? !sort!.desc : true })}>
                      {c.label}
                      {active && (sort!.desc ? <ArrowDown size={11} /> : <ArrowUp size={11} />)}
                    </button>
                  ) : (
                    c.label
                  )}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => (
            <tr key={rowKey(row)} onClick={onRowClick ? () => onRowClick(row) : undefined} className={clsx("border-b border-border/70 last:border-0", onRowClick && "cursor-pointer hover:bg-surface-2")}>
              {columns.map((c) => (
                <td key={c.key} className={clsx("px-3 whitespace-nowrap", dense ? "py-1.5" : "py-2", c.align === "right" && "num text-right", c.className)}>
                  {c.render(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Segmented<T extends string>({ value, options, onChange, label }: { value: T; options: { value: T; label: string }[]; onChange: (v: T) => void; label: string }) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-lg border border-border bg-surface-2 p-0.5">
      {options.map((o) => (
        <button key={o.value} type="button" role="radio" aria-checked={value === o.value} onClick={() => onChange(o.value)} className={clsx("rounded-md px-2.5 py-1 text-xs font-medium transition-colors", value === o.value ? "bg-surface text-text shadow-sm" : "text-muted hover:text-text")}>
          {o.label}
        </button>
      ))}
    </div>
  );
}
