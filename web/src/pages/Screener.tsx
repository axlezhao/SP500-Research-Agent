import { useQuery } from "@tanstack/react-query";
import { Search } from "lucide-react";
import { useMemo, useState } from "react";
import { useNavigate } from "react-router";
import { api, type ScreenerRow } from "../api";
import { Callout, Card, DataTable, ErrorState, LoadingGrid, PageHeader, StanceBadge, type Column } from "../components/ui";
import { money, num, pct, price, signedPct, tone } from "../lib/format";

const columns: Column<ScreenerRow>[] = [
  { key: "rank", label: "Rank", align: "right", render: (r) => r.rank, sortValue: (r) => -r.rank },
  {
    key: "ticker",
    label: "Stock",
    sortValue: (r) => r.ticker,
    render: (r) => (
      <div className="flex min-w-0 flex-col">
        <span className="num font-semibold">{r.ticker}</span>
        <span className="max-w-52 truncate text-xs text-muted">{r.company}</span>
      </div>
    ),
  },
  { key: "sector", label: "Sector", render: (r) => <span className="text-text-2">{r.sector ?? "—"}</span>, sortValue: (r) => r.sector ?? "" },
  { key: "p", label: "Probability", align: "right", render: (r) => pct(r.model_probability), sortValue: (r) => r.model_probability },
  { key: "stance", label: "View", render: (r) => <StanceBadge stance={r.stance} /> },
  { key: "close", label: "Price", align: "right", render: (r) => price(r.close), sortValue: (r) => r.close },
  { key: "r5", label: "5D", align: "right", render: (r) => <span className={tone(r.return_5d)}>{signedPct(r.return_5d)}</span>, sortValue: (r) => r.return_5d },
  { key: "r20", label: "20D", align: "right", render: (r) => <span className={tone(r.return_20d)}>{signedPct(r.return_20d)}</span>, sortValue: (r) => r.return_20d },
  { key: "m60", label: "60D", align: "right", render: (r) => <span className={tone(r.momentum_60d)}>{signedPct(r.momentum_60d)}</span>, sortValue: (r) => r.momentum_60d },
  { key: "vol", label: "Volatility", align: "right", render: (r) => pct(r.volatility_20d, 0), sortValue: (r) => r.volatility_20d },
  { key: "cap", label: "Market cap", align: "right", render: (r) => money(r.market_cap), sortValue: (r) => r.market_cap },
  { key: "pe", label: "P/E", align: "right", render: (r) => num(r.pe_ratio, 1), sortValue: (r) => r.pe_ratio },
  { key: "margin", label: "Margin", align: "right", render: (r) => pct(r.profit_margin), sortValue: (r) => r.profit_margin },
  { key: "growth", label: "Rev. growth", align: "right", render: (r) => <span className={tone(r.revenue_growth_yoy)}>{signedPct(r.revenue_growth_yoy)}</span>, sortValue: (r) => r.revenue_growth_yoy },
];

export default function Screener() {
  const navigate = useNavigate();
  const { data, isPending, isError, error } = useQuery({ queryKey: ["stocks"], queryFn: api.stocks });
  const [query, setQuery] = useState("");
  const [sector, setSector] = useState("all");
  const [maxVol, setMaxVol] = useState("");
  const [minProb, setMinProb] = useState("");

  const sectors = useMemo(() => [...new Set((data?.stocks ?? []).map((s) => s.sector).filter(Boolean) as string[])].sort(), [data]);
  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    return (data?.stocks ?? []).filter(
      (s) =>
        (sector === "all" || s.sector === sector) &&
        (!q || s.ticker.toLowerCase().includes(q) || s.company.toLowerCase().includes(q)) &&
        (!maxVol || (s.volatility_20d ?? Infinity) <= Number(maxVol) / 100) &&
        (!minProb || s.model_probability >= Number(minProb) / 100),
    );
  }, [data, query, sector, maxVol, minProb]);

  if (isPending) return <LoadingGrid />;
  if (isError) return <ErrorState error={error} />;
  const field = "rounded-lg border border-border bg-surface px-3 py-1.5 text-[13px] outline-none focus:border-accent";

  return (
    <div className="space-y-4">
      <PageHeader title="Screener" description={`Every ranked member with its model probability to ${data.target}, price signals and point-in-time fundamentals. Click a row for the full profile.`} />
      {Object.entries(data.excluded).map(([ticker, reason]) => (
        <Callout key={ticker} tone="warn">
          <strong>{ticker}</strong> is left out of the ranking: {reason}.
        </Callout>
      ))}
      <Card bodyClassName="p-0">
        <div className="flex flex-wrap items-center gap-2 border-b border-border p-3">
          <label className="relative">
            <span className="sr-only">Filter by ticker or company</span>
            <Search size={14} className="pointer-events-none absolute top-1/2 left-2.5 -translate-y-1/2 text-muted" />
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Ticker or company" className={`${field} w-52 pl-8`} />
          </label>
          <label className="sr-only" htmlFor="sector">Sector</label>
          <select id="sector" value={sector} onChange={(e) => setSector(e.target.value)} className={field}>
            <option value="all">All sectors</option>
            {sectors.map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
          <label className="flex items-center gap-1.5 text-xs text-muted">
            Min. probability
            <input type="number" inputMode="decimal" min={0} max={100} value={minProb} onChange={(e) => setMinProb(e.target.value)} placeholder="%" className={`${field} w-20`} />
          </label>
          <label className="flex items-center gap-1.5 text-xs text-muted">
            Max. volatility
            <input type="number" inputMode="decimal" min={0} value={maxVol} onChange={(e) => setMaxVol(e.target.value)} placeholder="%" className={`${field} w-20`} />
          </label>
          <span className="ml-auto text-xs text-muted">
            {rows.length} of {data.stocks.length}
          </span>
        </div>
        <DataTable rows={rows} columns={columns} rowKey={(r) => r.ticker} initialSort={{ key: "rank", desc: true }} onRowClick={(r) => navigate(`/stock/${r.ticker}`)} maxHeight={680} />
      </Card>
    </div>
  );
}
