import { useQuery } from "@tanstack/react-query";
import { Bot, ExternalLink, FileText, Newspaper } from "lucide-react";
import { Link, useParams } from "react-router";
import { api, type FundamentalRow } from "../api";
import { ColumnChart } from "../components/charts";
import PriceChart from "../components/PriceChart";
import { Callout, Card, DataTable, ErrorState, LoadingGrid, Pill, Skeleton, StanceBadge, Stat, type Column } from "../components/ui";
import { date, money, num, ordinal, pct, price, signedPct, tone } from "../lib/format";

const fundamentalColumns: Column<FundamentalRow>[] = [
  { key: "filed", label: "Public on", render: (r) => <span className="num">{date(r.available_date)}</span> },
  { key: "period", label: "Period end", render: (r) => <span className="num text-text-2">{date(r.period_end)}</span> },
  { key: "rev", label: "Revenue (TTM)", align: "right", render: (r) => money(r.revenue_ttm) },
  { key: "ni", label: "Net income (TTM)", align: "right", render: (r) => money(r.net_income_ttm) },
  { key: "margin", label: "Margin", align: "right", render: (r) => pct(r.profit_margin) },
  { key: "growth", label: "Rev. growth", align: "right", render: (r) => <span className={tone(r.revenue_growth_yoy)}>{signedPct(r.revenue_growth_yoy)}</span> },
  { key: "equity", label: "Equity", align: "right", render: (r) => money(r.equity) },
];

function Percentiles({ items }: { items: { label: string; percentile: number }[] }) {
  return (
    <ul className="space-y-2.5">
      {items.map((item) => {
        const value = Math.round(item.percentile * 100);
        return (
          <li key={item.label}>
            <div className="mb-1 flex justify-between text-xs">
              <span className="text-text-2">{item.label}</span>
              <span className="num text-text">{ordinal(value)} pct.</span>
            </div>
            <div className="relative h-1.5 rounded-full bg-surface-2" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={value} aria-label={`${item.label} percentile`}>
              <div className="absolute inset-y-0 left-0 rounded-full bg-accent" style={{ width: `${value}%` }} />
              <div className="absolute top-1/2 h-3 w-px -translate-y-1/2 bg-border-strong" style={{ left: "50%" }} aria-hidden />
            </div>
          </li>
        );
      })}
    </ul>
  );
}

function NewsList({ ticker }: { ticker: string }) {
  const { data, isPending, isError, error } = useQuery({ queryKey: ["news", ticker], queryFn: () => api.news(ticker), staleTime: 10 * 60_000 });
  if (isPending) return <Skeleton className="h-40" />;
  if (isError) return <p className="text-[13px] text-muted">{error instanceof Error ? error.message : "Unavailable"}</p>;
  if (!data.headlines.length) return <p className="text-[13px] text-muted">{data.note ?? "No recent headlines."}</p>;
  return (
    <ul className="divide-y divide-border">
      {data.headlines.slice(0, 8).map((h, i) => (
        <li key={i} className="py-2.5 first:pt-0 last:pb-0">
          {h.url ? (
            <a href={h.url} target="_blank" rel="noopener noreferrer" className="group text-[13px] leading-snug font-medium hover:text-accent">
              {h.title} <ExternalLink size={11} className="inline opacity-0 group-hover:opacity-100" aria-hidden />
            </a>
          ) : (
            <span className="text-[13px] font-medium">{h.title}</span>
          )}
          <div className="mt-0.5 text-xs text-muted">
            {date(h.date)}
            {h.source ? ` · ${h.source}` : ""}
          </div>
        </li>
      ))}
      {data.fetched_from && <li className="pt-2 text-[11px] text-muted">Live from {data.fetched_from}</li>}
    </ul>
  );
}

const FORM_LABEL: Record<string, string> = { "10-K": "Annual report", "10-Q": "Quarterly report", "8-K": "Current report" };
const ITEM_LABEL: Record<string, string> = { "2.02": "Results", "5.02": "Leadership change", "1.01": "Material agreement", "8.01": "Other events", "7.01": "Reg FD", "5.07": "Shareholder vote", "2.03": "Debt obligation" };

function FilingsList({ ticker }: { ticker: string }) {
  const { data, isPending, isError, error } = useQuery({ queryKey: ["filings", ticker], queryFn: () => api.filings(ticker), staleTime: 30 * 60_000 });
  if (isPending) return <Skeleton className="h-40" />;
  if (isError) return <p className="text-[13px] text-muted">{error instanceof Error ? error.message : "Unavailable"}</p>;
  if (!data.filings.length) return <p className="text-[13px] text-muted">{data.note ?? "No recent filings."}</p>;
  return (
    <ul className="divide-y divide-border">
      {data.filings.slice(0, 8).map((f, i) => (
        <li key={i} className="flex items-start justify-between gap-3 py-2.5 first:pt-0 last:pb-0">
          <div className="min-w-0">
            <a href={f.url} target="_blank" rel="noopener noreferrer" className="text-[13px] font-medium hover:text-accent">
              <span className="num">{f.form}</span> · {FORM_LABEL[f.form] ?? "Filing"}
            </a>
            {f.items && (
              <div className="mt-1 flex flex-wrap gap-1">
                {f.items.split(",").map((item) => (
                  <Pill key={item}>{ITEM_LABEL[item.trim()] ?? `Item ${item.trim()}`}</Pill>
                ))}
              </div>
            )}
          </div>
          <span className="num shrink-0 text-xs text-muted">{date(f.filed)}</span>
        </li>
      ))}
      <li className="pt-2 text-[11px] text-muted">Live from SEC EDGAR</li>
    </ul>
  );
}

export default function Stock() {
  const ticker = (useParams().ticker ?? "").toUpperCase();
  const detail = useQuery({ queryKey: ["stock", ticker], queryFn: () => api.stock(ticker) });
  const prices = useQuery({ queryKey: ["prices", ticker], queryFn: () => api.prices(ticker) });
  const fundamentals = useQuery({ queryKey: ["fundamentals", ticker], queryFn: () => api.fundamentals(ticker) });

  if (detail.isPending) return <LoadingGrid />;
  if (detail.isError) return <ErrorState error={detail.error} />;
  const d = detail.data;
  const s = d.snapshot;
  const f = s?.fundamentals ?? {};
  const history = fundamentals.data?.history ?? [];
  const chronological = [...history].reverse().slice(-12);
  const question = encodeURIComponent(`Give me a research brief on ${d.ticker}: what the model says, the fundamentals trend, recent filings and news, and the main risks.`);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="num text-2xl font-semibold tracking-tight">{d.ticker}</h1>
            {s && <StanceBadge stance={s.model_view} />}
            {d.membership && !d.membership.currently_in_index && <Pill>Not in index</Pill>}
          </div>
          <p className="mt-0.5 text-[13px] text-text-2">
            {d.company}
            {d.sector ? ` · ${d.sector}` : ""}
            {d.sub_industry ? ` · ${d.sub_industry}` : ""}
          </p>
        </div>
        <div className="sm:text-right">
          <div className="num text-2xl font-semibold">{price(d.latest.close)}</div>
          <div className={`num text-[13px] ${tone(d.latest.return_1d)}`}>
            {signedPct(d.latest.return_1d, 2)} on {date(d.latest.date)}
          </div>
        </div>
      </div>

      {d.excluded_reason && <Callout tone="warn">Left out of today's ranking because of {d.excluded_reason}.</Callout>}
      {s?.data_warning && <Callout tone="warn">{s.data_warning}</Callout>}

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <Stat label="Model probability" value={s ? pct(s.model_probability) : "—"} hint="Beat the median over 5 days" />
        <Stat label="Rank" value={s ? `${s.rank} / ${s.out_of}` : "—"} />
        <Stat label="5-day return" value={signedPct(d.latest.return_5d)} tone={tone(d.latest.return_5d)} />
        <Stat label="20-day return" value={signedPct(d.latest.return_20d)} tone={tone(d.latest.return_20d)} />
        <Stat label="60-day momentum" value={signedPct(d.latest.momentum_60d)} tone={tone(d.latest.momentum_60d)} />
        <Stat label="Volatility" value={pct(d.latest.volatility_20d, 0)} hint="Annualised, 20 days" />
      </div>

      <div className="grid gap-4 xl:grid-cols-3">
        <Card title="Price" subtitle="Adjusted for splits and dividends" className="xl:col-span-2">
          {prices.data ? <PriceChart prices={prices.data.prices} /> : <Skeleton className="h-[380px]" />}
        </Card>
        <div className="space-y-4">
          <Card title="Versus other index members" subtitle="Percentile today (50th = median)">
            {d.percentiles.length ? <Percentiles items={d.percentiles} /> : <p className="text-[13px] text-muted">No cross-sectional data.</p>}
          </Card>
          <Link to={`/agent?q=${question}`} className="card flex items-center gap-3 px-4 py-3 hover:border-accent">
            <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-accent-soft text-accent">
              <Bot size={18} />
            </span>
            <span className="min-w-0">
              <span className="block text-[13px] font-medium">Ask the agent about {d.ticker}</span>
              <span className="block text-xs text-muted">Model view, fundamentals, filings and news in one brief</span>
            </span>
          </Link>
        </div>
      </div>

      <Card title="Fundamentals" subtitle={s?.fundamentals_as_of ? `Point-in-time from SEC filings; latest public ${date(s.fundamentals_as_of)}` : "From the dataset"}>
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
          <Stat label="Market cap" value={money(f.market_cap)} />
          <Stat label="P/E" value={num(f.pe_ratio, 1)} />
          <Stat label="Revenue (TTM)" value={money(f.revenue)} />
          <Stat label="Profit margin" value={pct(f.profit_margin)} />
          <Stat label="ROE" value={pct(f.roe)} />
          <Stat label={f.liabilities_to_equity != null ? "Liabilities / equity" : "Debt / equity"} value={num(f.liabilities_to_equity ?? f.debt_to_equity, 2)} />
        </div>
        {chronological.length > 1 && (
          <div className="mt-5 grid gap-6 lg:grid-cols-2">
            <div>
              <h3 className="mb-1 text-xs font-medium text-muted">Revenue, trailing 12 months</h3>
              <ColumnChart data={chronological} x="available_date" y="revenue_ttm" xFormat={(v) => date(v)} yFormat={(v) => money(v)} height={200} tooltipRows={(r) => [{ label: "Revenue (TTM)", value: money(r.revenue_ttm) }, { label: "Period end", value: date(r.period_end) }]} />
            </div>
            <div>
              <h3 className="mb-1 text-xs font-medium text-muted">Profit margin, trailing 12 months</h3>
              <ColumnChart data={chronological} x="available_date" y="profit_margin" xFormat={(v) => date(v)} yFormat={(v) => pct(v, 0)} height={200} signed tooltipRows={(r) => [{ label: "Margin", value: pct(r.profit_margin) }, { label: "Net income (TTM)", value: money(r.net_income_ttm) }]} />
            </div>
          </div>
        )}
        {history.length > 0 ? (
          <div className="mt-4 -mx-4 -mb-4 border-t border-border">
            <DataTable rows={history.slice(0, 8)} columns={fundamentalColumns} rowKey={(r) => r.available_date} dense />
          </div>
        ) : (
          fundamentals.data?.note && <p className="mt-3 text-[13px] text-muted">{fundamentals.data.note}</p>
        )}
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title={<span className="inline-flex items-center gap-1.5"><Newspaper size={14} /> Recent news</span>} subtitle="Headlines only; open an article for the full story">
          <NewsList ticker={d.ticker} />
        </Card>
        <Card title={<span className="inline-flex items-center gap-1.5"><FileText size={14} /> SEC filings</span>} subtitle="10-K, 10-Q and 8-K">
          <FilingsList ticker={d.ticker} />
        </Card>
      </div>
    </div>
  );
}
