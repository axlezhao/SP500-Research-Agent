import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router";
import { api, type DataPageData } from "../api";
import { MultiLineChart } from "../components/charts";
import { Callout, Card, DataTable, ErrorState, LoadingGrid, PageHeader, Stat, type Column } from "../components/ui";
import { date, num, pct } from "../lib/format";

type Change = DataPageData["index_changes"][number];

const changeColumns: Column<Change>[] = [
  { key: "date", label: "Date", render: (r) => <span className="num">{date(r.date)}</span> },
  { key: "added", label: "Added", render: (r) => (r.added ? <Link className="hover:text-accent" to={`/stock/${r.added}`}><span className="num font-semibold">{r.added}</span> <span className="text-muted">{r.added_name}</span></Link> : "—") },
  { key: "removed", label: "Removed", render: (r) => (r.removed ? <span><span className="num font-semibold">{r.removed}</span> <span className="text-muted">{r.removed_name}</span></span> : "—") },
  { key: "reason", label: "Reason", render: (r) => <span className="block max-w-96 truncate text-text-2" title={r.reason ?? ""}>{r.reason ?? "—"}</span> },
];

const SOURCES: { key: string; name: string; provides: string }[] = [
  { key: "prices", name: "Yahoo Finance", provides: "Daily prices, splits" },
  { key: "universe", name: "Wikipedia", provides: "Index membership history" },
  { key: "fundamentals", name: "SEC EDGAR", provides: "Point-in-time fundamentals, filings" },
  { key: "macro", name: "FRED", provides: "VIX, Treasury yields" },
];

export default function DataPage() {
  const { data, isPending, isError, error } = useQuery({ queryKey: ["data"], queryFn: api.data });
  if (isPending) return <LoadingGrid />;
  if (isError) return <ErrorState error={error} />;
  const q = data.quality;
  const eda = data.eda;
  const prices = q.prices ?? {};
  const surv = q.survivorship;
  const fund = q.fundamentals;
  const fetched: Record<string, string> = q.fetched_at ?? {};

  return (
    <div className="space-y-5">
      <PageHeader title="Data" description="Where the data comes from, how complete it is, and what the checks found. Every feature uses only information public at the close of its date." />
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Price history" value={`${prices.tickers ?? "—"} stocks`} hint={`${date(prices.start)} – ${date(prices.end)}`} />
        <Stat label="Former members covered" value={surv ? `${surv.former_members_with_prices} / ${surv.former_members}` : "—"} hint={surv ? `${pct(surv.coverage, 0)} survivorship coverage` : "Only current members in this source"} />
        <Stat label="SEC fundamentals" value={fund ? `${fund.tickers_with_data} stocks` : "—"} hint={fund ? `Median ${fund.median_days_since_last_filing} days since last filing` : "Not in this source"} />
        <Stat label="Feature rows" value={eda ? `${(eda.rows / 1e6).toFixed(2)}M` : "—"} hint={eda ? `${eda.index_member_rows ? `${(eda.index_member_rows / 1e6).toFixed(2)}M as index members` : ""}` : undefined} />
      </div>

      {data.source === "live" && (
        <Card title="Sources" bodyClassName="p-0">
          <DataTable
            rows={SOURCES}
            rowKey={(r) => r.key}
            columns={[
              { key: "name", label: "Source", render: (r) => <span className="font-medium">{r.name}</span> },
              { key: "provides", label: "Provides", render: (r) => <span className="text-text-2">{r.provides}</span> },
              { key: "fetched", label: "Last fetched", render: (r) => <span className="num text-text-2">{fetched[r.key] ? new Date(fetched[r.key]).toLocaleString("en-US", { dateStyle: "medium", timeStyle: "short" }) : "—"}</span> },
            ]}
          />
        </Card>
      )}

      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Quality checks">
          <ul className="space-y-3 text-[13px]">
            {surv && (
              <li>
                <span className="font-medium">Survivorship.</span> <span className="text-text-2">{surv.former_members_with_prices} of {surv.former_members} stocks that left the index have usable prices; {surv.dropped_no_matching_history ?? 0} were dropped because the symbol's history doesn't match their time in the index (e.g. a ticker reused by another company).</span>
              </li>
            )}
            {q.membership && (
              <li>
                <span className="font-medium">Membership.</span> <span className="text-text-2">{q.membership.tickers_ever_in_index} stocks were in the index during the period; {q.membership.unresolved_changes} of {q.membership.changes_used} index changes couldn't be matched (usually renamed tickers).</span>
              </li>
            )}
            <li>
              <span className="font-medium">Extreme moves.</span> <span className="text-text-2">{prices.extreme_daily_moves ?? 0} daily moves above 40%. Most are real events; some are corporate actions the price source hasn't adjusted.</span>
              {prices.extreme_examples?.length > 0 && (
                <ul className="mt-1.5 flex flex-wrap gap-1.5">
                  {prices.extreme_examples.map((e: { ticker: string; date: string; move: number }) => (
                    <li key={`${e.ticker}-${e.date}`} className="num rounded-md border border-border bg-surface-2 px-1.5 py-0.5 text-[11px]">
                      {e.ticker} {e.move > 0 ? "+" : "−"}
                      {Math.abs(e.move * 100).toFixed(0)}% · {e.date}
                    </li>
                  ))}
                </ul>
              )}
            </li>
            {fund && fund.failed > 0 && (
              <li>
                <span className="font-medium">SEC failures.</span> <span className="text-text-2">{fund.failed} companies could not be parsed.</span>
              </li>
            )}
          </ul>
          {Object.entries(data.excluded).map(([ticker, reason]) => (
            <div key={ticker} className="mt-3">
              <Callout tone="warn">
                <Link to={`/stock/${ticker}`} className="font-semibold underline">{ticker}</Link> is left out of today's ranking: {reason}.
              </Callout>
            </div>
          ))}
        </Card>
        {eda && (
          <Card title="Return distribution" subtitle="5-day forward returns of index members">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
              <Stat label="Mean" value={pct(eda.forward_return_5d.mean, 2)} />
              <Stat label="Std. deviation" value={pct(eda.forward_return_5d.std, 2)} />
              <Stat label="Excess kurtosis" value={num(eda.forward_return_5d.excess_kurtosis, 1)} hint="Fat tails" />
              <Stat label="1st percentile" value={pct(eda.forward_return_5d.p01, 1)} />
              <Stat label="99th percentile" value={pct(eda.forward_return_5d.p99, 1)} />
              <Stat label="Target base rate" value={pct(eda.target_rate ?? eda.up_rate_5d, 1)} hint={eda.target ? `Beat the median` : "Closed higher"} />
            </div>
          </Card>
        )}
      </div>

      {data.macro.length > 0 && (
        <div className="grid gap-4 xl:grid-cols-2">
          <Card title="VIX" subtitle="Expected 30-day volatility of the S&P 500 (FRED)">
            <MultiLineChart data={data.macro.filter((r) => r.vix !== null) as { date: string; vix: number }[]} series={[{ key: "vix", label: "VIX" }]} yFormat={(v) => v.toFixed(0)} height={240} />
          </Card>
          <Card title="Treasury yields" subtitle="Below the 3-month yield, the curve is inverted">
            <MultiLineChart data={data.macro.filter((r) => r.tbill_3m !== null && r.treasury_10y !== null) as { date: string; tbill_3m: number; treasury_10y: number }[]} series={[{ key: "tbill_3m", label: "3-month T-bill" }, { key: "treasury_10y", label: "10-year Treasury" }]} yFormat={(v) => `${v.toFixed(1)}%`} height={240} />
          </Card>
        </div>
      )}

      {data.index_changes.length > 0 && (
        <Card title="Recent S&P 500 changes" subtitle="From Wikipedia's change log" bodyClassName="p-0">
          <DataTable rows={data.index_changes} columns={changeColumns} rowKey={(r) => `${r.date}-${r.added}-${r.removed}`} maxHeight={420} dense />
        </Card>
      )}
    </div>
  );
}
