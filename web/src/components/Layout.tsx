import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { BarChart3, Bot, Database, FlaskConical, LayoutDashboard, Menu, Monitor, Moon, Search, Sun, Table2, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { NavLink, Outlet, useLocation, useNavigate } from "react-router";
import { api } from "../api";
import { date } from "../lib/format";
import { useThemeChoice, type ThemeChoice } from "../lib/theme";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/screener", label: "Screener", icon: Table2 },
  { to: "/agent", label: "Research agent", icon: Bot },
  { to: "/backtest", label: "Backtest", icon: BarChart3 },
  { to: "/model", label: "Model lab", icon: FlaskConical },
  { to: "/data", label: "Data", icon: Database },
];

const SOURCE_LABEL: Record<string, string> = { live: "Live public data", kaggle: "Kaggle dataset", sample: "Synthetic sample" };

function ThemeSwitch() {
  const [choice, setChoice] = useThemeChoice();
  const options: { value: ThemeChoice; icon: typeof Sun; label: string }[] = [
    { value: "light", icon: Sun, label: "Light theme" },
    { value: "system", icon: Monitor, label: "System theme" },
    { value: "dark", icon: Moon, label: "Dark theme" },
  ];
  return (
    <div role="radiogroup" aria-label="Theme" className="inline-flex rounded-lg border border-border bg-surface-2 p-0.5">
      {options.map(({ value, icon: Icon, label }) => (
        <button key={value} type="button" role="radio" aria-checked={choice === value} aria-label={label} title={label} onClick={() => setChoice(value)} className={clsx("rounded-md p-1.5", choice === value ? "bg-surface text-text shadow-sm" : "text-muted hover:text-text")}>
          <Icon size={14} />
        </button>
      ))}
    </div>
  );
}

function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const { data } = useQuery({ queryKey: ["overview"], queryFn: api.overview });
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2.5 px-4 py-4">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent text-white">
          <svg viewBox="0 0 32 32" width="18" height="18" aria-hidden>
            <path d="M5 22l7-7 5 5 10-11" stroke="currentColor" strokeWidth="3.2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
        <div className="leading-tight">
          <div className="text-[13px] font-semibold">S&amp;P 500 Research</div>
          <div className="text-[11px] text-muted">Models · Backtests · Agent</div>
        </div>
      </div>
      <nav aria-label="Main" className="flex-1 space-y-0.5 px-2">
        {NAV.map(({ to, label, icon: Icon, end }) => (
          <NavLink key={to} to={to} end={end} onClick={onNavigate} className={({ isActive }) => clsx("flex items-center gap-2.5 rounded-md px-2.5 py-2 text-[13px] font-medium transition-colors", isActive ? "bg-accent-soft text-accent" : "text-text-2 hover:bg-surface-2 hover:text-text")}>
            <Icon size={16} aria-hidden />
            {label}
          </NavLink>
        ))}
      </nav>
      <div className="space-y-3 border-t border-border px-4 py-4 text-xs">
        {data && (
          <div className="space-y-1">
            <div className="flex items-center gap-1.5 text-text-2">
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-pos" aria-hidden />
              {SOURCE_LABEL[data.data_source] ?? data.data_source}
            </div>
            <div className="text-muted">
              Data as of <span className="num text-text-2">{date(data.as_of_date)}</span>
            </div>
          </div>
        )}
        <ThemeSwitch />
        <p className="text-[11px] leading-snug text-muted">Educational research output, not investment advice.</p>
      </div>
    </div>
  );
}

function TickerSearch() {
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const { data } = useQuery({ queryKey: ["search", query], queryFn: () => api.search(query), enabled: query.trim().length > 0, staleTime: 60_000 });
  const matches = data?.matches ?? [];

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen(true);
      }
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  useEffect(() => {
    if (open) setTimeout(() => input.current?.focus(), 0);
    else setQuery("");
  }, [open]);
  useEffect(() => {
    setActive(0);
  }, [query]);

  const go = (ticker: string) => {
    setOpen(false);
    navigate(`/stock/${encodeURIComponent(ticker)}`);
  };

  return (
    <>
      <button type="button" onClick={() => setOpen(true)} className="flex w-full max-w-sm items-center gap-2 rounded-lg border border-border bg-surface px-3 py-1.5 text-left text-[13px] text-muted hover:border-border-strong">
        <Search size={14} aria-hidden />
        <span className="flex-1">Search ticker or company</span>
        <kbd className="hidden rounded border border-border px-1.5 text-[10px] text-muted sm:inline">⌘K</kbd>
      </button>
      {open && (
        <div className="fixed inset-0 z-50 flex items-start justify-center bg-black/40 px-4 pt-[12vh]" onClick={() => setOpen(false)}>
          <div role="dialog" aria-modal="true" aria-label="Search stocks" className="card w-full max-w-lg overflow-hidden shadow-2xl" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center gap-2 border-b border-border px-3">
              <Search size={16} className="text-muted" aria-hidden />
              <input
                ref={input}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "ArrowDown") setActive((a) => Math.min(a + 1, matches.length - 1));
                  if (e.key === "ArrowUp") setActive((a) => Math.max(a - 1, 0));
                  if (e.key === "Enter") {
                    if (matches[active]) go(matches[active].ticker);
                    else if (query.trim()) go(query.trim().toUpperCase());
                  }
                }}
                placeholder="AAPL, Microsoft, bank..."
                aria-label="Ticker or company"
                className="w-full bg-transparent py-3 text-sm outline-none placeholder:text-muted"
              />
              <button type="button" aria-label="Close search" onClick={() => setOpen(false)} className="text-muted hover:text-text">
                <X size={16} />
              </button>
            </div>
            <ul role="listbox" className="max-h-80 overflow-auto py-1">
              {matches.map((m, i) => (
                <li key={m.ticker} role="option" aria-selected={i === active}>
                  <button type="button" onMouseEnter={() => setActive(i)} onClick={() => go(m.ticker)} className={clsx("flex w-full items-center gap-3 px-3 py-2 text-left text-[13px]", i === active && "bg-surface-2")}>
                    <span className="num w-16 font-semibold">{m.ticker}</span>
                    <span className="flex-1 truncate text-text-2">{m.company}</span>
                    <span className="text-xs text-muted">{m.sector}</span>
                  </button>
                </li>
              ))}
              {query && matches.length === 0 && <li className="px-3 py-3 text-[13px] text-muted">No matches. Press Enter to open “{query.toUpperCase()}”.</li>}
              {!query && <li className="px-3 py-3 text-[13px] text-muted">Type a ticker or part of a company name.</li>}
            </ul>
          </div>
        </div>
      )}
    </>
  );
}

export default function Layout() {
  const [drawer, setDrawer] = useState(false);
  const location = useLocation();
  useEffect(() => {
    setDrawer(false);
  }, [location.pathname]);
  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[232px_1fr]">
      <aside className="sticky top-0 hidden h-screen border-r border-border bg-surface lg:block">
        <Sidebar />
      </aside>
      {drawer && (
        <div className="fixed inset-0 z-40 lg:hidden" onClick={() => setDrawer(false)}>
          <div className="absolute inset-0 bg-black/40" />
          <aside className="absolute inset-y-0 left-0 w-64 border-r border-border bg-surface" onClick={(e) => e.stopPropagation()}>
            <Sidebar onNavigate={() => setDrawer(false)} />
          </aside>
        </div>
      )}
      <div className="min-w-0">
        <header className="sticky top-0 z-30 flex items-center gap-3 border-b border-border bg-bg/85 px-4 py-2.5 backdrop-blur lg:px-6">
          <button type="button" aria-label="Open menu" onClick={() => setDrawer(true)} className="rounded-md p-1.5 text-text-2 hover:bg-surface-2 lg:hidden">
            <Menu size={18} />
          </button>
          <TickerSearch />
          <div className="flex-1" />
          <NavLink to="/agent" className="hidden items-center gap-1.5 rounded-lg bg-accent px-3 py-1.5 text-[13px] font-medium text-white hover:opacity-90 sm:inline-flex">
            <Bot size={15} aria-hidden />
            Ask the agent
          </NavLink>
        </header>
        <main className="mx-auto max-w-[1400px] px-4 py-6 lg:px-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
