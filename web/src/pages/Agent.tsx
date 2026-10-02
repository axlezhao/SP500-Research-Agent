import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { ArrowUp, Bot, Check, ChevronRight, Loader2, RotateCcw, Square, TriangleAlert, User } from "lucide-react";
import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router";
import { api, resetChat, streamChat, type AgentEvent } from "../api";
import { Pill } from "../components/ui";

const Markdown = lazy(() => import("../components/Markdown"));

interface Step {
  name: string;
  arguments: Record<string, unknown>;
  status: "running" | "done" | "error";
  error?: string | null;
  result?: unknown;
}
interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  steps?: Step[];
  pending?: boolean;
  meta?: string;
  error?: string;
}

const SUGGESTIONS = [
  "Which stocks does the model rank highest, and how much should I trust it?",
  "Compare NVDA and AMD on growth, valuation and recent filings",
  "What does the backtest say about the strategy after costs?",
  "Find low-volatility stocks with improving revenue growth",
  "What's the market backdrop: VIX, rates and the yield curve?",
  "Which stocks joined or left the S&P 500 recently?",
];

const TOOL_LABEL: Record<string, string> = {
  dataset_overview: "Checking the dataset and model quality",
  search_companies: "Searching companies",
  stock_snapshot: "Reading the stock snapshot",
  rank_stocks: "Ranking stocks",
  screen_stocks: "Screening stocks",
  compare_stocks: "Comparing stocks",
  price_history: "Loading price history",
  recent_news: "Fetching recent news",
  sector_summary: "Summarising sectors",
  fundamentals_history: "Reading SEC fundamentals",
  macro_snapshot: "Checking market conditions",
  index_changes: "Looking up index changes",
  recent_filings: "Fetching SEC filings",
  model_performance: "Reviewing model performance",
  backtest_results: "Reviewing the backtest",
};

function storage<T>(key: string, fallback: T): T {
  try {
    const raw = sessionStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}
function save(key: string, value: unknown) {
  try {
    sessionStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable: the conversation still works for this page view */
  }
}
const newId = () => (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`);

function describeArgs(args: Record<string, unknown>) {
  const parts = Object.entries(args).map(([k, v]) => `${k}=${Array.isArray(v) ? v.join(",") : String(v)}`);
  return parts.length ? parts.join(" · ") : "";
}

function Steps({ steps }: { steps: Step[] }) {
  const [open, setOpen] = useState<number | null>(null);
  return (
    <ol className="mb-3 space-y-1">
      {steps.map((step, i) => (
        <li key={i} className="text-xs">
          <button type="button" onClick={() => setOpen(open === i ? null : i)} className="flex w-full items-center gap-2 rounded-md px-2 py-1 text-left text-text-2 hover:bg-surface-2" aria-expanded={open === i}>
            {step.status === "running" ? <Loader2 size={13} className="animate-spin text-accent" /> : step.status === "error" ? <TriangleAlert size={13} className="text-warn" /> : <Check size={13} className="text-pos" />}
            <span className="font-medium text-text">{TOOL_LABEL[step.name] ?? step.name}</span>
            <span className="num truncate text-muted">{describeArgs(step.arguments)}</span>
            <ChevronRight size={12} className={clsx("ml-auto shrink-0 text-muted transition-transform", open === i && "rotate-90")} />
          </button>
          {open === i && (
            <pre className="num mt-1 max-h-64 overflow-auto rounded-md border border-border bg-surface-2 p-2 text-[11px] whitespace-pre-wrap text-text-2">
              {step.error ? `Error: ${step.error}` : JSON.stringify(step.result, null, 2)}
            </pre>
          )}
        </li>
      ))}
    </ol>
  );
}

export default function Agent() {
  const status = useQuery({ queryKey: ["agent-status"], queryFn: api.agentStatus });
  const [sessionId, setSessionId] = useState<string>(() => storage("agent-session", newId()));
  const [messages, setMessages] = useState<Message[]>(() => storage("agent-messages", []));
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);
  const [params, setParams] = useSearchParams();

  useEffect(() => {
    save("agent-session", sessionId);
  }, [sessionId]);
  useEffect(() => {
    save("agent-messages", messages.filter((m) => !m.pending));
  }, [messages]);
  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  const update = (id: string, change: (m: Message) => Message) => setMessages((all) => all.map((m) => (m.id === id ? change(m) : m)));

  const send = useCallback(
    async (text: string) => {
      const question = text.trim();
      if (!question || busy) return;
      const replyId = newId();
      setMessages((all) => [...all, { id: newId(), role: "user", text: question }, { id: replyId, role: "assistant", text: "", steps: [], pending: true }]);
      setInput("");
      setBusy(true);
      abort.current = new AbortController();
      const onEvent = (event: AgentEvent) => {
        if (event.type === "tool_start") update(replyId, (m) => ({ ...m, steps: [...(m.steps ?? []), { name: event.name, arguments: event.arguments, status: "running" }] }));
        if (event.type === "tool_end")
          update(replyId, (m) => {
            const steps = [...(m.steps ?? [])];
            const index = steps.map((s) => s.name === event.name && s.status === "running").lastIndexOf(true);
            if (index >= 0) steps[index] = { ...steps[index], status: event.error ? "error" : "done", error: event.error, result: event.result };
            return { ...m, steps };
          });
        if (event.type === "answer") {
          const tokens = event.usage?.input_tokens ? ` · ${(event.usage.input_tokens + (event.usage.output_tokens ?? 0)).toLocaleString()} tokens` : "";
          update(replyId, (m) => ({ ...m, text: event.text, pending: false, meta: event.provider === "rule-based" ? "Rule-based assistant (no LLM key configured)" : `${event.model}${tokens}` }));
        }
        if (event.type === "error") update(replyId, (m) => ({ ...m, pending: false, error: event.message }));
      };
      try {
        await streamChat(sessionId, question, onEvent, abort.current.signal);
      } catch (err) {
        const message = err instanceof DOMException && err.name === "AbortError" ? "Stopped." : err instanceof Error ? err.message : String(err);
        update(replyId, (m) => ({ ...m, pending: false, error: message }));
      } finally {
        update(replyId, (m) => (m.pending ? { ...m, pending: false, error: m.error ?? "The connection closed before an answer arrived." } : m));
        setBusy(false);
      }
    },
    [busy, sessionId],
  );

  // A question passed from another page (e.g. "Ask the agent about NVDA") is sent once.
  const linked = useRef<string | null>(null);
  useEffect(() => {
    const q = params.get("q");
    if (q && linked.current !== q) {
      linked.current = q;
      setParams({}, { replace: true });
      void send(q);
    }
  }, [params, send, setParams]);

  const reset = async () => {
    abort.current?.abort();
    await resetChat(sessionId).catch(() => undefined);
    setSessionId(newId());
    setMessages([]);
  };

  const providerLabel = status.data?.mode === "llm" ? (status.data.providers[0] === "anthropic" ? "Claude" : "DeepSeek") : "Rule-based";

  return (
    <div className="mx-auto flex max-w-3xl flex-col" style={{ minHeight: "calc(100vh - 110px)" }}>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Research agent</h1>
          <p className="text-[13px] text-text-2">Answers from 15 research tools over the data, the model, the backtest, SEC filings and live news.</p>
        </div>
        <div className="flex items-center gap-2">
          {status.data && <Pill>{providerLabel}</Pill>}
          <button type="button" onClick={reset} className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-surface px-2.5 py-1.5 text-xs font-medium text-text-2 hover:text-text">
            <RotateCcw size={13} /> New chat
          </button>
        </div>
      </div>

      <div className="flex-1 space-y-5 pb-4" aria-live="polite">
        {messages.length === 0 && (
          <div className="card p-5">
            <div className="mb-3 flex items-center gap-2 text-[13px] font-medium">
              <Bot size={16} className="text-accent" /> Try one of these
            </div>
            <div className="grid gap-2 sm:grid-cols-2">
              {SUGGESTIONS.map((s) => (
                <button key={s} type="button" onClick={() => send(s)} className="rounded-lg border border-border bg-surface-2 px-3 py-2.5 text-left text-[13px] text-text-2 hover:border-accent hover:text-text">
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m) =>
          m.role === "user" ? (
            <div key={m.id} className="flex justify-end gap-2">
              <div className="max-w-[85%] rounded-2xl rounded-br-md bg-accent px-4 py-2.5 text-[13px] text-white">{m.text}</div>
              <span className="mt-1 hidden h-7 w-7 shrink-0 items-center justify-center rounded-full bg-surface-2 text-text-2 sm:flex" aria-hidden>
                <User size={14} />
              </span>
            </div>
          ) : (
            <div key={m.id} className="flex gap-3">
              <span className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-accent-soft text-accent" aria-hidden>
                <Bot size={15} />
              </span>
              <div className="card min-w-0 flex-1 px-4 py-3">
                {m.steps && m.steps.length > 0 && <Steps steps={m.steps} />}
                {m.pending && !m.text && (
                  <div className="flex items-center gap-2 text-[13px] text-muted">
                    <Loader2 size={14} className="animate-spin" /> {m.steps?.length ? "Writing the answer…" : "Thinking…"}
                  </div>
                )}
                {m.text && (
                  <Suspense fallback={<p className="text-[13px]">{m.text}</p>}>
                    <Markdown text={m.text} />
                  </Suspense>
                )}
                {m.error && <p className="text-[13px] text-warn">{m.error}</p>}
                {m.meta && <p className="mt-2 border-t border-border pt-2 text-[11px] text-muted">{m.meta}</p>}
              </div>
            </div>
          ),
        )}
        <div ref={bottom} />
      </div>

      <form
        className="sticky bottom-0 bg-bg pt-2 pb-4"
        onSubmit={(e) => {
          e.preventDefault();
          void send(input);
        }}
      >
        <div className="card flex items-end gap-2 p-2 focus-within:border-accent">
          <label htmlFor="agent-input" className="sr-only">
            Ask the research agent
          </label>
          <textarea
            id="agent-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void send(input);
              }
            }}
            rows={1}
            placeholder="Ask about a stock, a sector, the model or the market…"
            className="max-h-40 min-h-9 flex-1 resize-none bg-transparent px-2 py-1.5 text-[13px] outline-none placeholder:text-muted"
          />
          {busy ? (
            <button type="button" onClick={() => abort.current?.abort()} aria-label="Stop" className="flex h-8 w-8 items-center justify-center rounded-lg bg-surface-2 text-text">
              <Square size={13} />
            </button>
          ) : (
            <button type="submit" disabled={!input.trim()} aria-label="Send" className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent text-white disabled:opacity-40">
              <ArrowUp size={16} />
            </button>
          )}
        </div>
        <p className="mt-1.5 text-center text-[11px] text-muted">
          {status.data?.demo && status.data.limits.per_hour ? `Public demo: up to ${status.data.limits.per_hour} questions per hour. ` : ""}
          Research output for education, not investment advice. Answers can be wrong; check the numbers in the linked views.
        </p>
      </form>
    </div>
  );
}
