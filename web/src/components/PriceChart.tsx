import { AreaSeries, createChart, HistogramSeries, type IChartApi, type ISeriesApi, type UTCTimestamp } from "lightweight-charts";
import { useEffect, useMemo, useRef, useState } from "react";
import type { PricePoint } from "../api";
import { useChartColors } from "../lib/theme";
import { Segmented } from "./ui";

const RANGES = [
  { value: "1M", label: "1M", days: 21 },
  { value: "3M", label: "3M", days: 63 },
  { value: "6M", label: "6M", days: 126 },
  { value: "1Y", label: "1Y", days: 252 },
  { value: "3Y", label: "3Y", days: 756 },
  { value: "5Y", label: "5Y", days: 1260 },
  { value: "MAX", label: "Max", days: Infinity },
] as const;
type Range = (typeof RANGES)[number]["value"];

const toTime = (d: string) => (Date.parse(`${d}T00:00:00Z`) / 1000) as UTCTimestamp;

function withAlpha(color: string, alpha: number) {
  const hex = color.replace("#", "");
  if (hex.length !== 6) return color;
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16));
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/** TradingView price chart: area of closes, volume underneath, crosshair with values, range selector. */
export default function PriceChart({ prices, height = 340 }: { prices: PricePoint[]; height?: number }) {
  const container = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);
  const area = useRef<ISeriesApi<"Area"> | null>(null);
  const volume = useRef<ISeriesApi<"Histogram"> | null>(null);
  const colors = useChartColors();
  const [range, setRange] = useState<Range>("1Y");

  const visible = useMemo(() => {
    const days = RANGES.find((r) => r.value === range)!.days;
    return Number.isFinite(days) ? prices.slice(-days) : prices;
  }, [prices, range]);
  const change = visible.length > 1 ? visible[visible.length - 1].close / visible[0].close - 1 : null;
  const lineColor = change !== null && change < 0 ? colors.neg : colors.pos;

  useEffect(() => {
    if (!container.current) return;
    const instance = createChart(container.current, {
      autoSize: true,
      layout: { background: { color: "transparent" }, textColor: colors.muted, fontFamily: "Inter Variable, system-ui, sans-serif", fontSize: 11, attributionLogo: false },
      grid: { vertLines: { visible: false }, horzLines: { color: colors.grid } },
      rightPriceScale: { borderVisible: false },
      timeScale: { borderVisible: false, fixLeftEdge: true, fixRightEdge: true, lockVisibleTimeRangeOnResize: true },
      crosshair: { vertLine: { color: colors.muted, labelBackgroundColor: colors.text2 }, horzLine: { color: colors.muted, labelBackgroundColor: colors.text2 } },
      handleScroll: false,
      handleScale: false,
    });
    area.current = instance.addSeries(AreaSeries, { lineWidth: 2, priceLineVisible: false, lastValueVisible: true });
    volume.current = instance.addSeries(HistogramSeries, { priceScaleId: "volume", priceFormat: { type: "volume" }, lastValueVisible: false, priceLineVisible: false });
    instance.priceScale("volume").applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
    instance.priceScale("right").applyOptions({ scaleMargins: { top: 0.08, bottom: 0.22 } });
    chart.current = instance;
    // The container can still be settling when the chart is created (e.g. on phones); refit when it resizes.
    const observer = new ResizeObserver(() => instance.timeScale().fitContent());
    observer.observe(container.current);
    return () => {
      observer.disconnect();
      instance.remove();
    };
  }, [colors]);

  useEffect(() => {
    if (!area.current || !volume.current || !chart.current) return;
    area.current.applyOptions({ lineColor, topColor: withAlpha(lineColor, 0.22), bottomColor: withAlpha(lineColor, 0.0) });
    area.current.setData(visible.map((p) => ({ time: toTime(p.date), value: p.close })));
    volume.current.setData(visible.filter((p) => p.volume).map((p) => ({ time: toTime(p.date), value: Number(p.volume), color: withAlpha(colors.muted, 0.35) })));
    chart.current.timeScale().fitContent();
  }, [visible, lineColor, colors]);

  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <Segmented label="Chart range" value={range} onChange={setRange} options={RANGES.map(({ value, label }) => ({ value, label }))} />
        {change !== null && (
          <span className={`num text-xs font-medium ${change >= 0 ? "text-pos" : "text-neg"}`}>
            {change >= 0 ? "+" : "−"}
            {Math.abs(change * 100).toFixed(1)}% over {range === "MAX" ? "the full history" : range}
          </span>
        )}
      </div>
      <div ref={container} style={{ height }} className="w-full" role="img" aria-label={`Price chart, ${range}`} />
    </div>
  );
}
