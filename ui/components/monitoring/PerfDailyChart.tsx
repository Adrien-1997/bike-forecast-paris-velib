// ui/components/monitoring/PerfDailyChart.tsx
import dynamic from "next/dynamic";
import React from "react";
import type { Layout, PlotData } from "plotly.js";
import type { PerfDailyResponse } from "@/lib/types";

// Chargement côté client uniquement
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

type Props = {
  data: PerfDailyResponse; // réponse typée de l’API
  metric?: "rmse_model" | "mae_model" | "mae_baseline";
  title?: string;
};

/**
 * PerfDailyChart
 * - Trace une série journalière à partir de PerfDailyResponse.metrics
 * - Ignore les valeurs null, tout en conservant l’axe temporel
 */
export default function PerfDailyChart({
  data,
  metric = "mae_baseline",
  title,
}: Props) {
  const x = data.metrics.map((m) => m.date);
  const y = data.metrics.map((m) => {
    const v = (m as any)[metric] as number | null | undefined;
    return v == null ? null : v;
  });

  const layout: Partial<Layout> = {
    title: { text: title ?? `Daily ${metric} — h=${data.horizon_min} min` },
    margin: { l: 50, r: 20, t: 48, b: 40 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    xaxis: { title: { text: "Date" }, type: "category", tickangle: -45, automargin: true },
    yaxis: { title: { text: metric }, rangemode: "tozero" },
    hovermode: "x unified",
    showlegend: false,
  };

  const trace: Partial<PlotData> = {
    type: "scatter",
    mode: "lines+markers",
    x,
    y,
    name: metric,
    connectgaps: false, // ne relie pas les trous si null
    marker: { size: 6 },
    line: { shape: "spline", smoothing: 0.35 },
  };

  return (
    <div className="rounded-2xl ring-1 ring-slate-700 bg-slate-900/40 p-3">
      <Plot
        data={[trace]}
        layout={layout}
        style={{ width: "100%", height: 360 }}
        useResizeHandler
        config={{ displayModeBar: false, responsive: true }}
      />
      <div className="mt-2 text-xs text-slate-400">
        Schema v{data.schema_version} — generated {data.generated_at}
      </div>
    </div>
  );
}
