// ui/lib/plotlyTheme.ts
import type * as Plotly from "plotly.js";

/* ──────────────────────────── CONFIG ──────────────────────────── */
export const chartConfig: Partial<Plotly.Config> = {
  displayModeBar: false,
  responsive: true,
  scrollZoom: false,
};

/* ──────────────────────────── LAYOUT ──────────────────────────── */
export function chartLayout(overrides: Partial<Plotly.Layout> = {}): Partial<Plotly.Layout> {
  const base: Partial<Plotly.Layout> = {
    autosize: true,

    // marges plus généreuses (haut/bas) pour éviter le clipping
    margin: { l: 52, r: 12, t: 34, b: 48 },

    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    hovermode: "x unified",

    font: {
      family: "var(--mn-font)",
      size: 12,
      color: "var(--text)",
    },

    // légende placée au-dessus du graphique
    legend: {
      orientation: "h",
      x: 0,
      y: 1.12,
      yanchor: "bottom",
      font: { size: 12, color: "var(--text-dim)" },
      tracegroupgap: 4,
    },

    // Axe X — espacement du titre et ticks sortants
    xaxis: {
      title: {
        font: { size: 12, color: "var(--text-dim)" },
        standoff: 12,
      },
      tickfont: { color: "var(--text-dim)" },
      gridcolor: "rgba(255,255,255,.08)",
      zeroline: false,
      automargin: true,
      ticks: "outside",
      ticklen: 6,
    },

    // Axe Y — idem
    yaxis: {
      title: {
        font: { size: 12, color: "var(--text-dim)" },
        standoff: 8,
      },
      tickfont: { color: "var(--text-dim)" },
      gridcolor: "rgba(255,255,255,.08)",
      zeroline: false,
      rangemode: "tozero",
      automargin: true,
      ticks: "outside",
      ticklen: 6,
    },
  };

  return { ...base, ...overrides };
}
