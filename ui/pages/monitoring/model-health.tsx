// ui/pages/monitoring/model-health.tsx
import Head from "next/head";
import dynamic from "next/dynamic";
import type { GetStaticProps } from "next";
import { useMemo, useState, useEffect } from "react";
import { getPerfDaily } from "@/lib/services/monitoring";
import type { PerfDailyResponse } from "@/lib/types";

// Plotly côté client uniquement (évite le SSR/hydration mismatch)
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

type Props = {
  daily15: PerfDailyResponse | null;
  daily60: PerfDailyResponse | null; // peut être null si JSON absent
  generatedAt: string;
};

export const revalidate = 600;

/* ─────────────── SSR ─────────────── */
export const getStaticProps: GetStaticProps<Props> = async () => {
  const daily15 = (await getPerfDaily(15)) ?? null;

  let daily60: PerfDailyResponse | null = null;
  try {
    daily60 = await getPerfDaily(60);
  } catch {
    daily60 = null;
  }

  return {
    props: {
      daily15,
      daily60,
      generatedAt: new Date().toISOString(),
    },
    revalidate,
  };
};

/* ─────────────── utils ─────────────── */

function useMounted() {
  const [m, setM] = useState(false);
  useEffect(() => setM(true), []);
  return m;
}

function fmt(n?: number | null, d = 2) {
  const v = Number(n);
  if (!Number.isFinite(v)) return "—";
  return new Intl.NumberFormat("fr-FR", { maximumFractionDigits: d }).format(v);
}

function fmtPct01(x?: number | null, d = 1) {
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  const p = v <= 1 ? v * 100 : v;
  return `${p.toFixed(d)}%`;
}

function mean(arr?: Array<number | null | undefined>) {
  const xs = (arr ?? [])
    .map((v) => (Number.isFinite(Number(v)) ? Number(v) : null))
    .filter((v): v is number => v !== null);
  if (!xs.length) return null;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}
function median(arr?: Array<number | null | undefined>) {
  const xs = (arr ?? [])
    .map((v) => (Number.isFinite(Number(v)) ? Number(v) : null))
    .filter((v): v is number => v !== null)
    .sort((a, b) => a - b);
  if (!xs.length) return null;
  const m = Math.floor(xs.length / 2);
  return xs.length % 2 ? xs[m] : (xs[m - 1] + xs[m]) / 2;
}

/* ─────────────── Page ─────────────── */

export default function ModelHealthPage({ daily15, daily60, generatedAt }: Props) {
  const mounted = useMounted();

  // toggle d’horizon (persiste la query ?h=…)
  const [h, setH] = useState<15 | 60>(15);
  useEffect(() => {
    if (typeof window === "undefined") return;
    const url = new URL(window.location.href);
    const hv = Number(url.searchParams.get("h"));
    if (hv === 60 && daily60) setH(60);
  }, [daily60]);

  const setHorizon = (nh: 15 | 60) => {
    setH(nh);
    if (typeof window !== "undefined") {
      const url = new URL(window.location.href);
      url.searchParams.set("h", String(nh));
      window.history.replaceState({}, "", url.toString());
    }
  };

  const daily = h === 15 ? daily15 : daily60;

  // Séries + KPIs calculées sur metrics[]
  const metrics = daily?.metrics ?? [];
  const last = metrics.length ? metrics[metrics.length - 1] : null;
  const last7 = metrics.slice(-7);

  const kpis = useMemo(() => {
    const rmse7 = median(last7.map((r) => r.rmse_model));
    const mae7 = median(last7.map((r) => r.mae_model));
    const cov7 = mean(last7.map((r) => r.coverage_pred_pct));
    return {
      last_rmse: last?.rmse_model ?? null,
      last_mae: last?.mae_model ?? null,
      last_cov: last?.coverage_pred_pct ?? null,
      rmse7,
      mae7,
      cov7,
      n_last: last?.n ?? null,
      last_date: last?.date ?? null,
    };
  }, [last, last7]);

  const line = useMemo(() => {
    const dates = metrics.map((r) => r.date);
    return {
      x: dates,
      rmse: metrics.map((r) => r.rmse_model),
      mae: metrics.map((r) => r.mae_model),
      cov: metrics.map((r) => r.coverage_pred_pct),
    };
  }, [metrics]);

  return (
    <>
      <Head>
        <title>Monitoring — Model Health</title>
        <meta
          name="description"
          content="Santé du modèle : derniers indicateurs, tendance 7 jours et couverture."
        />
      </Head>

      <main style={{ padding: "2rem", maxWidth: 1200, margin: "0 auto" }}>
        <header style={{ display: "flex", justifyContent: "space-between", gap: 16 }}>
          <div>
            <h1 style={{ margin: 0 }}>Model Health</h1>
            <div className="small" style={{ opacity: 0.7 }}>
              Page ISR : {new Date(generatedAt).toLocaleString("fr-FR")}
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span className="small" style={{ opacity: 0.7 }}>Horizon:</span>
            <button
              onClick={() => setHorizon(15)}
              aria-pressed={h === 15}
              style={{
                padding: "6px 10px",
                borderRadius: 8,
                border: "1px solid #d1d5db",
                background: h === 15 ? "#111827" : "white",
                color: h === 15 ? "white" : "#111827",
              }}
            >
              15 min
            </button>
            <button
              onClick={() => setHorizon(60)}
              aria-pressed={h === 60}
              disabled={!daily60}
              title={!daily60 ? "h=60 indisponible (JSON manquant)" : ""}
              style={{
                padding: "6px 10px",
                borderRadius: 8,
                border: "1px solid #d1d5db",
                background: h === 60 ? "#111827" : "white",
                color: h === 60 ? "white" : "#111827",
                opacity: !daily60 ? 0.6 : 1,
                cursor: !daily60 ? "not-allowed" : "pointer",
              }}
            >
              60 min
            </button>
          </div>
        </header>

        {/* KPIs récents */}
        <section style={{ marginTop: 16 }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(5, minmax(0, 1fr))", gap: 12 }}>
            <Kpi label="Dernier RMSE" value={fmt(kpis.last_rmse)} />
            <Kpi label="Dernier MAE" value={fmt(kpis.last_mae)} />
            <Kpi label="Dernière couverture" value={fmtPct01(kpis.last_cov)} />
            <Kpi label="RMSE méd. 7j" value={fmt(kpis.rmse7)} />
            <Kpi label="MAE méd. 7j" value={fmt(kpis.mae7)} />
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 12, marginTop: 12 }}>
            <Kpi label="Couverture moy. 7j" value={fmtPct01(kpis.cov7)} />
            <Kpi label="Obs. dernier jour (N)" value={fmt(kpis.n_last, 0)} />
            <Kpi label="Dernière date" value={kpis.last_date ?? "—"} />
          </div>
        </section>

        {/* Courbe couverture + erreurs */}
        <section style={{ marginTop: 24 }}>
          <h2 style={{ margin: "12px 0" }}>
            Tendances — h={daily?.horizon_min ?? "—"} min
          </h2>

          {!daily ? (
            <div
              className="small"
              style={{
                opacity: 0.7,
                border: "1px solid #eab308",
                background: "rgba(234,179,8,.08)",
                padding: 12,
                borderRadius: 8,
              }}
            >
              Les métriques pour h={h} ne sont pas encore disponibles.
            </div>
          ) : !mounted ? (
            <div className="small" style={{ opacity: 0.7 }}>
              Rendu client en attente…
            </div>
          ) : (
            <Plot
              data={[
                { x: line.x, y: line.rmse, type: "scatter", mode: "lines+markers", name: "RMSE (modèle)" } as any,
                { x: line.x, y: line.mae, type: "scatter", mode: "lines+markers", name: "MAE (modèle)" } as any,
                {
                  x: line.x,
                  y: line.cov.map((v) => (Number(v) <= 1 ? Number(v) * 100 : Number(v))),
                  type: "scatter",
                  mode: "lines",
                  name: "Coverage (%)",
                  yaxis: "y2",
                } as any,
              ]}
              layout={{
                autosize: true,
                height: 420,
                margin: { l: 50, r: 50, t: 30, b: 40 },
                xaxis: { title: { text: "Date" } },
                yaxis: { title: { text: "Erreur" } },
                yaxis2: { title: { text: "Coverage (%)" }, overlaying: "y", side: "right" },
                legend: { orientation: "h" },
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
              } as any}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: "100%" }}
            />
          )}
        </section>

        {/* Table brute (fallback / debug) */}
        {daily && (
          <section style={{ marginTop: 24 }}>
            <details>
              <summary style={{ cursor: "pointer" }}>Voir les lignes brutes</summary>
              <div style={{ overflowX: "auto", marginTop: 8 }}>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead>
                    <tr>
                      <th align="left">Date</th>
                      <th align="right">RMSE (modèle)</th>
                      <th align="right">MAE (modèle)</th>
                      <th align="right">MAE (baseline)</th>
                      <th align="right">Coverage</th>
                      <th align="right">N</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(daily.metrics ?? []).map((r) => (
                      <tr key={`${r.date}-${daily.horizon_min}`}>
                        <td>{r.date}</td>
                        <td align="right">{fmt(r.rmse_model)}</td>
                        <td align="right">{fmt(r.mae_model)}</td>
                        <td align="right">{fmt(r.mae_baseline)}</td>
                        <td align="right">{fmtPct01(r.coverage_pred_pct)}</td>
                        <td align="right">{fmt(r.n, 0)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          </section>
        )}
      </main>
    </>
  );
}

/* ─────────────── UI ─────────────── */

function Kpi({ label, value }: { label: string; value: any }) {
  return (
    <div
      style={{
        border: "1px solid #e5e7eb",
        borderRadius: 10,
        padding: "12px 14px",
        background: "white",
      }}
    >
      <div className="small" style={{ opacity: 0.7 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 700 }}>{value ?? "—"}</div>
    </div>
  );
}
