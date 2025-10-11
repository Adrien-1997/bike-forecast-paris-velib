// ui/pages/monitoring/perf.tsx
import Head from "next/head";
import { useMemo, useState, useEffect } from "react";
import type { GetStaticProps } from "next";
import { getPerfDaily } from "@/lib/services/monitoring";
import type { PerfDailyResponse } from "@/lib/types";

// Plotly (client only)
import dynamic from "next/dynamic";

export const Plot = dynamic(
  () => import("react-plotly.js").then((m) => m.default),
  {
    ssr: false,
    loading: () => (
      <div style={{ height: 280, display: "grid", placeItems: "center", opacity: 0.7 }}>
        Chargement du graphique…
      </div>
    ),
  }
);

type Props = {
  daily15: PerfDailyResponse | null;
  daily60: PerfDailyResponse | null; // peut être null si h=60 absent côté GCS
  generatedAt: string; // horodatage de build
};

export const revalidate = 600; // ISR 10 min

export const getStaticProps: GetStaticProps<Props> = async () => {
  // charge 15 en priorité (ne jette pas)
  const daily15 = (await getPerfDaily(15)) ?? null;

  // tente 60 mais NE JAMAIS bloquer le SSR
  let daily60: PerfDailyResponse | null = null;
  try {
    daily60 = (await getPerfDaily(60)) ?? null;
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

function useQueryH(): 15 | 60 {
  const [h, setH] = useState<15 | 60>(15);
  useEffect(() => {
    if (typeof window === "undefined") return;
    const url = new URL(window.location.href);
    const hv = Number(url.searchParams.get("h"));
    if (hv === 60) setH(60);
  }, []);
  return h;
}

function fmtPct(x?: number | null, digits = 1) {
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

function fmt(n?: number | null, digits = 2) {
  const v = Number(n);
  if (!Number.isFinite(v)) return "—";
  return v.toFixed(digits);
}

/** Kpi inline minimal pour éviter un import de composant séparé */
function Kpi({
  label,
  value,
  fmt: f,
}: {
  label: string;
  value: number | null | undefined;
  fmt?: (v: number | null | undefined) => string;
}) {
  const text = f ? f(value) : fmt(value);
  return (
    <div
      style={{
        border: "1px solid #374151",
        background: "rgba(15, 23, 42, 0.5)",
        borderRadius: 12,
        padding: "10px 12px",
      }}
    >
      <div style={{ fontSize: 12, color: "#9CA3AF", textTransform: "uppercase" }}>{label}</div>
      <div style={{ fontSize: 22, color: "#E5E7EB", fontWeight: 600, marginTop: 2 }}>{text}</div>
    </div>
  );
}

export default function PerfPage({ daily15, daily60, generatedAt }: Props) {
  const h = useQueryH();
  const [horizon, setHorizon] = useState<15 | 60>(15);

  // Synchronise l’état local avec le param ?h
  useEffect(() => {
    setHorizon(h);
  }, [h]);

  const daily = horizon === 15 ? daily15 : daily60;

  // Séries pour le line chart :
  // - RMSE = rmse_model (souvent null tant que le modèle n’écrit pas)
  // - MAE (baseline) = mae_baseline (toujours présent d’après le probe)
  // - Coverage = coverage_pred_pct
  const lineSeries = useMemo(() => {
    const dates = (daily?.metrics ?? []).map((r) => r.date);
    const rmse_model = (daily?.metrics ?? []).map((r) => r.rmse_model);
    const mae_model = (daily?.metrics ?? []).map((r) => r.mae_model);
    const mae_baseline = (daily?.metrics ?? []).map((r) => r.mae_baseline);
    const coverage = (daily?.metrics ?? []).map((r) => r.coverage_pred_pct);
    const n = (daily?.metrics ?? []).map((r) => r.n);

    return { dates, rmse_model, mae_model, mae_baseline, coverage, n };
  }, [daily]);

  // Navigate horizons en mettant à jour la query string
  const setH = (h: 15 | 60) => {
    setHorizon(h);
    if (typeof window !== "undefined") {
      const url = new URL(window.location.href);
      url.searchParams.set("h", String(h));
      window.history.replaceState({}, "", url.toString());
    }
  };

  return (
    <>
      <Head>
        <title>Monitoring — Model Performance</title>
        <meta
          name="description"
          content="Suivi des performances du modèle de prévision Vélib’ (RMSE/MAE par jour, couverture)."
        />
      </Head>

      <main style={{ padding: "2rem", maxWidth: 1200, margin: "0 auto" }}>
        <header style={{ display: "flex", justifyContent: "space-between", gap: 16 }}>
          <div>
            <h1 style={{ margin: 0 }}>Model Performance</h1>
            <div className="small" style={{ opacity: 0.7 }}>
              Données préchargées (ISR). Généré : {new Date(generatedAt).toLocaleString("fr-FR")}
            </div>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span className="small" style={{ opacity: 0.7 }}>
              Horizon :
            </span>
            <button
              onClick={() => setH(15)}
              aria-pressed={horizon === 15}
              style={{
                padding: "6px 10px",
                borderRadius: 8,
                border: "1px solid #d1d5db",
                background: horizon === 15 ? "#111827" : "white",
                color: horizon === 15 ? "white" : "#111827",
              }}
            >
              15 min
            </button>
            <button
              onClick={() => setH(60)}
              aria-pressed={horizon === 60}
              style={{
                padding: "6px 10px",
                borderRadius: 8,
                border: "1px solid #d1d5db",
                background: horizon === 60 ? "#111827" : "white",
                color: horizon === 60 ? "white" : "#111827",
                opacity: daily60 ? 1 : 0.5,
                cursor: daily60 ? "pointer" : "not-allowed",
              }}
              disabled={!daily60} // h=60 indisponible => disabled
              title={!daily60 ? "h=60 indisponible (JSON manquant)" : ""}
            >
              60 min
            </button>
          </div>
        </header>

        {/* State de disponibilité */}
        {!daily && (
          <div
            style={{
              marginTop: 16,
              border: "1px solid #B45309",
              background: "rgba(180,83,9,0.1)",
              color: "#F59E0B",
              borderRadius: 10,
              padding: "10px 12px",
            }}
          >
            Les métriques pour h={horizon} ne sont pas encore disponibles.
          </div>
        )}

        {/* Line chart: RMSE/MAE par jour */}
        {daily && (
          <section style={{ marginTop: 24 }}>
            <h2 style={{ margin: "12px 0" }}>
              Daily — RMSE (modèle) &amp; MAE (baseline) — h={daily.horizon_min} min
            </h2>

            {typeof window === "undefined" ? (
              <div className="small" style={{ opacity: 0.7 }}>
                Rendu client en attente…
              </div>
            ) : (
              <Plot
                data={[
                  {
                    x: lineSeries.dates,
                    y: lineSeries.rmse_model,
                    type: "scatter",
                    mode: "lines+markers",
                    name: "RMSE (modèle)",
                    connectgaps: false,
                  },
                  {
                    x: lineSeries.dates,
                    y: lineSeries.mae_baseline,
                    type: "scatter",
                    mode: "lines+markers",
                    name: "MAE (baseline)",
                    connectgaps: false,
                  },
                ]}
                layout={{
                autosize: true,
                height: 380,
                margin: { l: 50, r: 10, t: 30, b: 40 },
                xaxis: { title: { text: "Date" } },
                yaxis: { title: { text: "Erreur" } },
                legend: { orientation: "h" },
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: "100%" }}
              />
            )}

            {/* KPIs */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
                gap: 12,
                marginTop: 12,
              }}
            >
              <Kpi label="Coverage (moy.)" value={mean(lineSeries.coverage)} fmt={(v) => fmtPct(v)} />
              <Kpi label="RMSE (méd.)" value={median(lineSeries.rmse_model)} />
              <Kpi label="MAE model (méd.)" value={median(lineSeries.mae_model)} />
              <Kpi label="MAE baseline (méd.)" value={median(lineSeries.mae_baseline)} />
            </div>

            {/* Table daily */}
            <div style={{ overflowX: "auto", marginTop: 16 }}>
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
                      <td align="right">{fmtPct(r.coverage_pred_pct)}</td>
                      <td align="right">{r.n}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div style={{ marginTop: 8, fontSize: 12, opacity: 0.7 }}>
              Schema v{daily.schema_version} — generated {daily.generated_at}
            </div>
          </section>
        )}
      </main>
    </>
  );
}

/* ───────────────── helpers ───────────────── */

function median(arr?: Array<number | null | undefined>): number | null {
  if (!arr || !arr.length) return null;
  const xs = arr
    .map((x) => (Number.isFinite(Number(x)) ? Number(x) : null))
    .filter((x): x is number => x !== null)
    .sort((a, b) => a - b);
  if (!xs.length) return null;
  const mid = Math.floor(xs.length / 2);
  return xs.length % 2 ? xs[mid] : (xs[mid - 1] + xs[mid]) / 2;
}

function mean(arr?: Array<number | null | undefined>): number | null {
  if (!arr || !arr.length) return null;
  let n = 0,
    s = 0;
  for (const x of arr) {
    const v = Number(x);
    if (Number.isFinite(v)) {
      s += v;
      n++;
    }
  }
  return n ? s / n : null;
}
