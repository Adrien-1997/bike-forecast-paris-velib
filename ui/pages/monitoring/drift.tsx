// ui/pages/monitoring/drift.tsx
import Head from "next/head";
import type { GetStaticProps } from "next";
import dynamic from "next/dynamic";
import { useMemo, useState, useEffect } from "react";
import { getDriftSummary } from "@/lib/services/monitoring";
import type { DriftSummary } from "@/lib/types";

// Plotly côté client uniquement
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

/* ─────────────── Types locaux ─────────────── */

type DriftRow = {
  name: string;                 // nom de la feature
  psi?: number | null;          // Population Stability Index (0..+)
  ks?: number | null;           // Kolmogorov–Smirnov (0..1)
  p_value?: number | null;      // éventuelle p-value
  drifted?: boolean | null;     // drapeau source si présent
  alert?: boolean | null;       // calculé côté UI (seuils)
};

type Props = {
  rows: DriftRow[];
  reference_window: string | null;
  current_window: string | null;
  generatedAt: string;
};

export const revalidate = 600;

/* ─────────────── SSR ─────────────── */

export const getStaticProps: GetStaticProps<Props> = async () => {
  const data = await getDriftSummary<DriftSummary>();

  // Map features[] -> rows[], et calcule un drapeau d’alerte simple
  const rows: DriftRow[] = (data?.features ?? []).map((f) => {
    const name = (f as any)?.feature ?? "(unnamed)"; // évite undefined (serialize)
    const psi = Number.isFinite(Number(f.psi)) ? Number(f.psi) : null;
    const ks = Number.isFinite(Number(f.ks)) ? Number(f.ks) : null;

    // Seuils simples (ajuste au besoin)
    const isAlert =
      (psi != null && psi >= 0.2) ||
      (ks != null && ks >= 0.1) ||
      (typeof f.drifted === "boolean" ? f.drifted : false);

    return {
      name,
      psi,
      ks,
      p_value: f.p_value ?? null,
      drifted: f.drifted ?? null,
      alert: isAlert,
    };
  });

  return {
    props: {
      rows,
      reference_window: data?.reference_window ?? null,
      current_window: data?.current_window ?? null,
      generatedAt: new Date().toISOString(),
    },
    revalidate,
  };
};

/* ─────────────── Helpers UI ─────────────── */

function useMounted() {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  return mounted;
}

function fmt(n?: number | null, d = 3) {
  const v = Number(n);
  if (!Number.isFinite(v)) return "—";
  return new Intl.NumberFormat("fr-FR", { maximumFractionDigits: d }).format(v);
}

function median(arr: Array<number | null | undefined>): number | null {
  const xs = arr
    .map((x) => (Number.isFinite(Number(x)) ? Number(x) : null))
    .filter((x): x is number => x !== null)
    .sort((a, b) => a - b);
  if (!xs.length) return null;
  const m = Math.floor(xs.length / 2);
  return xs.length % 2 ? xs[m] : (xs[m - 1] + xs[m]) / 2;
}

function fmtParis(iso?: string | null) {
  if (!iso) return "—";
  const d = new Date(iso);
  return new Intl.DateTimeFormat("fr-FR", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "Europe/Paris",
  }).format(d);
}

/* ─────────────── Composant principal ─────────────── */

export default function DriftPage({
  rows,
  reference_window,
  current_window,
  generatedAt,
}: Props) {
  const mounted = useMounted();

  const [q, setQ] = useState("");
  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    if (!needle) return rows;
    return rows.filter((r) => r.name.toLowerCase().includes(needle));
  }, [rows, q]);

  const alerts = filtered.filter((r) => !!r.alert);

  const psiSeries = useMemo(() => {
    return {
      x: filtered.map((r) => r.name),
      y: filtered.map((r) => (Number.isFinite(Number(r.psi)) ? Number(r.psi) : 0)),
    };
  }, [filtered]);

  const ksSeries = useMemo(() => {
    return {
      x: filtered.map((r) => r.name),
      y: filtered.map((r) => (Number.isFinite(Number(r.ks)) ? Number(r.ks) : 0)),
    };
  }, [filtered]);

  return (
    <>
      <Head>
        <title>Monitoring — Drift</title>
        <meta
          name="description"
          content="Drift des features : PSI/KS par feature, synthèse et JSON brut."
        />
      </Head>

      <main style={{ padding: "2rem", maxWidth: 1200, margin: "0 auto" }}>
        <header style={{ display: "flex", justifyContent: "space-between", gap: 16 }}>
          <div>
            <h1 style={{ margin: 0 }}>Drift</h1>
            <div className="small" style={{ opacity: 0.7 }}>
              Ref: {reference_window ?? "—"} · Cur: {current_window ?? "—"} · Généré:{" "}
              {fmtParis(generatedAt)}
            </div>
          </div>
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Filtrer une feature…"
            style={{
              padding: "6px 10px",
              borderRadius: 8,
              border: "1px solid #e5e7eb",
              minWidth: 260,
            }}
          />
        </header>

        {/* Alerts & KPIs */}
        <section style={{ marginTop: 16 }}>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
              gap: 12,
            }}
          >
            <Kpi label="Features (total)" value={rows.length} />
            <Kpi label="En alerte" value={alerts.length} />
            <Kpi
              label="PSI médian (filtré)"
              value={median(filtered.map((r) => (Number.isFinite(Number(r.psi)) ? Number(r.psi) : null))) ?? "—"}
            />
          </div>
        </section>

        {/* PSI bar chart */}
        <section style={{ marginTop: 24 }}>
          <h2 style={{ margin: "12px 0" }}>PSI par feature</h2>
          {!mounted ? (
            <div
              className="small"
              style={{
                opacity: 0.7,
                border: "1px solid #e5e7eb",
                borderRadius: 8,
                padding: 12,
              }}
            >
              Chargement du graphique…
            </div>
          ) : (
            <Plot
              data={[
                {
                  x: psiSeries.x,
                  y: psiSeries.y,
                  type: "bar",
                  name: "PSI",
                } as any,
              ]}
              layout={{
                autosize: true,
                height: 360,
                margin: { l: 50, r: 10, t: 30, b: 80 },
                xaxis: { tickangle: -45, automargin: true, title: { text: "Feature" } },
                yaxis: { title: { text: "PSI" } },
                shapes: [
                  // seuil informatif PSI = 0.2
                  {
                    type: "line",
                    x0: -0.5,
                    x1: psiSeries.x.length - 0.5,
                    y0: 0.2,
                    y1: 0.2,
                    line: { dash: "dot", width: 1 },
                  },
                ],
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: "100%" }}
            />
          )}
        </section>

        {/* KS bar chart */}
        <section style={{ marginTop: 28 }}>
          <h2 style={{ margin: "12px 0" }}>KS par feature</h2>
          {!mounted ? (
            <div
              className="small"
              style={{
                opacity: 0.7,
                border: "1px solid #e5e7eb",
                borderRadius: 8,
                padding: 12,
              }}
            >
              Chargement du graphique…
            </div>
          ) : (
            <Plot
              data={[
                {
                  x: ksSeries.x,
                  y: ksSeries.y,
                  type: "bar",
                  name: "KS",
                } as any,
              ]}
              layout={{
                autosize: true,
                height: 360,
                margin: { l: 50, r: 10, t: 30, b: 80 },
                xaxis: { tickangle: -45, automargin: true, title: { text: "Feature" } },
                yaxis: { title: { text: "KS" } },
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: "100%" }}
            />
          )}
        </section>

        {/* Table */}
        <section style={{ marginTop: 28 }}>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  <th align="left">Feature</th>
                  <th align="right">PSI</th>
                  <th align="right">KS</th>
                  <th align="center">Alerte</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((r) => (
                  <tr key={r.name}>
                    <td>{r.name}</td>
                    <td align="right">{fmt(r.psi)}</td>
                    <td align="right">{fmt(r.ks)}</td>
                    <td align="center">{r.alert ? "🚨" : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* JSON brut */}
        <section style={{ marginTop: 28 }}>
          <details>
            <summary style={{ cursor: "pointer" }}>Voir le JSON brut</summary>
            <pre
              style={{
                background: "#0b1220",
                color: "#d1d5db",
                padding: "1rem",
                borderRadius: 8,
                overflowX: "auto",
                maxHeight: "50vh",
                marginTop: 8,
              }}
            >
              {JSON.stringify(
                { reference_window, current_window, features: rows },
                null,
                2
              )}
            </pre>
          </details>
        </section>
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
      <div className="small" style={{ opacity: 0.7 }}>
        {label}
      </div>
      <div style={{ fontSize: 22, fontWeight: 700 }}>{value ?? "—"}</div>
    </div>
  );
}
