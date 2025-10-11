// ui/pages/monitoring/network/dynamics.tsx
import Head from "next/head";
import dynamic from "next/dynamic";
import type { GetStaticProps } from "next";
import { useMemo, useState, useEffect } from "react";
import { getNetworkDynamics } from "@/lib/services/monitoring";

// Plotly (client only)
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

type Grid = { key: string; shape?: number[] | [number, number]; values: number[][] };
type Dynamics = {
  schema_version?: string;
  generated_at?: string;
  window_days?: number;
  grids?: Grid[];
  stats?: { events_total?: number; stations_active?: number };
};

type Props = {
  dyn: Dynamics | null;
  generatedAt: string;
};

export const revalidate = 600;

export const getStaticProps: GetStaticProps<Props> = async () => {
  const dyn = (await getNetworkDynamics<Dynamics>()) ?? null;
  return {
    props: {
      dyn,
      generatedAt: new Date().toISOString(),
    },
    revalidate,
  };
};

function fmt(n?: number | null, d = 0) {
  const v = Number(n);
  if (!Number.isFinite(v)) return "—";
  return new Intl.NumberFormat("fr-FR", { maximumFractionDigits: d }).format(v);
}

export default function NetworkDynamicsPage({ dyn, generatedAt }: Props) {
  const grids = dyn?.grids ?? [];
  const defaultKey =
    grids.find((g) => g.key === "dow_hour_bins")?.key || grids[0]?.key || "";
  const [gridKey, setGridKey] = useState<string>(defaultKey);

  // garde le default si SSR hydratation diffère
  useEffect(() => {
    if (defaultKey) setGridKey(defaultKey);
  }, [defaultKey]);

  const current = useMemo(() => grids.find((g) => g.key === gridKey), [grids, gridKey]);

  const heat = useMemo(() => {
    if (!current) return null;
    const values = current.values;
    if (!Array.isArray(values) || !values.length) return null;

    // Heures et jours “par défaut”
    const yLabels = ["Dim", "Lun", "Mar", "Mer", "Jeu", "Ven", "Sam"];
    const xLabels = Array.from({ length: 24 }, (_, i) => `${String(i).padStart(2, "0")}h`);

    // Si la grille n’est pas 7x24, on construit des axes génériques
    const rows = values.length;
    const cols = Array.isArray(values[0]) ? values[0].length : 0;
    const y = rows === 7 ? yLabels : Array.from({ length: rows }, (_, i) => `r${i + 1}`);
    const x = cols === 24 ? xLabels : Array.from({ length: cols }, (_, i) => `c${i + 1}`);

    return { z: values, x, y };
  }, [current]);

  const totals = useMemo(() => {
    if (!current?.values) return { sum: 0, mean: 0, max: 0 };
    let sum = 0;
    let count = 0;
    let max = -Infinity;
    for (const row of current.values) {
      for (const v of row) {
        const n = Number(v);
        if (Number.isFinite(n)) {
          sum += n;
          count += 1;
          if (n > max) max = n;
        }
      }
    }
    return { sum, mean: count ? sum / count : 0, max: Number.isFinite(max) ? max : 0 };
  }, [current]);

  return (
    <>
      <Head>
        <title>Monitoring — Network / Dynamics</title>
        <meta
          name="description"
          content="Analyse détaillée des dynamiques réseau (grilles temps × catégories)."
        />
      </Head>

      <main style={{ padding: "2rem", maxWidth: 1200, margin: "0 auto" }}>
        <header style={{ display: "flex", justifyContent: "space-between", gap: 16 }}>
          <div>
            <h1 style={{ margin: 0 }}>Network · Dynamics</h1>
            <div className="small" style={{ opacity: 0.7 }}>
              Fenêtre: {dyn?.window_days ?? "—"} jours · Généré:{" "}
              {new Date(generatedAt).toLocaleString("fr-FR")}
            </div>
          </div>
          <div>
            <label htmlFor="grid" className="small" style={{ marginRight: 8, opacity: 0.7 }}>
              Grille :
            </label>
            <select
              id="grid"
              value={gridKey}
              onChange={(e) => setGridKey(e.target.value)}
              style={{ padding: "6px 8px", borderRadius: 8, border: "1px solid #e5e7eb" }}
            >
              {grids.map((g) => (
                <option key={g.key} value={g.key}>
                  {g.key} {g.shape ? `(${g.shape.join("×")})` : ""}
                </option>
              ))}
            </select>
          </div>
        </header>

        {/* KPIs de la grille */}
        <section style={{ marginTop: 16 }}>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
              gap: 12,
            }}
          >
            <Kpi label="Évènements (total)" value={fmt(dyn?.stats?.events_total)} />
            <Kpi label="Stations actives" value={fmt(dyn?.stats?.stations_active)} />
            <Kpi label="Somme grille" value={fmt(totals.sum)} />
            <Kpi label="Max cellule" value={fmt(totals.max)} />
          </div>
        </section>

        {/* Heatmap */}
        <section style={{ marginTop: 24 }}>
          <h2 style={{ margin: "12px 0" }}>Heatmap — {gridKey || "grille"}</h2>

          {!heat ? (
            <div className="small" style={{ opacity: 0.7 }}>
              Grille vide ou non reconnue pour <code>{gridKey}</code>.
            </div>
          ) : (
            <Plot
              data={[
                {
                  z: heat.z,
                  x: heat.x,
                  y: heat.y,
                  type: "heatmap",
                  hoverongaps: false,
                  colorbar: { title: { text: "Volume" } },
                } as any,
              ]}
              layout={{
                autosize: true,
                height: 520,
                margin: { l: 60, r: 20, t: 30, b: 40 },
                xaxis: { title: { text: "Colonne" } },
                yaxis: { title: { text: "Ligne" } },
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: "100%" }}
            />
          )}
        </section>

        {/* JSON brut (optionnel) */}
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
              {JSON.stringify(dyn ?? {}, null, 2)}
            </pre>
          </details>
        </section>
      </main>
    </>
  );
}

function Kpi({
  label,
  value,
}: {
  label: string;
  value: string | number | null | undefined;
}) {
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
