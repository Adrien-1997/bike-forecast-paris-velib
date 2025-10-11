// ui/pages/monitoring/network/overview.tsx
import Head from "next/head";
import dynamic from "next/dynamic";
import type { GetStaticProps } from "next";
import { useMemo } from "react";
import { getNetworkDynamics } from "@/lib/services/monitoring";

// Plotly côté client uniquement
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

type Dynamics = {
  schema_version?: string;
  generated_at?: string;
  window_days?: number;
  grids?: { key: string; shape: [number, number] | number[]; values: number[][] }[];
  stats?: { events_total?: number; stations_active?: number };
};

type Props = {
  dyn: Dynamics | null;
  generatedAt: string;
};

export const revalidate = 600; // ISR 10 min

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

// Helpers
function fmt(n?: number | null, d = 0) {
  const v = Number(n);
  if (!Number.isFinite(v)) return "—";
  return new Intl.NumberFormat("fr-FR", { maximumFractionDigits: d }).format(v);
}

export default function NetworkOverviewPage({ dyn, generatedAt }: Props) {
  const windowDays = dyn?.window_days ?? null;
  const stats = dyn?.stats ?? {};
  const heat = useMemo(() => {
    const grid = dyn?.grids?.find((g) => g.key === "dow_hour_bins") ?? dyn?.grids?.[0];
    if (!grid || !Array.isArray(grid.values)) return null;

    const values = grid.values;
    const y = ["Dim", "Lun", "Mar", "Mer", "Jeu", "Ven", "Sam"];
    const x = Array.from({ length: 24 }, (_, i) => `${i.toString().padStart(2, "0")}h`);

    return { z: values, x, y };
  }, [dyn]);

  return (
    <>
      <Head>
        <title>Monitoring — Network Overview</title>
        <meta
          name="description"
          content="Vue d’ensemble du réseau : métriques agrégées, heatmap dow × hour."
        />
      </Head>

      <main style={{ padding: "2rem", maxWidth: 1200, margin: "0 auto" }}>
        <header style={{ display: "flex", justifyContent: "space-between", gap: 16, alignItems: "baseline" }}>
          <div>
            <h1 style={{ margin: 0 }}>Network · Overview</h1>
            <div className="small" style={{ opacity: 0.7 }}>
              Fenêtre: {windowDays ?? "—"} jours · Généré: {new Date(generatedAt).toLocaleString("fr-FR")}
            </div>
          </div>
        </header>

        {/* KPIs */}
        <section style={{ marginTop: 20 }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 12 }}>
            <Kpi label="Évènements (total)" value={fmt(stats.events_total)} />
            <Kpi label="Stations actives" value={fmt(stats.stations_active)} />
            <Kpi label="Window (jours)" value={fmt(windowDays)} />
          </div>
        </section>

        {/* Heatmap dow × hour */}
        <section style={{ marginTop: 28 }}>
          <h2 style={{ margin: "12px 0" }}>Dynamics — Heatmap (dow × hour)</h2>

          {!heat ? (
            <div className="small" style={{ opacity: 0.7 }}>
              Aucune grille exploitable dans la réponse (grids). Vérifie le champ <code>grids</code> du JSON renvoyé par <code>/monitoring/network/dynamics</code>.
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
                height: 480,
                margin: { l: 60, r: 20, t: 30, b: 40 },
                xaxis: { title: { text: "Heure" } },
                yaxis: { title: { text: "Jour (0=Dim, 6=Sam)" } },
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: "100%" }}
            />
          )}
        </section>

        {/* JSON debug */}
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

/* ──────────────── UI ──────────────── */

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
