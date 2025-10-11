// ui/pages/monitoring/network/stations.tsx
import Head from "next/head";
import type { GetStaticProps } from "next";
import dynamic from "next/dynamic";
import { useMemo, useState } from "react";
import { getNetworkStations } from "@/lib/services/monitoring";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

type Station = {
  station_id: string;
  name?: string;
  lat?: number;
  lon?: number;
  capacity_est?: number;
  volatility?: number;       // 0..1 (ou %)
  penury_rate?: number;      // 0..1
  saturation_rate?: number;  // 0..1
  coverage_pct?: number;     // 0..1
};

type Props = {
  stations: Station[];
  generatedAt: string;
};

export const revalidate = 600;

export const getStaticProps: GetStaticProps<Props> = async () => {
  const payload = (await getNetworkStations()) ?? {};
  const stations: Station[] = Array.isArray((payload as any).stations)
    ? (payload as any).stations
    : Array.isArray(payload)
    ? (payload as any)
    : [];

  return {
    props: {
      stations,
      generatedAt: new Date().toISOString(),
    },
    revalidate,
  };
};

function fmt(n?: number | null, d = 1) {
  const v = Number(n);
  return Number.isFinite(v)
    ? new Intl.NumberFormat("fr-FR", { maximumFractionDigits: d }).format(v)
    : "—";
}

function pct01(x?: number | null, d = 1) {
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  const p = v <= 1 ? v * 100 : v;
  return `${p.toFixed(d)}%`;
}

export default function StationsPage({ stations, generatedAt }: Props) {
  const [colorBy, setColorBy] = useState<keyof Station>("penury_rate");

  const scatter = useMemo(() => {
    if (!stations?.length) return null;
    const x = stations.map((s) => s.volatility ?? null);
    const y = stations.map((s) => s.coverage_pct ?? null);
    const c = stations.map((s) => Number(s[colorBy] ?? 0));
    const text = stations.map((s) => `${s.name ?? s.station_id}`);
    return { x, y, c, text };
  }, [stations, colorBy]);

  return (
    <>
      <Head>
        <title>Monitoring — Network Stations</title>
        <meta
          name="description"
          content="Stations : couverture, volatilité, pénurie et saturation."
        />
      </Head>

      <main style={{ padding: "2rem", maxWidth: 1200, margin: "0 auto" }}>
        <header
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: 16,
          }}
        >
          <div>
            <h1 style={{ margin: 0 }}>Network · Stations</h1>
            <div className="small" style={{ opacity: 0.7 }}>
              Généré : {new Date(generatedAt).toLocaleString("fr-FR")}
            </div>
          </div>

          <div>
            <label htmlFor="colorBy" style={{ marginRight: 8 }}>
              Couleur par :
            </label>
            <select
              id="colorBy"
              value={colorBy}
              onChange={(e) => setColorBy(e.target.value as keyof Station)}
            >
              <option value="penury_rate">penury_rate</option>
              <option value="saturation_rate">saturation_rate</option>
              <option value="capacity_est">capacity_est</option>
              <option value="volatility">volatility</option>
              <option value="coverage_pct">coverage_pct</option>
            </select>
          </div>
        </header>

        <section style={{ marginTop: 24 }}>
          {!stations.length ? (
            <div className="small" style={{ opacity: 0.7 }}>
              Aucune donnée station disponible.
            </div>
          ) : (
            <Plot
              data={[
                {
                  x: scatter?.x,
                  y: scatter?.y,
                  mode: "markers",
                  type: "scattergl",
                  text: scatter?.text,
                  marker: {
                    color: scatter?.c,
                    colorscale: "YlOrRd",
                    colorbar: { title: { text: String(colorBy) } },
                    size: 6,
                    opacity: 0.8,
                  },
                } as any,
              ]}
              layout={{
                autosize: true,
                height: 500,
                margin: { l: 60, r: 40, t: 30, b: 50 },
                xaxis: { title: { text: "Volatility" } },
                yaxis: { title: { text: "Coverage (%)" } },
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: "100%" }}
            />
          )}
        </section>

        <section style={{ marginTop: 28 }}>
          <h2 style={{ margin: "12px 0" }}>Aperçu tabulaire (50 premières)</h2>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  <th align="left">Station</th>
                  <th align="right">Capacité</th>
                  <th align="right">Volatilité</th>
                  <th align="right">Pénurie</th>
                  <th align="right">Saturation</th>
                  <th align="right">Couverture</th>
                </tr>
              </thead>
              <tbody>
                {stations.slice(0, 50).map((s) => (
                  <tr key={s.station_id}>
                    <td>{s.name ?? s.station_id}</td>
                    <td align="right">{fmt(s.capacity_est, 0)}</td>
                    <td align="right">{pct01(s.volatility, 1)}</td>
                    <td align="right">{pct01(s.penury_rate, 1)}</td>
                    <td align="right">{pct01(s.saturation_rate, 1)}</td>
                    <td align="right">{pct01(s.coverage_pct, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </main>
    </>
  );
}
