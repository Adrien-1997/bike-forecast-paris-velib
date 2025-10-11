// ui/pages/monitoring/data-health.tsx
import Head from "next/head";
import type { GetStaticProps } from "next";
import { useMemo } from "react";
import { getManifest, getNetworkStations } from "@/lib/services/monitoring";

type Manifest = {
  generated_at?: string;
  files?: Record<
    string,
    Record<
      string,
      {
        path: string;
        updated?: string;
      }
    >
  >;
};

type StationRow = {
  station_id: string;
  name?: string;
  bins_seen?: number;
  coverage_pct?: number; // 0..100 or 0..1 depending on source; we normalize below
  volatility?: number;
  capacity_est?: number;
  penury_rate?: number;
  saturation_rate?: number;
};

type Props = {
  manifest: Manifest | null;
  stations: StationRow[];
  generatedAt: string;
};

export const revalidate = 600; // ISR 10 min

export const getStaticProps: GetStaticProps<Props> = async () => {
  const [manifest, stationsPayload] = await Promise.all([
    getManifest().catch(() => null),
    getNetworkStations().catch(() => null),
  ]);

  // /monitoring/network/stations peut renvoyer { stations:[...] } ou directement un array
  const stations: StationRow[] = Array.isArray((stationsPayload as any)?.stations)
    ? (stationsPayload as any).stations
    : Array.isArray(stationsPayload)
    ? (stationsPayload as any)
    : [];

  return {
    props: {
      manifest: manifest ?? null,
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
  // tolère 0..1 ou 0..100 → on normalise sur 0..100
  const p = v <= 1 ? v * 100 : v;
  return `${p.toFixed(d)}%`;
}

export default function DataHealthPage({ manifest, stations, generatedAt }: Props) {
  // KPIs couverture & qualité basés sur l'agrégat des stations
  const kpis = useMemo(() => {
    if (!stations?.length) {
      return {
        stations: 0,
        coverage_mean: null as number | null,
        coverage_p10: null as number | null,
        coverage_p90: null as number | null,
        penury_mean: null as number | null,
        saturation_mean: null as number | null,
      };
    }
    const covs: number[] = [];
    const pen: number[] = [];
    const sat: number[] = [];
    for (const s of stations) {
      const c = Number(s.coverage_pct);
      if (Number.isFinite(c)) covs.push(c <= 1 ? c * 100 : c);
      const p = Number(s.penury_rate);
      if (Number.isFinite(p)) pen.push(p <= 1 ? p * 100 : p);
      const t = Number(s.saturation_rate);
      if (Number.isFinite(t)) sat.push(t <= 1 ? t * 100 : t);
    }
    covs.sort((a, b) => a - b);
    const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
    const pct = (xs: number[], p: number) =>
      xs.length ? xs[Math.max(0, Math.min(xs.length - 1, Math.floor((p / 100) * xs.length)))] : null;

    return {
      stations: stations.length,
      coverage_mean: mean(covs),
      coverage_p10: pct(covs, 10),
      coverage_p90: pct(covs, 90),
      penury_mean: mean(pen),
      saturation_mean: mean(sat),
    };
  }, [stations]);

  // Fraîcheur des artefacts via manifest
  const freshness = useMemo(() => {
    const updatedPaths: { label: string; iso?: string }[] = [];
    const files = manifest?.files ?? {};
    // on extrait quelques clés représentatives si dispo
    const pick = (group: string, key: string, label: string) => {
      const iso = (files as any)?.[group]?.[key]?.updated;
      if (iso) updatedPaths.push({ label, iso });
    };
    pick("model_perf", "daily_h15", "Perf daily h15");
    pick("model_perf", "daily_h60", "Perf daily h60");
    pick("network", "stations", "Stations");
    pick("network", "dynamics", "Dynamics");
    pick("drift", "summary", "Drift summary");

    return updatedPaths.sort((a, b) => (a.label < b.label ? -1 : 1));
  }, [manifest]);

  return (
    <>
      <Head>
        <title>Monitoring — Data Health</title>
        <meta
          name="description"
          content="Santé des données : couverture par station, fraîcheur des artefacts et indicateurs globaux."
        />
      </Head>

      <main style={{ padding: "2rem", maxWidth: 1200, margin: "0 auto" }}>
        <header style={{ display: "flex", justifyContent: "space-between", gap: 16 }}>
          <div>
            <h1 style={{ margin: 0 }}>Data Health</h1>
            <div className="small" style={{ opacity: 0.7 }}>
              Généré: {new Date(generatedAt).toLocaleString("fr-FR")}
            </div>
          </div>
        </header>

        {/* KPIs couverture & qualité */}
        <section style={{ marginTop: 16 }}>
          <h2 style={{ margin: "12px 0" }}>Couverture & Qualité (stations)</h2>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
              gap: 12,
            }}
          >
            <Kpi label="Stations (total)" value={kpis.stations} />
            <Kpi label="Couverture moyenne" value={kpis.coverage_mean != null ? `${fmt(kpis.coverage_mean)}%` : "—"} />
            <Kpi label="Couverture p10" value={kpis.coverage_p10 != null ? `${fmt(kpis.coverage_p10)}%` : "—"} />
            <Kpi label="Couverture p90" value={kpis.coverage_p90 != null ? `${fmt(kpis.coverage_p90)}%` : "—"} />
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
              gap: 12,
              marginTop: 12,
            }}
          >
            <Kpi label="Taux de pénurie (moy.)" value={kpis.penury_mean != null ? `${fmt(kpis.penury_mean)}%` : "—"} />
            <Kpi
              label="Taux de saturation (moy.)"
              value={kpis.saturation_mean != null ? `${fmt(kpis.saturation_mean)}%` : "—"}
            />
            <Kpi label="Fraîcheur manifest" value={manifest?.generated_at ?? "—"} />
          </div>
        </section>

        {/* Fraîcheur des artefacts */}
        <section style={{ marginTop: 28 }}>
          <h2 style={{ margin: "12px 0" }}>Fraîcheur des artefacts</h2>
          {!freshness.length ? (
            <div className="small" style={{ opacity: 0.7 }}>
              Manifest incomplet ou non disponible.
            </div>
          ) : (
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr>
                    <th align="left">Artefact</th>
                    <th align="left">Dernière mise à jour (UTC)</th>
                    <th align="left">Heure Paris</th>
                  </tr>
                </thead>
                <tbody>
                  {freshness.map((f) => (
                    <tr key={f.label}>
                      <td>{f.label}</td>
                      <td>{f.iso ?? "—"}</td>
                      <td>{toParisHHmm(f.iso)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        {/* Table stations (échantillon) */}
        <section style={{ marginTop: 28 }}>
          <h2 style={{ margin: "12px 0" }}>Stations — aperçu</h2>
          {!stations.length ? (
            <div className="small" style={{ opacity: 0.7 }}>
              Aucune donnée station disponible depuis l’API.
            </div>
          ) : (
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
              {stations.length > 50 && (
                <div className="small" style={{ opacity: 0.7, marginTop: 6 }}>
                  {stations.length - 50} lignes supplémentaires… (on paginera/virtualisera plus tard)
                </div>
              )}
            </div>
          )}
        </section>
      </main>
    </>
  );
}

/* ─────────────── UI helpers ─────────────── */

function toParisHHmm(iso?: string) {
  if (!iso) return "—";
  const s = iso.endsWith("Z") ? iso : `${iso}Z`;
  const d = new Date(s);
  return new Intl.DateTimeFormat("fr-FR", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "Europe/Paris",
  }).format(d);
}

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
