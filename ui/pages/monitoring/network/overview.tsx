// ui/pages/monitoring/overview.tsx
import Head from "next/head";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import type * as Plotly from "plotly.js";

/* ───────────────────────── Plotly (client only) ───────────────────────── */
export const Plot = dynamic(() => import("react-plotly.js").then((m) => m.default), {
  ssr: false,
  loading: () => (
    <div style={{ height: 320, display: "grid", placeItems: "center", opacity: 0.7 }}>
      Chargement du graphique…
    </div>
  ),
});

/* ───────────────────────── Helpers ───────────────────────── */
// HTTP JSON
async function getJSON<T = unknown>(path: string): Promise<T> {
  const base =
    (typeof window !== "undefined" ? (window as any).NEXT_PUBLIC_API_BASE : undefined) ||
    process.env.NEXT_PUBLIC_API_BASE ||
    "";
  const url = base ? new URL(path, base).toString() : path;

  const res = await fetch(url, { headers: { accept: "application/json" }, cache: "no-store" });
  const ct = res.headers.get("content-type") || "";
  if (!res.ok) {
    const hint = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status} on ${url} — ${hint.slice(0, 200)}`);
  }
  if (!ct.includes("application/json")) {
    const peek = await res.text().catch(() => "");
    throw new Error(`Non-JSON response from ${url}: ${peek.slice(0, 160)}`);
  }
  return (await res.json()) as T;
}
function ok<T>(r: PromiseSettledResult<T>): T | null {
  return r.status === "fulfilled" ? r.value : null;
}
function fmtPct(x?: number | null, digits = 1) {
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  return `${v.toFixed(digits)}%`;
}
function fmtInt(x?: number | null) {
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  return v.toLocaleString("fr-FR");
}
function isFiniteNum(x: number): boolean {
  return Number.isFinite(x);
}

// Convertit "HH:MM" (UTC) → local en appliquant l'offset du navigateur
function shiftHHMMUtcToLocal(hhmm: string): string {
  const [H, M] = hhmm.split(":").map(Number);
  if (!Number.isFinite(H) || !Number.isFinite(M)) return hhmm;
  // construit un Date en UTC, puis récupère l'heure locale
  const d = new Date(Date.UTC(2000, 0, 1, H, M)); // date factice
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

// Optionnel: remplace des zéros isolés par null pour éviter le spike visuel
function deSpikeZeros(y: (number|null)[], window = 1): (number|null)[] {
  const out = y.slice();
  for (let i = 0; i < out.length; i++) {
    const v = out[i];
    if (v === 0) {
      const left = i - window >= 0 ? out[i - window] : null;
      const right = i + window < out.length ? out[i + window] : null;
      const lOk = left != null && left > 50;
      const rOk = right != null && right > 50;
      if (lOk && rOk) out[i] = null; // zéro isolé → null
    }
  }
  return out;
}

/* ───────────────────────── Types (API Overview) ───────────────────────── */
export type OverviewKpis = {
  schema_version: string;
  generated_at: string;
  snapshot_ts_utc: string;
  snapshot_ts_local: string;
  stations_universe: number;
  stations_active: number;
  stations_offline: number;
  availability_bike_pct: number | null;
  availability_dock_pct: number | null;
  penury_pct: number | null;
  saturation_pct: number | null;
  coverage_pct: number | null;
  volatility_today: number | null;
  last_days: number;
  ref_days: number;
};

export type OverviewSnapshotDistributionItem = {
  metric: "bike_avail" | "dock_avail" | "penury" | "saturation";
  count: number;
  total_active: number;
  pct: number; // 0..100
};
export type OverviewSnapshotDistribution = OverviewSnapshotDistributionItem[];

export type OverviewTodayCurve = {
  schema_version: string;
  generated_at: string;
  points: Array<{ hhmm: string; pct: number | null }>;
};

export type OverviewRefMedianCurve = {
  schema_version: string;
  generated_at: string;
  median: Array<{ hhmm: string; pct_median: number | null }>;
};

export type OverviewKpisTodayVsLags = {
  schema_version: string;
  generated_at: string;
  today: {
    avail_bike: number | null;
    avail_dock: number | null;
    pen: number | null;
    sat: number | null;
  };
  lags: {
    "J-7": OverviewKpisTodayVsLags["today"];
    "J-14": OverviewKpisTodayVsLags["today"];
    "J-21": OverviewKpisTodayVsLags["today"];
  };
};

export type OverviewSnapshotMap = {
  schema_version: string;
  generated_at: string;
  rows: Array<{
    station_id: string;
    name: string;
    lat: number | null;
    lon: number | null;
    bikes: number | null;
    docks_avail: number | null;
    is_penury: 0 | 1;
    is_saturation: 0 | 1;
  }>;
};

export type OverviewStationsTension = {
  schema_version: string;
  generated_at: string;
  rows: Array<{
    station_id: string;
    penury_rate: number | null;      // 0..1
    saturation_rate: number | null;  // 0..1
  }>;
};

/* ───────────────────────── Stations index (pour noms + lat/lon) ───────────────────────── */
type StationMeta = {
  station_id: string;
  name?: string | null;
  lat?: number | null;
  lon?: number | null;
};
async function fetchStationsIndex(): Promise<Record<string, StationMeta>> {
  const arr = await getJSON<StationMeta[]>("/stations").catch(() => []);
  const idx: Record<string, StationMeta> = {};
  for (const s of arr) {
    const sid = String((s as any).station_id);
    idx[sid] = {
      station_id: sid,
      name: (s as any).name ?? null,
      lat: (s as any).lat ?? null,
      lon: (s as any).lon ?? null,
    };
  }
  return idx;
}

/* ───────────────────────── Mini Map (snapshot) ───────────────────────── */
type MapRow = OverviewSnapshotMap["rows"][number];

const SnapshotMap = dynamic(async () => {
  const RL = await import("react-leaflet");
  const { MapContainer, TileLayer, CircleMarker, Tooltip, useMap } = RL as any;

  function FitBounds({ rows }: { rows: MapRow[] }) {
    const map = useMap();
    useEffect(() => {
      const pts = rows.filter(r => Number.isFinite(Number(r.lat)) && Number.isFinite(Number(r.lon)));
      if (!pts.length) return;
      // @ts-ignore
      const b = (window as any).L.latLngBounds(pts.map(r => [r.lat, r.lon]));
      map.fitBounds(b, { padding: [20, 20] });
    }, [rows, map]);
    return null;
  }

  function MapInner({ rows }: { rows: MapRow[] }) {
    const valid = rows.filter(r => Number.isFinite(Number(r.lat)) && Number.isFinite(Number(r.lon)));
    const latMed = valid.length ? valid.map(r=>Number(r.lat)).sort((a,b)=>a-b)[Math.floor(valid.length/2)] : 48.8566;
    const lonMed = valid.length ? valid.map(r=>Number(r.lon)).sort((a,b)=>a-b)[Math.floor(valid.length/2)] : 2.3522;

    return (
      <MapContainer center={[latMed, lonMed]} zoom={12} style={{ height: "100%", width: "100%", background: "#fff" }}>
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png"
          attribution='&copy; OpenStreetMap, &copy; <a href="https://carto.com/">CARTO</a>'
          detectRetina
        />
        <FitBounds rows={valid} />
        {valid.map((r) => {
          const pen = r.is_penury === 1;
          const sat = r.is_saturation === 1;
          const col = pen ? "#ef4444" : sat ? "#3b82f6" : "#10b981";
          const rad = Math.max(3, Math.min(9, Math.sqrt(Math.max(0, Number(r.bikes ?? 0))) + (sat ? 2 : 0)));
          return (
            <CircleMarker key={r.station_id} center={[Number(r.lat), Number(r.lon)]}
              radius={rad}
              pathOptions={{ color: col, weight: 0.8, fillColor: col, fillOpacity: 0.85 }}>
              <Tooltip>
                <div style={{display:"grid", gap:4}}>
                  <div><b>{r.name}</b></div>
                  <div>vélos: {Number.isFinite(Number(r.bikes)) ? Number(r.bikes) : "?"}</div>
                  <div>places: {Number.isFinite(Number(r.docks_avail)) ? Number(r.docks_avail) : "?"}</div>
                  {pen && <div style={{color:"#ef4444"}}>pénurie</div>}
                  {sat && <div style={{color:"#3b82f6"}}>saturation</div>}
                  <a href={`/monitoring/network/dynamics?station_id=${encodeURIComponent(r.station_id)}`} style={{textDecoration:"underline"}}>
                    Voir dynamique →
                  </a>
                </div>
              </Tooltip>
            </CircleMarker>
          );
        })}
      </MapContainer>
    );
  }

  return MapInner;
}, { ssr: false });

/* ───────────────────────── Page ───────────────────────── */
export default function OverviewPage() {
  // State
  const [kpis, setKpis] = useState<OverviewKpis | null>(null);
  const [dist, setDist] = useState<OverviewSnapshotDistribution | null>(null);
  const [today, setToday] = useState<OverviewTodayCurve | null>(null);
  const [refMedian, setRefMedian] = useState<OverviewRefMedianCurve | null>(null);
  const [lags, setLags] = useState<OverviewKpisTodayVsLags | null>(null);
  const [snapMap, setSnapMap] = useState<OverviewSnapshotMap | null>(null);
  const [tension, setTension] = useState<OverviewStationsTension | null>(null);

  // ▼ ajout: index stations pour afficher les noms
  const [stationsIdx, setStationsIdx] = useState<Record<string, StationMeta>>({});

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load all
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setLoading(true);
        const res = await Promise.allSettled([
          getJSON<OverviewKpis>("/monitoring/network/overview/kpis"),
          getJSON<OverviewSnapshotDistribution>("/monitoring/network/overview/snapshot_distribution"),
          getJSON<OverviewTodayCurve>("/monitoring/network/overview/today_curve"),
          getJSON<OverviewRefMedianCurve>("/monitoring/network/overview/ref_median_curve"),
          getJSON<OverviewKpisTodayVsLags>("/monitoring/network/overview/kpis_today_vs_lags"),
          getJSON<OverviewSnapshotMap>("/monitoring/network/overview/snapshot_map"),
          getJSON<OverviewStationsTension>("/monitoring/network/overview/stations_tension"),
        ]);
        if (!alive) return;
        setKpis(ok(res[0]));
        setDist(ok(res[1]));
        setToday(ok(res[2]));
        setRefMedian(ok(res[3]));
        setLags(ok(res[4]));
        setSnapMap(ok(res[5]));
        setTension(ok(res[6]));

        // ▼ ajout: charger l'index des stations (peut échouer silencieusement)
        fetchStationsIndex().then(setStationsIdx).catch(() => setStationsIdx({}));

        setError(null);
      } catch (e: any) {
        if (alive) setError(String(e?.message ?? e));
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, []);

  const generatedAt = kpis?.generated_at ?? today?.generated_at ?? refMedian?.generated_at ?? snapMap?.generated_at;

  // Curves
  const curveToday: Partial<Plotly.PlotData> | null = useMemo(() => {
    if (!today?.points?.length) return null;
    const x = today.points.map(p => p.hhmm);
    const y = today.points.map(p => (Number.isFinite(Number(p.pct)) ? Number(p.pct) : null));
    return { x, y, type: "scatter", mode: "lines", name: "Hier (UTC)", connectgaps: false };
  }, [today]);
  const curveRef: Partial<Plotly.PlotData> | null = useMemo(() => {
    if (!refMedian?.median?.length) return null;
    const x = refMedian.median.map(p => p.hhmm);
    const y = refMedian.median.map(p => (Number.isFinite(Number(p.pct_median)) ? Number(p.pct_median) : null));
    return { x, y, type: "scatter", mode: "lines", name: "Référence (UTC)", connectgaps: false };
  }, [refMedian]);

  // Distribution snapshot → barres
  const distBars: Partial<Plotly.PlotData>[] = useMemo(() => {
    const d = Array.isArray(dist) ? dist : [];
    if (!d.length) return [];
    const labelMap: Record<string, string> = {
      bike_avail: "≥1 vélo",
      dock_avail: "≥1 place",
      penury: "pénurie",
      saturation: "saturation",
    };
    const x = d.map(x => labelMap[x.metric] ?? x.metric);
    const y = d.map(x => Number(x.pct));
    return [{ x, y, type: "bar" as const, name: "Snapshot (%)" }];
  }, [dist]);

  // Tension — top 20 penury/saturation
  const topTension = useMemo(() => {
    const rows = tension?.rows ?? [];
    const pen = [...rows]
      .filter(r => Number.isFinite(Number(r.penury_rate)))
      .sort((a,b)=>Number(b.penury_rate)-Number(a.penury_rate))
      .slice(0, 20);
    const sat = [...rows]
      .filter(r => Number.isFinite(Number(r.saturation_rate)))
      .sort((a,b)=>Number(b.saturation_rate)-Number(a.saturation_rate))
      .slice(0, 20);
    return { pen, sat };
  }, [tension]);

  // ▼ ajout: vue enrichie avec noms + % + lien
  const topTensionView = useMemo(() => {
    const toView = (rows: OverviewStationsTension["rows"]) =>
      rows.map(r => {
        const id = String(r.station_id);
        const meta = stationsIdx[id];
        return {
          station_id: id,
          name: meta?.name ?? id,
          penury_pct: Number.isFinite(Number(r.penury_rate)) ? Number(r.penury_rate) * 100 : null,
          saturation_pct: Number.isFinite(Number(r.saturation_rate)) ? Number(r.saturation_rate) * 100 : null,
          href: `/monitoring/network/dynamics?station_id=${encodeURIComponent(id)}`
        };
      });
    return { pen: toView(topTension.pen), sat: toView(topTension.sat) };
  }, [topTension, stationsIdx]);

  return (
    <>
      <Head>
        <title>Monitoring — Network / Overview</title>
        <meta name="description" content="KPIs réseau, snapshot, courbes du jour, référence et carte." />
        {/* Leaflet CSS */}
        <link
          rel="stylesheet"
          href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
          integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
          crossOrigin=""
        />
        <style
          dangerouslySetInnerHTML={{
            __html: `
              .leaflet-container { background: #fff !important; }
              html, body, #__next { height: auto !important; }
              html, body { overflow-y: auto !important; }
            `,
          }}
        />
      </Head>

      <main style={{ padding: "2rem", maxWidth: 1200, margin: "0 auto" }}>
        {/* Header */}
        <header style={{ display: "flex", justifyContent: "space-between", gap: 16 }}>
          <div>
            <h1 style={{ margin: 0 }}>Network — Overview</h1>
            <div className="small" style={{ opacity: 0.7 }}>
              {generatedAt ? `Généré : ${new Date(generatedAt).toLocaleString("fr-FR")}` : "—"}
            </div>
          </div>

          <nav style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <Link
              href="/monitoring/network/stations"
              style={{ padding: "6px 10px", borderRadius: 8, border: "1px solid #d1d5db", background: "white", color: "#111827", textDecoration: "none" }}
            >
              Stations
            </Link>
            <Link
              href="/monitoring/network/dynamics"
              style={{ padding: "6px 10px", borderRadius: 8, border: "1px solid #d1d5db", background: "white", color: "#111827", textDecoration: "none" }}
            >
              Dynamics
            </Link>
          </nav>
        </header>

        {/* Loading / Error */}
        {loading && <Banner kind="info">Chargement…</Banner>}
        {error && <Banner kind="error">{error}</Banner>}

        {/* KPIs Snapshot */}
        <section style={{ marginTop: 24 }}>
          <h2 style={{ margin: "12px 0" }}>Résumé — snapshot</h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(5, minmax(0, 1fr))", gap: 12 }}>
            <Kpi label="Stations actives" value={kpis?.stations_active} fmt={fmtInt as any} />
            <Kpi label="Hors ligne" value={kpis?.stations_offline} fmt={fmtInt as any} />
            <Kpi label="≥1 vélo" value={kpis?.availability_bike_pct} fmt={(v)=>fmtPct(v,1)} />
            <Kpi label="≥1 place" value={kpis?.availability_dock_pct} fmt={(v)=>fmtPct(v,1)} />
            <Kpi label="Couverture (récente)" value={Number(kpis?.coverage_pct ?? NaN)} fmt={(v)=>fmtPct(v,1)} />
          </div>
          <div style={{ fontSize: 12, opacity: 0.7, marginTop: 6 }}>
            Fenêtre récente: {kpis?.last_days ?? "—"} j · Réf: {kpis?.ref_days ?? "—"} j · Schema v{kpis?.schema_version ?? "—"}
            <span style={{ marginLeft: 8, opacity: 0.8 }}>({kpis?.snapshot_ts_local ?? "—"})</span>
          </div>
        </section>

        {/* Distribution Snapshot + Carte */}
        <section style={{ marginTop: 24 }}>
          <h2 style={{ margin: "12px 0" }}>Snapshot — distribution & carte</h2>
          <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1.5fr)", gap: 16 }}>
            <Card>
              <h3 style={{ margin: "6px 0 10px 0", fontSize: 16 }}>Distribution (%)</h3>
              {distBars.length ? (
                <Plot
                  data={distBars as Plotly.Data[]}
                  layout={{
                    autosize: true,
                    height: 280,
                    margin: { l: 48, r: 10, t: 10, b: 36 },
                    yaxis: { title: { text: "%" }, range: [0, 100] },
                    xaxis: { title: { text: "Etat" } },
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: "100%" }}
                />
              ) : (
                <Empty>Pas de distribution snapshot.</Empty>
              )}
            </Card>

            <Card>
              <h3 style={{ margin: "6px 0 10px 0", fontSize: 16 }}>Carte snapshot</h3>
              <div style={{ width: "100%", height: 320 }}>
                {snapMap?.rows?.length ? (
                  <SnapshotMap rows={snapMap.rows} />
                ) : (
                  <Empty>Pas de données carte snapshot.</Empty>
                )}
              </div>
              <div style={{ fontSize: 12, opacity: 0.7, marginTop: 6 }}>
                Fond: Carto Light (no labels) · Rouge=pénurie · Bleu=saturation · Vert=OK (taille ~ √vélos).
              </div>
            </Card>
          </div>
        </section>

        {/* Courbes du jour vs référence */}
        <section style={{ marginTop: 24 }}>
          <h2 style={{ margin: "12px 0" }}>Hier (UTC) vs Référence</h2>
          {curveToday || curveRef ? (
            <>
              <Plot
                data={[curveRef, curveToday].filter(Boolean) as Plotly.Data[]}
                layout={{
                  autosize: true,
                  height: 380,
                  margin: { l: 52, r: 10, t: 30, b: 40 },
                  xaxis: { title: { text: "Heure (UTC)" } },
                  yaxis: { title: { text: "Stations avec ≥1 vélo (%)" }, range: [0, 100] },
                  legend: { orientation: "h" },
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                  hovermode: "x unified",
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: "100%" }}
              />
              <div style={{ fontSize: 12, opacity: 0.7, marginTop: 6 }}>
                Les courbes sont en UTC. « Hier (UTC) » correspond au jour UTC complet (00:00–23:55).
              </div>
            </>
          ) : (
            <Empty>Courbes non disponibles.</Empty>
          )}

        </section>

        {/* KPIs Today vs J-7/J-14/J-21 */}
        <section style={{ marginTop: 24 }}>
          <h2 style={{ margin: "12px 0" }}>Aujourd’hui vs J-7 / J-14 / J-21</h2>
          {lags ? (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: 12 }}>
              <CardSmall title="≥1 vélo (temps %)" values={[
                { label: "J", v: lags.today.avail_bike },
                { label: "J-7", v: lags.lags["J-7"].avail_bike },
                { label: "J-14", v: lags.lags["J-14"].avail_bike },
                { label: "J-21", v: lags.lags["J-21"].avail_bike },
              ]}/>
              <CardSmall title="≥1 place (temps %)" values={[
                { label: "J", v: lags.today.avail_dock },
                { label: "J-7", v: lags.lags["J-7"].avail_dock },
                { label: "J-14", v: lags.lags["J-14"].avail_dock },
                { label: "J-21", v: lags.lags["J-21"].avail_dock },
              ]}/>
              <CardSmall title="Pénurie (temps %)" values={[
                { label: "J", v: lags.today.pen },
                { label: "J-7", v: lags.lags["J-7"].pen },
                { label: "J-14", v: lags.lags["J-14"].pen },
                { label: "J-21", v: lags.lags["J-21"].pen },
              ]}/>
              <CardSmall title="Saturation (temps %)" values={[
                { label: "J", v: lags.today.sat },
                { label: "J-7", v: lags.lags["J-7"].sat },
                { label: "J-14", v: lags.lags["J-14"].sat },
                { label: "J-21", v: lags.lags["J-21"].sat },
              ]}/>
            </div>
          ) : (
            <Empty>Comparatifs non disponibles.</Empty>
          )}
        </section>

        {/* Stations en tension */}
        <section style={{ marginTop: 24 }}>
          <h2 style={{ margin: "12px 0" }}>Stations en tension (fenêtre récente)</h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 16 }}>
            <Card>
              <h3 style={{ margin: "6px 0 10px 0", fontSize: 16 }}>Top pénurie</h3>
              {topTensionView.pen.length ? (
                <ul style={{ margin: 0, padding: 0, listStyle: "none", display: "grid", gap: 8 }}>
                  {topTensionView.pen.map((r) => (
                    <li key={r.station_id}>
                      <a href={r.href} style={{ textDecoration: "none", color: "inherit" }}>
                        <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) 64px", alignItems: "center", gap: 10 }}>
                          <div style={{ overflow: "hidden" }}>
                            <div style={{ whiteSpace: "nowrap", textOverflow: "ellipsis" }}>
                              <b>{r.name}</b> <span style={{ opacity: 0.6 }}>({r.station_id})</span>
                            </div>
                            <div style={{ height: 6, borderRadius: 999, background: "#111827", marginTop: 6, position: "relative", overflow: "hidden" }}>
                              <div
                                style={{
                                  width: `${Math.max(0, Math.min(100, r.penury_pct ?? 0))}%`,
                                  height: "100%",
                                  background: "#ef4444",
                                }}
                              />
                            </div>
                          </div>
                          <div style={{ textAlign: "right", fontWeight: 700 }}>
                            {fmtPct(r.penury_pct ?? undefined, 1)}
                          </div>
                        </div>
                      </a>
                    </li>
                  ))}
                </ul>
              ) : (
                <Empty>—</Empty>
              )}
            </Card>

            <Card>
              <h3 style={{ margin: "6px 0 10px 0", fontSize: 16 }}>Top saturation</h3>
              {topTensionView.sat.length ? (
                <ul style={{ margin: 0, padding: 0, listStyle: "none", display: "grid", gap: 8 }}>
                  {topTensionView.sat.map((r) => (
                    <li key={r.station_id}>
                      <a href={r.href} style={{ textDecoration: "none", color: "inherit" }}>
                        <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) 64px", alignItems: "center", gap: 10 }}>
                          <div style={{ overflow: "hidden" }}>
                            <div style={{ whiteSpace: "nowrap", textOverflow: "ellipsis" }}>
                              <b>{r.name}</b> <span style={{ opacity: 0.6 }}>({r.station_id})</span>
                            </div>
                            <div style={{ height: 6, borderRadius: 999, background: "#111827", marginTop: 6, position: "relative", overflow: "hidden" }}>
                              <div
                                style={{
                                  width: `${Math.max(0, Math.min(100, r.saturation_pct ?? 0))}%`,
                                  height: "100%",
                                  background: "#3b82f6",
                                }}
                              />
                            </div>
                          </div>
                          <div style={{ textAlign: "right", fontWeight: 700 }}>
                            {fmtPct(r.saturation_pct ?? undefined, 1)}
                          </div>
                        </div>
                      </a>
                    </li>
                  ))}
                </ul>
              ) : (
                <Empty>—</Empty>
              )}
            </Card>
          </div>
          {tension && (
            <div style={{ marginTop: 8, fontSize: 12, opacity: 0.7 }}>
              Schema v{tension.schema_version} — generated {tension.generated_at}
            </div>
          )}
        </section>
      </main>
    </>
  );
}

/* ───────────────────────── UI atoms ───────────────────────── */
function Kpi({
  label,
  value,
  fmt,
}: {
  label: string;
  value: number | null | undefined;
  fmt?: (v: number | null | undefined) => string;
}) {
  const text = fmt ? fmt(value) : Number.isFinite(Number(value)) ? String(value) : "—";
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

function Card({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ border: "1px solid #374151", background: "rgba(15, 23, 42, 0.5)", borderRadius: 12, padding: 12 }}>
      {children}
    </div>
  );
}
function CardSmall({
  title,
  values,
}: {
  title: string;
  values: { label: string; v: number | null }[];
}) {
  return (
    <div
      style={{
        border: "1px solid #374151",
        background: "rgba(15, 23, 42, 0.5)",
        borderRadius: 12,
        padding: 12,
      }}
    >
      <h3 style={{ margin: "6px 0 10px 0", fontSize: 16 }}>{title}</h3>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: 8 }}>
        {values.map((x) => (
          <div key={x.label} style={{ textAlign: "center" }}>
            <div style={{ fontSize: 12, color: "#9CA3AF" }}>{x.label}</div>
            <div style={{ fontSize: 18, color: "#E5E7EB", fontWeight: 600 }}>{fmtPct(x.v ?? undefined, 1)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function Banner({ kind, children }: { kind: "info" | "error"; children: React.ReactNode }) {
  const style =
    kind === "error"
      ? { border: "1px solid #DC2626", background: "rgba(220, 38, 38, 0.08)", color: "#F87171" }
      : { border: "1px solid #374151", background: "rgba(15, 23, 42, 0.5)", color: "inherit", opacity: 0.85 };
  return (
    <div style={{ marginTop: 16, borderRadius: 10, padding: "10px 12px", fontSize: 13, ...style }}>{children}</div>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        border: "1px solid #374151",
        background: "rgba(15,23,42,0.35)",
        borderRadius: 12,
        padding: "14px",
        color: "#9CA3AF",
        fontSize: 13,
      }}
    >
      {children}
    </div>
  );
}
