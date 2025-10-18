// ui/pages/monitoring/overview.tsx
import Head from "next/head";
import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import type * as Plotly from "plotly.js";
import MonitoringNav from "@/components/monitoring/MonitoringNav";

/* ───────────────────────── Plotly (client only) ───────────────────────── */
const Plot = dynamic(() => import("react-plotly.js").then((m) => m.default), {
  ssr: false,
  loading: () => (
    <div style={{ height: 320, display: "grid", placeItems: "center", opacity: 0.7 }}>
      Loading chart…
    </div>
  ),
});

/* ───────────────────────── Helpers ───────────────────────── */
// HTTP JSON robuste
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
const isFiniteNum = (x: number) => Number.isFinite(x);

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
  today: { avail_bike: number | null; avail_dock: number | null; pen: number | null; sat: number | null };
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
  rows: Array<{ station_id: string; penury_rate: number | null; saturation_rate: number | null }>;
};

/* ───────────────────────── Stations index (fallback noms/coords) ───────────────────────── */
type StationMeta = { station_id: string; name?: string | null; lat?: number | null; lon?: number | null };
async function fetchStationsIndex(): Promise<Record<string, StationMeta>> {
  const arr = await getJSON<StationMeta[]>("/stations").catch(() => []);
  const idx: Record<string, StationMeta> = {};
  for (const s of arr) {
    const sid = String((s as any).station_id);
    idx[sid] = {
      station_id: sid,
      name: (s as any).name ?? null,
      lat: Number.isFinite(Number((s as any).lat)) ? Number((s as any).lat) : null,
      lon: Number.isFinite(Number((s as any).lon)) ? Number((s as any).lon) : null,
    };
  }
  return idx;
}

/* ───────────────────────── Mini Map (snapshot) ───────────────────────── */
type MapRow = OverviewSnapshotMap["rows"][number];

const SnapshotMap = dynamic(async () => {
  const RL = await import("react-leaflet");
  const { MapContainer, TileLayer, CircleMarker, Tooltip, useMap } = RL as any;
  const { useEffect, useMemo, useState } = await import("react");

  function FitBounds({ rows }: { rows: MapRow[] }) {
    const map = useMap();
    useEffect(() => {
      const pts = rows.filter(r => Number.isFinite(Number(r.lat)) && Number.isFinite(Number(r.lon)));
      if (!pts.length) return;
      let minLat = 90, maxLat = -90, minLon = 180, maxLon = -180;
      for (const r of pts) {
        const la = Number(r.lat), lo = Number(r.lon);
        if (la < minLat) minLat = la;
        if (la > maxLat) maxLat = la;
        if (lo < minLon) minLon = lo;
        if (lo > maxLon) maxLon = lo;
      }
      if (minLat <= maxLat && minLon <= maxLon) {
        map.fitBounds([[minLat, minLon], [maxLat, maxLon]], { padding: [20, 20] });
      }
    }, [rows, map]);
    return null;
  }

  function MapInner({ rows }: { rows: MapRow[] }) {
    const valid = useMemo(() => rows.filter(r => Number.isFinite(Number(r.lat)) && Number.isFinite(Number(r.lon))), [rows]);
    const latMed = valid.length ? valid.map(r=>Number(r.lat)).sort((a,b)=>a-b)[Math.floor(valid.length/2)] : 48.8566;
    const lonMed = valid.length ? valid.map(r=>Number(r.lon)).sort((a,b)=>a-b)[Math.floor(valid.length/2)] : 2.3522;

    const [tileUrl, setTileUrl] = useState("https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png");
    useEffect(() => {
      const img = new Image();
      img.onerror = () => setTileUrl("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png");
      img.src = "https://a.basemaps.cartocdn.com/light_nolabels/3/4/2.png";
    }, []);

    return (
      <div style={{ position: "relative", width: "100%", height: "100%" }}>
        <MapContainer center={[latMed, lonMed]} zoom={12} style={{ height: "100%", width: "100%", background: "#fff" }}>
          <TileLayer
            url={tileUrl}
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
              <CircleMarker
                key={r.station_id}
                center={[Number(r.lat), Number(r.lon)]}
                radius={rad}
                pathOptions={{ color: col, weight: 0.8, fillColor: col, fillOpacity: 0.85 }}
              >
                <Tooltip>
                  <div style={{display:"grid", gap:4}}>
                    <div><b>{r.name}</b></div>
                    <div>bikes: {Number.isFinite(Number(r.bikes)) ? Number(r.bikes) : "?"}</div>
                    <div>docks: {Number.isFinite(Number(r.docks_avail)) ? Number(r.docks_avail) : "?"}</div>
                    {pen && <div style={{color:"#ef4444"}}>penury</div>}
                    {sat && <div style={{color:"#3b82f6"}}>saturation</div>}
                    <a href={`/monitoring/network/dynamics?station_id=${encodeURIComponent(r.station_id)}`} style={{textDecoration:"underline"}}>
                      View dynamics →
                    </a>
                  </div>
                </Tooltip>
              </CircleMarker>
            );
          })}
        </MapContainer>

        {/* Légende snapshot */}
        <div
          style={{
            position: "absolute",
            right: 8,
            bottom: 8,
            zIndex: 1000,
            background: "rgba(255,255,255,0.92)",
            color: "#111",
            borderRadius: 10,
            padding: "8px 10px",
            fontSize: 12,
            boxShadow: "0 2px 10px rgba(0,0,0,0.15)",
            border: "1px solid #0001",
          }}
        >
          <div style={{ fontWeight: 700, marginBottom: 6 }}>Snapshot</div>
          <div style={{ display: "grid", gap: 4 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ width: 12, height: 12, borderRadius: 999, background: "#ef4444", border: "1px solid #0002" }} />
              <span>Pénurie</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ width: 12, height: 12, borderRadius: 999, background: "#3b82f6", border: "1px solid #0002" }} />
              <span>Saturation</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ width: 12, height: 12, borderRadius: 999, background: "#10b981", border: "1px solid #0002" }} />
              <span>OK</span>
            </div>
          </div>
        </div>
      </div>
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

        fetchStationsIndex().then(idx => alive && setStationsIdx(idx)).catch(() => alive && setStationsIdx({}));

        setError(null);
      } catch (e: any) {
        if (alive) setError(String(e?.message ?? e));
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, []);

  const generatedAt =
    kpis?.generated_at ?? today?.generated_at ?? refMedian?.generated_at ?? snapMap?.generated_at;

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

  // Tension — top 20
  const topTension = useMemo(() => {
    const rows = tension?.rows ?? [];
    const pen = [...rows].filter(r => Number.isFinite(Number(r.penury_rate)))
      .sort((a,b)=>Number(b.penury_rate)-Number(a.penury_rate)).slice(0, 20);
    const sat = [...rows].filter(r => Number.isFinite(Number(r.saturation_rate)))
      .sort((a,b)=>Number(b.saturation_rate)-Number(a.saturation_rate)).slice(0, 20);
    return { pen, sat };
  }, [tension]);

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
          href: `/monitoring/network/dynamics?station_id=${encodeURIComponent(id)}`,
        };
      });
    return { pen: toView(topTension.pen), sat: toView(topTension.sat) };
  }, [topTension, stationsIdx]);

  return (
    <div className="monitoring">
      <Head>
        <title>Monitoring — Network / Overview</title>
        <meta name="description" content="KPIs réseau, snapshot, courbes et carte." />
        <link rel="stylesheet" href="/css/monitoring.css" />
        {/* Leaflet CSS + petits correctifs globaux */}
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

      <main className="page">
        <MonitoringNav
          title="Network — Overview"
          subtitle="KPIs, snapshot distribution, daily curves & map"
          generatedAt={generatedAt}
          crumbs={[
            { label: "Accueil", href: "/" },
            { label: "Monitoring", href: "/monitoring" },
            { label: "App", href: "/app" },
          ]}
          extraActions={[
            { label: "Stations", href: "/monitoring/network/stations" },
            { label: "Dynamics", href: "/monitoring/network/dynamics" },
          ]}
        />

        {/* Loading / Error */}
        {loading && <div className="banner">Loading…</div>}
        {error && <div className="banner banner--error">{error}</div>}

        {/* KPIs Snapshot */}
        <section className="mt-4">
          <h2>Résumé — snapshot</h2>
          <div className="kpi-grid kpi-grid--5">
            <Kpi label="Stations actives" value={kpis?.stations_active} fmt={fmtInt} />
            <Kpi label="Hors ligne" value={kpis?.stations_offline} fmt={fmtInt} />
            <Kpi label="≥1 vélo" value={kpis?.availability_bike_pct} fmt={(v)=>fmtPct(v,1)} />
            <Kpi label="≥1 place" value={kpis?.availability_dock_pct} fmt={(v)=>fmtPct(v,1)} />
            <Kpi label="Couverture (récente)" value={Number(kpis?.coverage_pct ?? NaN)} fmt={(v)=>fmtPct(v,1)} />
          </div>
          <div className="small mt-2">
            Window: {kpis?.last_days ?? "—"} days · Ref: {kpis?.ref_days ?? "—"} days · Schema v{kpis?.schema_version ?? "—"}
            <span style={{ marginLeft: 8, opacity: 0.8 }}>({kpis?.snapshot_ts_local ?? "—"})</span>
          </div>
        </section>

        {/* === Carte snapshot — même intégration que "stations" === */}
        <section className="mt-6">
          <h2>Stations map — Snapshot</h2>

          <div className="map-block">
            <div className="map-wrap" style={{ width: "100%", height: 520 }}>
              {snapMap?.rows?.length ? (
                <SnapshotMap rows={snapMap.rows} />
              ) : (
                <div className="empty">No snapshot map data.</div>
              )}
            </div>
          </div>

          <div className="small mt-2">
            Basemap: Carto Light (no labels) · Red=pénurie · Blue=saturation · Green=OK (size ~ √bikes).
          </div>
        </section>

        {/* === Distribution snapshot — séparée === */}
        <section className="mt-6">
          <h2>Snapshot — distribution</h2>
          <div className="card plot-card">
            <h3 style={{ margin: "6px 0 10px 0", fontSize: 16 }}>Distribution (%)</h3>
            {distBars.length ? (
              <Plot
                data={distBars as Plotly.Data[]}
                layout={{
                  autosize: true,
                  height: 280,
                  margin: { l: 48, r: 10, t: 10, b: 36 },
                  yaxis: { title: { text: "%" }, range: [0, 100] },
                  xaxis: { title: { text: "State" } },
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                }}
                config={{ displayModeBar: false, responsive: true }}
                className="plot plot--sm"
              />
            ) : (
              <div className="empty">No snapshot distribution.</div>
            )}
          </div>
        </section>

        {/* Courbes du jour vs référence */}
        <section className="mt-6">
          <h2>Hier (UTC) vs Référence</h2>
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
                className="plot plot--lg"
              />
              <div className="small mt-2">
                Les courbes sont en UTC. « Hier (UTC) » correspond au jour UTC complet (00:00–23:55).
              </div>
            </>
          ) : (
            <div className="empty">Curves unavailable.</div>
          )}
        </section>

        {/* KPIs Today vs J-7/J-14/J-21 */}
        <section className="mt-6">
          <h2>Aujourd’hui vs J-7 / J-14 / J-21</h2>
          {lags ? (
            <div className="grid-4">
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
            <div className="empty">Comparisons unavailable.</div>
          )}
        </section>

        {/* Stations en tension */}
        <section className="mt-6">
          <h2>Stations en tension (fenêtre récente)</h2>
          <div className="grid-2">
            <div className="card">
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
                <div className="empty">—</div>
              )}
            </div>

            <div className="card">
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
                <div className="empty">—</div>
              )}
            </div>
          </div>

          {tension && (
            <div className="small mt-2">
              Schema v{tension.schema_version} — generated {tension.generated_at}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

/* ───────────────────────── UI atoms (compat monitoring.css) ───────────────────────── */
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
    <div className="kpi">
      <div className="kpi__label">{label}</div>
      <div className="kpi__value">{text}</div>
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
    <div className="card">
      <h3 style={{ margin: "6px 0 10px 0", fontSize: 16 }}>{title}</h3>
      <div className="grid-4">
        {values.map((x) => (
          <div key={x.label} style={{ textAlign: "center" }}>
            <div className="small" style={{ opacity: 0.75 }}>{x.label}</div>
            <div style={{ fontSize: 18, fontWeight: 700 }}>{fmtPct(x.v ?? undefined, 1)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}