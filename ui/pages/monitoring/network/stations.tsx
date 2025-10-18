// ui/pages/monitoring/network/stations.tsx
import Head from "next/head";
import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import type * as Plotly from "plotly.js";
import MonitoringNav from "@/components/monitoring/MonitoringNav";

/* ───────────────── HTTP helper ───────────────── */
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

/* ───────────────── Plotly (client only) ───────────────── */
const Plot = dynamic(() => import("react-plotly.js").then((m) => m.default), {
  ssr: false,
  loading: () => (
    <div style={{ height: 320, display: "grid", placeItems: "center", opacity: 0.7 }}>
      Loading chart…
    </div>
  ),
});

/* ───────────────── Helpers ───────────────── */
function ok<T>(r: PromiseSettledResult<T>): T | null {
  return r.status === "fulfilled" ? r.value : null;
}
function fmt3(x?: number | null) {
  const v = Number(x);
  return Number.isFinite(v) ? v.toFixed(3) : "—";
}
function parseRatioLoose(v: unknown): number {
  if (v == null) return NaN;
  if (typeof v === "number") return v;
  let s = String(v).trim();
  const isPercent = s.endsWith("%");
  s = s.replace("%", "").replace(/\s/g, "").replace(",", ".");
  const n = Number(s);
  return Number.isFinite(n) ? (isPercent ? n / 100 : n) : NaN;
}
function parseNumberLoose(v: unknown): number {
  if (v == null) return NaN;
  if (typeof v === "number") return v;
  const n = Number(String(v).trim().replace(/\s/g, "").replace(",", "."));
  return Number.isFinite(n) ? n : NaN;
}
const isFiniteNum = (x: number) => Number.isFinite(x);
function hist(x: number[], name: string, nbins = 24): Partial<Plotly.PlotData> & { nbinsx?: number } {
  return { x, type: "histogram" as const, name, nbinsx: nbins, opacity: 0.9 };
}
function boxY(
  y: number[],
  name: string
): Partial<Plotly.PlotData> & { boxpoints?: false | "all" | "outliers" | "suspectedoutliers" } {
  return { y, type: "box" as const, name, boxpoints: false };
}

/* ───────────────── Fallback stations ───────────────── */
type StationMeta = {
  station_id: string;
  name?: string | null;
  lat?: number | null;
  lon?: number | null;
  capacity?: number | null;
};
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
      capacity: Number.isFinite(Number((s as any).capacity)) ? Number((s as any).capacity) : null,
    };
  }
  return idx;
}

/* ───────────────── Mini MapView (Leaflet) ───────────────── */
type ClusterRow = {
  station_id: string;
  name?: string;
  lat: number;
  lon: number;
  capacity_est?: number;
  cluster?: number | null;
};

const MapView = dynamic(async () => {
  const RL = await import("react-leaflet");
  const { MapContainer, TileLayer, CircleMarker, Tooltip, useMap } = RL as any;
  const { useEffect, useMemo, useState } = await import("react");

  const palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"];

  function colorOfCluster(c: number | null | undefined, uniq: number[]) {
    if (c === -1) return "#9e9e9e";
    const idx = uniq.indexOf(Number(c));
    return idx >= 0 ? palette[idx % palette.length] : "#4c78a8";
  }

  // ✅ version corrigée sans window.L
  function FitBounds({ rows }: { rows: ClusterRow[] }) {
    const map = useMap();
    useEffect(() => {
      if (!rows.length) return;
      let minLat = 90, maxLat = -90, minLon = 180, maxLon = -180;
      for (const r of rows) {
        const la = Number(r.lat), lo = Number(r.lon);
        if (!Number.isFinite(la) || !Number.isFinite(lo)) continue;
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

  function MapViewInner({
    rows,
    sizeByCapacity,
    autoFit,
  }: {
    rows: ClusterRow[];
    sizeByCapacity: boolean;
    autoFit: boolean;
  }) {
    const valid = useMemo(
      () => rows.filter((r) => Number.isFinite(Number(r.lat)) && Number.isFinite(Number(r.lon))),
      [rows]
    );

    const latMed = valid.length ? valid.map((r) => r.lat).sort((a,b)=>a-b)[Math.floor(valid.length/2)] : 48.8566;
    const lonMed = valid.length ? valid.map((r) => r.lon).sort((a,b)=>a-b)[Math.floor(valid.length/2)] : 2.3522;

    const uniq = Array.from(new Set(valid.map((r) => r.cluster).filter((v) => v != null))) as number[];

    const [tileUrl, setTileUrl] = useState("https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png");
    useEffect(() => {
      const img = new Image();
      img.onerror = () => setTileUrl("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png");
      img.src = "https://a.basemaps.cartocdn.com/light_nolabels/3/4/2.png";
    }, []);

    return (
        <div
          className="map-wrap"
          style={{ position: "relative", width: "100%", height: 520 }}
        >
          <MapContainer
          center={[latMed, lonMed]}
          zoom={12}
          className="leaflet-container"
          style={{ width: "100%", height: "100%" }}
        >
          <TileLayer
            url={tileUrl}
            attribution='&copy; OpenStreetMap, &copy; <a href="https://carto.com/">CARTO</a>'
            detectRetina
          />
          {autoFit && !!valid.length && <FitBounds rows={valid} />}

          {valid.map((r) => {
            const cap = Number(r.capacity_est);
            const rad = sizeByCapacity
              ? (Number.isFinite(cap) && cap > 0 ? Math.max(2, Math.min(9, Math.sqrt(cap) / 2.4)) : 3)
              : 4;
            const col = colorOfCluster(r.cluster ?? null, uniq);
            return (
              <CircleMarker
                key={r.station_id}
                center={[r.lat, r.lon]}
                radius={rad}
                pathOptions={{ color: col, weight: 0.8, fillColor: col, fillOpacity: 0.8 }}
              >
                <Tooltip>
                  <div style={{ display: "grid", gap: 4 }}>
                    <div><b>{r.name ?? r.station_id}</b></div>
                    {Number.isFinite(cap) && <div>cap≈{cap | 0}</div>}
                    <div>cluster: {r.cluster ?? "—"}</div>
                    <a
                      href={`/monitoring/network/dynamics?station_id=${encodeURIComponent(r.station_id)}`}
                      style={{ textDecoration: "underline" }}
                    >
                      View dynamics →
                    </a>
                  </div>
                </Tooltip>
              </CircleMarker>
            );
          })}
        </MapContainer>

        <div
          style={{
            position: "absolute",
            right: 8,
            bottom: 8,
            zIndex: 1000,               // au-dessus de la carte
            background: "rgba(255,255,255,0.92)",
            color: "#111",
            borderRadius: 10,
            padding: "8px 10px",
            fontSize: 12,
            boxShadow: "0 2px 10px rgba(0,0,0,0.15)",
            border: "1px solid #0001",
            pointerEvents: "auto",
          }}
        >
          <div style={{ fontWeight: 700, marginBottom: 6 }}>Clusters</div>
          {uniq.map((c) => (
            <div key={String(c)} style={{ display: "flex", alignItems: "center", gap: 6, margin: "2px 0" }}>
              <span
                style={{
                  width: 12,
                  height: 12,
                  borderRadius: 999,
                  background: colorOfCluster(c, uniq),
                  border: "1px solid #0002",
                  display: "inline-block",
                  flex: "0 0 auto",
                }}
              />
              <span>Cluster {c}</span>
            </div>
          ))}
          <div style={{ opacity: 0.7, marginTop: 6 }}>{valid.length} stations</div>
        </div>

      </div>
    );
  }

  return MapViewInner;
}, { ssr: false });

/* ───────────────── Types ───────────────── */
export type KpisDoc = {
  schema_version: string;
  generated_at: string;
  n_stations: number | null;
  k_effective: number | null;
  silhouette: number | null;
  davies_bouldin: number | null;
  window_days: number;
};
export type CentroidsDoc = {
  schema_version: string;
  generated_at: string;
  x_labels: string[];
  centroids: Array<{ cluster: number; y: (number | null)[] }>;
};
export type Stats7Doc = {
  schema_version: string;
  generated_at: string;
  rows: Array<{
    station_id: string;
    name?: string | null;
    lat?: number | null;
    lon?: number | null;
    capacity_est?: number | null;
    volatility?: number | null;
    penury_rate?: number | null;
    saturation_rate?: number | null;
    coverage_pct?: number | null;
    cluster?: number | null;
  }>;
};

/* ───────────────── Page ───────────────── */
export default function NetworkStationsPage() {
  const [kpis, setKpis] = useState<KpisDoc | null>(null);
  const [centroids, setCentroids] = useState<CentroidsDoc | null>(null);
  const [stats7Doc, setStats7Doc] = useState<Stats7Doc | null>(null);
  const [clusterRows, setClusterRows] = useState<ClusterRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [clusterFilter, setClusterFilter] = useState<number | null>(null);
  const [query, setQuery] = useState<string>("");
  const [sizeByCapacity, setSizeByCapacity] = useState<boolean>(true);
  const [autoFit, setAutoFit] = useState<boolean>(true);

  function exportCSV() {
    const rows = filteredRows;
    const header = ["station_id","name","lat","lon","capacity_est","cluster"];
    const lines = [
      header.join(","),
      ...rows.map(r => header.map(k => {
        const v = (r as any)[k];
        if (v == null) return "";
        const s = String(v).replace(/"/g, '""');
        return /[",\n]/.test(s) ? `"${s}"` : s;
      }).join(","))
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "network_stations_clusters.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setLoading(true);
        const [rKpis, rCent, rStats] = await Promise.allSettled([
          getJSON<KpisDoc>("/monitoring/network/stations/kpis"),
          getJSON<CentroidsDoc>("/monitoring/network/stations/centroids"),
          getJSON<Stats7Doc>("/monitoring/network/stations/stats7"),
        ]);
        if (!alive) return;

        const k = ok(rKpis);
        const c = ok(rCent);
        const s7 = ok(rStats);
        setKpis(k);
        setCentroids(c);
        setStats7Doc(s7);

        if (s7 && Array.isArray(s7.rows)) {
          let rows = s7.rows.map((x: any) => ({
            station_id: String(x.station_id),
            name: x.name ?? null,
            lat: Number.isFinite(Number(x.lat)) ? Number(x.lat) : NaN,
            lon: Number.isFinite(Number(x.lon)) ? Number(x.lon) : NaN,
            capacity_est: Number.isFinite(Number(x.capacity_est)) ? Number(x.capacity_est) : NaN,
            cluster: x.cluster != null ? Number(x.cluster) : null,
          })) as ClusterRow[];

          let haveCoords = rows.some(r => Number.isFinite(r.lat) && Number.isFinite(r.lon));
          if (!haveCoords) {
            const idx = await fetchStationsIndex().catch(() => ({} as Record<string, StationMeta>));
            rows = rows.map(r => {
              if (!Number.isFinite(r.lat) || !Number.isFinite(r.lon)) {
                const m = idx[r.station_id];
                if (m) {
                  return {
                    ...r,
                    name: r.name ?? (m.name ?? undefined),
                    lat: Number.isFinite(Number(m.lat)) ? Number(m.lat) : (r.lat as any),
                    lon: Number.isFinite(Number(m.lon)) ? Number(m.lon) : (r.lon as any),
                    capacity_est: Number.isFinite(Number(r.capacity_est)) ? r.capacity_est
                              : (Number.isFinite(Number(m.capacity)) ? Number(m.capacity) : undefined),
                  };
                }
              }
              return r;
            });
          }
          setClusterRows(rows.filter((r) => Number.isFinite(r.lat) && Number.isFinite(r.lon)));
        }

        setError(null);
      } catch (e: any) {
        if (alive) setError(String(e?.message ?? e));
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, []);

  const generatedAt = kpis?.generated_at ?? centroids?.generated_at ?? stats7Doc?.generated_at;
  const filteredRows = useMemo(() => {
    let arr = clusterRows;
    if (clusterFilter != null) arr = arr.filter(r => r.cluster === clusterFilter);
    if (query.trim()) {
      const q = query.trim().toLowerCase();
      arr = arr.filter(r =>
        (r.name ?? "").toLowerCase().includes(q) || String(r.station_id).toLowerCase().includes(q)
      );
    }
    return arr;
  }, [clusterRows, clusterFilter, query]);

  const xLabels = centroids?.x_labels ?? Array.from({ length: 24 }, (_, h) => `${String(h).padStart(2, "0")}:00`);
  const centroidTraces: Partial<Plotly.PlotData>[] = useMemo(() => {
    if (!centroids?.centroids) return [];
    return centroids.centroids
      .sort((a, b) => a.cluster - b.cluster)
      .map((c) => ({
        x: xLabels,
        y: (c.y ?? []).map((v) => (Number.isFinite(Number(v)) ? Number(v) : null)),
        type: "scatter" as const,
        mode: "lines" as const,
        name: `Cluster ${c.cluster}`,
        connectgaps: false,
      }));
  }, [centroids, xLabels]);

  const rows = stats7Doc?.rows ?? [];
  const coverageAll = rows.map(r => parseRatioLoose(r["coverage_pct"])).filter(isFiniteNum).map(v => v * 100);
  const volatilityAll = rows.map(r => parseNumberLoose(r["volatility"])).filter(isFiniteNum);
  const penuryPct = rows.map(r => parseRatioLoose(r["penury_rate"]) * 100).filter(isFiniteNum);
  const saturationPct = rows.map(r => parseRatioLoose(r["saturation_rate"]) * 100).filter(isFiniteNum);

  return (
    <div className="monitoring">
      <Head>
        <title>Monitoring — Network / Stations</title>
        <meta name="description" content="Clusters map + 24h profiles (centroids) and recent distributions." />
        <link rel="stylesheet" href="/css/monitoring.css" />
        {/* Leaflet CSS + small global fixes */}
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
          title="Network — Stations"
          subtitle="Clusters, map and distributions"
          generatedAt={generatedAt}
          crumbs={[
            { label: "Accueil", href: "/" },
            { label: "Monitoring", href: "/monitoring" },
            { label: "App", href: "/app" },
          ]}
          extraActions={[
            { label: "Stations", href: "/monitoring/network/stations" },
            { label: "Performance", href: "/monitoring/model/performance" },
          ]}
        />

        {loading && <div className="banner">Loading…</div>}
        {error && <div className="banner banner--error">{error}</div>}

        {/* KPIs */}
        <section className="mt-4">
          <h2>Clustering summary</h2>
          <div className="kpi-grid">
            <div className="kpi">
              <div className="kpi__label">Stations</div>
              <div className="kpi__value">{Number.isFinite(Number(kpis?.n_stations)) ? kpis?.n_stations : "—"}</div>
            </div>
            <div className="kpi">
              <div className="kpi__label">Clusters (k)</div>
              <div className="kpi__value">{Number.isFinite(Number(kpis?.k_effective)) ? kpis?.k_effective : "—"}</div>
            </div>
            <div className="kpi">
              <div className="kpi__label">Silhouette</div>
              <div className="kpi__value">{fmt3(kpis?.silhouette)}</div>
            </div>
            <div className="kpi">
              <div className="kpi__label">Davies–Bouldin</div>
              <div className="kpi__value">{fmt3(kpis?.davies_bouldin)}</div>
            </div>
          </div>
          <div className="small mt-2">
            Window: {kpis?.window_days ?? "—"} days · Schema v{kpis?.schema_version ?? "—"}
          </div>
        </section>

        {/* Map */}
        <section className="mt-6">
          <h2>Stations map — Clusters</h2>
          <div className="filters">
            <input
              className="input"
              placeholder="Search a station…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <select
              className="select"
              value={clusterFilter ?? ""}
              onChange={(e) => setClusterFilter(e.target.value === "" ? null : Number(e.target.value))}
            >
              <option value="">All clusters</option>
              {Array.from(new Set(clusterRows.map((r) => r.cluster).filter((v) => v != null)))
                .sort((a: any, b: any) => Number(a) - Number(b))
                .map((c: any) => (
                  <option key={String(c)} value={String(c)}>
                    Cluster {String(c)}
                  </option>
                ))}
            </select>
            <label className="check">
              <input type="checkbox" checked={sizeByCapacity} onChange={(e) => setSizeByCapacity(e.target.checked)} />
              Size = capacity
            </label>
            <label className="check">
              <input type="checkbox" checked={autoFit} onChange={(e) => setAutoFit(e.target.checked)} />
              Auto-fit
            </label>
            <button className="btn btn-primary" onClick={exportCSV}>
              Export CSV
            </button>
          </div>

          <div className="map-block">
            <div className="map-wrap" style={{ width: "100%", height: 520 }}>
              {filteredRows.length ? (
                <MapView rows={filteredRows} sizeByCapacity={sizeByCapacity} autoFit={autoFit} />
              ) : (
                <div className="empty">No cluster/coordinate data available.</div>
              )}
            </div>
          </div>
          <div className="small mt-2">
            Basemap: Carto Light (no labels) · Color = cluster · Size ≈ estimated capacity.
          </div>
        </section>

        {/* Centroids */}
        <section className="mt-6">
          <h2>24h profiles — Centroids</h2>
          {centroidTraces.length ? (
            <Plot
              data={centroidTraces as Plotly.Data[]}
              layout={{
                autosize: true,
                height: 380,
                margin: { l: 52, r: 10, t: 30, b: 40 },
                xaxis: { title: { text: "Hour" } },
                yaxis: { title: { text: "Occupancy (ratio)" }, rangemode: "tozero" },
                legend: { orientation: "h" },
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                hovermode: "x unified",
              }}
              config={{ displayModeBar: false, responsive: true }}
              className="plot plot--lg"
            />
          ) : (
            <div className="empty">Centroids unavailable.</div>
          )}
        </section>

        {/* Distributions */}
        <section className="mt-6">
          <h2>Recent indicators — distributions</h2>
          <div className="grid-2">
            <div className="card plot-card">
              <h3 style={{ margin: "6px 0 10px 0", fontSize: 16 }}>Coverage (7 days)</h3>
              {coverageAll.length ? (
                <Plot
                  data={[hist(coverageAll, "coverage", 24)] as Plotly.Data[]}
                  layout={{
                    autosize: true,
                    height: 280,
                    margin: { l: 48, r: 10, t: 10, b: 36 },
                    xaxis: { title: { text: "Coverage (%)" }, range: [0, 100] },
                    yaxis: { title: { text: "Stations" } },
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  className="plot plot--sm"
                />
              ) : (
                <div className="empty">No coverage data.</div>
              )}
            </div>

            <div className="card plot-card">
              <h3 style={{ margin: "6px 0 10px 0", fontSize: 16 }}>Bike volatility (σ, 7 days)</h3>
              {volatilityAll.length ? (
                <Plot
                  data={[boxY(volatilityAll, "σ(bikes)")] as Plotly.Data[]}
                  layout={{
                    autosize: true,
                    height: 280,
                    margin: { l: 48, r: 10, t: 10, b: 36 },
                    yaxis: { title: { text: "σ bikes / station (7d)" }, rangemode: "tozero" },
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  className="plot plot--sm"
                />
              ) : (
                <div className="empty">No volatility data.</div>
              )}
            </div>

            <div className="card plot-card">
              <h3 style={{ margin: "6px 0 10px 0", fontSize: 16 }}>% penury (7 days)</h3>
              {penuryPct.length ? (
                <Plot
                  data={[hist(penuryPct, "% penury", 24)] as Plotly.Data[]}
                  layout={{
                    autosize: true,
                    height: 280,
                    margin: { l: 48, r: 10, t: 10, b: 36 },
                    xaxis: { title: { text: "Percentage of time (%)" }, range: [0, 100] },
                    yaxis: { title: { text: "Stations" } },
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  className="plot plot--sm"
                />
              ) : (
                <div className="empty">No penury data.</div>
              )}
            </div>

            <div className="card plot-card">
              <h3 style={{ margin: "6px 0 10px 0", fontSize: 16 }}>% saturation (7 days)</h3>
            {saturationPct.length ? (
                <Plot
                  data={[hist(saturationPct, "% saturation", 24)] as Plotly.Data[]}
                  layout={{
                    autosize: true,
                    height: 280,
                    margin: { l: 48, r: 10, t: 10, b: 36 },
                    xaxis: { title: { text: "Percentage of time (%)" }, range: [0, 100] },
                    yaxis: { title: { text: "Stations" } },
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  className="plot plot--sm"
                />
              ) : (
                <div className="empty">No saturation data.</div>
              )}
            </div>
          </div>

          {stats7Doc && (
            <div className="small mt-2">
              Schema v{stats7Doc.schema_version} — generated {stats7Doc.generated_at}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}