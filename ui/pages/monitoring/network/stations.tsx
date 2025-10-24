// ui/pages/monitoring/network/stations.tsx
import Head from "next/head";
import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import type * as Plotly from "plotly.js";
import MonitoringNav from "@/components/monitoring/MonitoringNav";
import LoadingBar, { type LoadingBarStatus } from "@/components/common/LoadingBar";
import KpiBar from "@/components/monitoring/KpiBar";
import { chartConfig, chartLayout } from "@/lib/plotlyTheme";

import {
  getStationsKpis,
  getStationsCentroids,
  getStationsStats7,
  fetchStationsIndex,
  type KpisDoc,
  type CentroidsDoc,
  type Stats7Doc,
  type StationMeta,
} from "@/lib/services/monitoring/network_stations";

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
function hist(
  x: number[],
  name: string,
  nbins = 24
): Partial<Plotly.PlotData> & { nbinsx?: number } {
  return { x, type: "histogram" as const, name, nbinsx: nbins, opacity: 0.9, hovertemplate: "%{x:.1f}<extra></extra>" };
}
function boxY(
  y: number[],
  name: string
): Partial<Plotly.PlotData> & {
  boxpoints?: false | "all" | "outliers" | "suspectedoutliers";
} {
  return { y, type: "box" as const, name, boxpoints: false, hovertemplate: "%{y:.2f}<extra></extra>" };
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

  const palette = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
  ];

  function colorOfCluster(c: number | null | undefined, uniq: number[]) {
    if (c === -1) return "#9e9e9e";
    const idx = uniq.indexOf(Number(c));
    return idx >= 0 ? palette[idx % palette.length] : "#4c78a8";
  }

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

    const latMed = valid.length
      ? valid.map((r) => r.lat).sort((a, b) => a - b)[Math.floor(valid.length / 2)]
      : 48.8566;
    const lonMed = valid.length
      ? valid.map((r) => r.lon).sort((a, b) => a - b)[Math.floor(valid.length / 2)]
      : 2.3522;

    const uniq = Array.from(new Set(valid.map((r) => r.cluster).filter((v) => v != null))) as number[];

    const [tileUrl, setTileUrl] = useState(
      "https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png"
    );
    useEffect(() => {
      const img = new Image();
      img.onerror = () => setTileUrl("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png");
      img.src = "https://a.basemaps.cartocdn.com/light_nolabels/3/4/2.png";
    }, []);

    return (
      <div className="map-wrap" style={{ position: "relative", width: "100%", height: 520 }}>
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
              ? Number.isFinite(cap) && cap > 0
                ? Math.max(2, Math.min(9, Math.sqrt(cap) / 2.4))
                : 3
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
            zIndex: 1000,
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

/* ───────────────── Page ───────────────── */
export default function NetworkStationsPage() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [kpis, setKpis] = useState<KpisDoc | null>(null);
  const [centroids, setCentroids] = useState<CentroidsDoc | null>(null);
  const [stats7Doc, setStats7Doc] = useState<Stats7Doc | null>(null);
  const [clusterRows, setClusterRows] = useState<ClusterRow[]>([]);
  const [clusterFilter, setClusterFilter] = useState<number | null>(null);
  const [query, setQuery] = useState<string>("");
  const [sizeByCapacity, setSizeByCapacity] = useState<boolean>(true);
  const [autoFit, setAutoFit] = useState<boolean>(true);

  // ✅ LoadingBar status (aligned with overview.tsx)
  const barStatus: LoadingBarStatus = loading ? "loading" : error ? "error" : "success";

  /* // CSV export (désactivé pour le moment)
  function exportCSV() {
    const rows = filteredRows;
    const header = ["station_id", "name", "lat", "lon", "capacity_est", "cluster"];
    const lines = [
      header.join(","),
      ...rows.map((r) =>
        header.map((k) => {
          const v = (r as any)[k];
          if (v == null) return "";
          const s = String(v).replace(/"/g, '""');
          return /[",\n]/.test(s) ? `"${s}"` : s;
        }).join(",")
      ),
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "network_stations_clusters.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  }
  */

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setLoading(true);

        // Fetch all in parallel
        const [rKpis, rCent, rStats] = await Promise.allSettled([
          getStationsKpis(),
          getStationsCentroids(),
          getStationsStats7(),
        ]);
        if (!alive) return;

        const k = ok(rKpis);
        const c = ok(rCent);
        const s7 = ok(rStats);
        setKpis(k);
        setCentroids(c);
        setStats7Doc(s7);

        // Error on partial failure
        const failures = [rKpis, rCent, rStats].filter(
          (r): r is PromiseRejectedResult => r.status === "rejected"
        );
        setError(failures.length ? failures.map((f) => String((f.reason && (f.reason.message ?? f.reason)) || "request failed")).join(" | ") : null);

        // Build clusterRows from stats7
        if (s7 && Array.isArray(s7.rows)) {
          let rows = s7.rows.map((x: any) => ({
            station_id: String(x.station_id),
            name: x.name ?? undefined,
            lat: Number.isFinite(Number(x.lat)) ? Number(x.lat) : NaN,
            lon: Number.isFinite(Number(x.lon)) ? Number(x.lon) : NaN,
            capacity_est: Number.isFinite(Number(x.capacity_est)) ? Number(x.capacity_est) : NaN,
            cluster: x.cluster != null ? Number(x.cluster) : null,
          })) as ClusterRow[];

          // Backfill if missing
          let haveCoords = rows.some((r) => Number.isFinite(r.lat) && Number.isFinite(r.lon));
          if (!haveCoords) {
            const idx = await fetchStationsIndex().catch(() => ({} as Record<string, StationMeta>));
            rows = rows.map((r) => {
              if (!Number.isFinite(r.lat) || !Number.isFinite(r.lon)) {
                const m = idx[r.station_id];
                if (m) {
                  return {
                    ...r,
                    name: r.name ?? (m.name ?? undefined),
                    lat: Number.isFinite(Number(m.lat)) ? Number(m.lat) : (r.lat as any),
                    lon: Number.isFinite(Number(m.lon)) ? Number(m.lon) : (r.lon as any),
                    capacity_est: Number.isFinite(Number(r.capacity_est))
                      ? r.capacity_est
                      : Number.isFinite(Number((m as any).capacity))
                      ? Number((m as any).capacity)
                      : undefined,
                  };
                }
              }
              return r;
            });
          }
          setClusterRows(rows.filter((r) => Number.isFinite(r.lat) && Number.isFinite(r.lon)));
        }
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
    if (clusterFilter != null) arr = arr.filter((r) => r.cluster === clusterFilter);
    if (query.trim()) {
      const q = query.trim().toLowerCase();
      arr = arr.filter(
        (r) =>
          (r.name ?? "").toLowerCase().includes(q) ||
          String(r.station_id).toLowerCase().includes(q)
      );
    }
    return arr;
  }, [clusterRows, clusterFilter, query]);

  const xLabels =
    (centroids?.x_labels as string[] | undefined) ??
    Array.from({ length: 24 }, (_, h) => `${String(h).padStart(2, "0")}:00`);
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
        hovertemplate: "%{x} — %{y:.2f}<extra>Cluster " + c.cluster + "</extra>",
      }));
  }, [centroids, xLabels]);

  const rows = stats7Doc?.rows ?? [];
  const coverageAll = rows
    .map((r) => parseRatioLoose(r["coverage_pct"]))
    .filter(isFiniteNum)
    .map((v) => v * 100);
  const volatilityAll = rows.map((r) => parseNumberLoose(r["volatility"])).filter(isFiniteNum);
  const penuryPct = rows.map((r) => parseRatioLoose(r["penury_rate"]) * 100).filter(isFiniteNum);
  const saturationPct = rows.map((r) => parseRatioLoose(r["saturation_rate"]) * 100).filter(isFiniteNum);

  return (
    <div className="monitoring">
      <Head>
        <title>Monitoring — Network / Stations</title>
        <meta name="description" content="Clusters map + 24h profiles (centroids) and recent distributions." />
      </Head>

      {/* Main content (header/footer injected by _app.tsx) */}
      <main className="page" style={{ paddingTop: "calc(var(--header-h, 70px) + 12px)" }}>
        <MonitoringNav
          title="Network — Stations"
          subtitle="Clusters, map and distributions"
          generatedAt={generatedAt}
          extraActions={[
            { label: "Overview", href: "/monitoring/network/overview" },
            { label: "Dynamics", href: "/monitoring/network/dynamics" },
          ]}
        />

        <LoadingBar status={barStatus} />

        {/* ───────────────── KPIs (KpiBar) ───────────────── */}
        <section className="mt-4">
          <h2>Clustering summary</h2>

          <KpiBar
            dense
            items={[
              { label: "Stations", value: Number.isFinite(Number(kpis?.n_stations)) ? String(kpis?.n_stations) : "—" },
              { label: "Clusters (k)", value: Number.isFinite(Number(kpis?.k_effective)) ? String(kpis?.k_effective) : "—" },
              { label: "Silhouette", value: fmt3(kpis?.silhouette) },
              { label: "Davies–Bouldin", value: fmt3(kpis?.davies_bouldin) },
            ]}
          />

          <div className="kpi-bar-meta">
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
                  <option key={String(c)} value={String(c)}>Cluster {String(c)}</option>
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
            {/* <button className="btn btn--primary" onClick={exportCSV}>Export CSV</button> */}
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
          <div className="figure-note small">
            Basemap: Carto Light (no labels). Color encodes the cluster; point size ≈ estimated capacity.
          </div>
        </section>

        {/* Centroids */}
        <section className="mt-6">
          <h2>24h profiles — Centroids</h2>
          <div className="card plot-card">
            <h3>Mean 24-hour occupancy profile by cluster (ratio)</h3>
            {centroidTraces.length ? (
              <>
                <Plot
                  data={centroidTraces as Plotly.Data[]}
                  layout={chartLayout({
                    height: 380,
                    xaxis: { title: { text: "Hour (HH:MM)" } },
                    yaxis: { title: { text: "Occupancy (ratio)" }, rangemode: "tozero" },
                  })}
                  config={chartConfig}
                  className="plot plot--lg"
                />
              </>
            ) : (
              <div className="empty">Centroids unavailable.</div>
            )}
          </div>
          <div className="figure-note small">
            Each series is the cluster centroid over 24h; X-axis labels come from the provided `x_labels` or default to HH:00.
          </div>
        </section>

        {/* Distributions */}
        <section className="mt-6">
          <h2>Recent indicators — distributions</h2>
          <div className="grid-2">
            <div className="card plot-card">
              <h3>Coverage over last 7 days — share of time (%)</h3>
              {coverageAll.length ? (
                <Plot
                  data={[hist(coverageAll, "coverage", 24)] as Plotly.Data[]}
                  layout={chartLayout({
                    height: 280,
                    xaxis: { title: { text: "Coverage (%)" }, range: [0, 100] },
                    yaxis: { title: { text: "Stations (count)" } },
                  })}
                  config={chartConfig}
                  className="plot plot--sm"
                />
              ) : (
                <div className="empty">No coverage data.</div>
              )}
            </div>

            <div className="card plot-card">
              <h3>Bike count volatility σ — last 7 days</h3>
              {volatilityAll.length ? (
                <Plot
                  data={[boxY(volatilityAll, "σ(bikes)")] as Plotly.Data[]}
                  layout={chartLayout({
                    height: 280,
                    yaxis: { title: { text: "σ bikes / station (7d)" }, rangemode: "tozero" },
                  })}
                  config={chartConfig}
                  className="plot plot--sm"
                />
              ) : (
                <div className="empty">No volatility data.</div>
              )}
            </div>

            <div className="card plot-card">
              <h3>Penury — share of time over last 7 days (%)</h3>
              {penuryPct.length ? (
                <Plot
                  data={[hist(penuryPct, "% penury", 24)] as Plotly.Data[]}
                  layout={chartLayout({
                    height: 280,
                    xaxis: { title: { text: "Percentage of time (%)" }, range: [0, 100] },
                    yaxis: { title: { text: "Stations (count)" } },
                  })}
                  config={chartConfig}
                  className="plot plot--sm"
                />
              ) : (
                <div className="empty">No penury data.</div>
              )}
            </div>

            <div className="card plot-card">
              <h3>Saturation — share of time over last 7 days (%)</h3>
              {saturationPct.length ? (
                <Plot
                  data={[hist(saturationPct, "% saturation", 24)] as Plotly.Data[]}
                  layout={chartLayout({
                    height: 280,
                    xaxis: { title: { text: "Percentage of time (%)" }, range: [0, 100] },
                    yaxis: { title: { text: "Stations (count)" } },
                  })}
                  config={chartConfig}
                  className="plot plot--sm"
                />
              ) : (
                <div className="empty">No saturation data.</div>
              )}
            </div>
          </div>

          {stats7Doc && (
            <div className="figure-note small">
              Schema v{stats7Doc.schema_version} — generated {stats7Doc.generated_at}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
