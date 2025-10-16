// ui/pages/monitoring/network/stations.tsx
import Head from "next/head";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import type * as Plotly from "plotly.js";

// Plotly (client only)
export const Plot = dynamic(() => import("react-plotly.js").then((m) => m.default), {
  ssr: false,
  loading: () => (
    <div style={{ height: 320, display: "grid", placeItems: "center", opacity: 0.7 }}>
      Chargement du graphique…
    </div>
  ),
});

/* ───────────────────────── Helpers communs (hoisted) ───────────────────────── */

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

function fmt3(x?: number | null) {
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  return v.toFixed(3);
}

// Parsing robuste pour stats
function parseRatioLoose(v: unknown): number {
  if (v == null) return NaN;
  if (typeof v === "number") return v;
  let s = String(v).trim();
  const isPercent = s.endsWith("%");
  s = s.replace("%", "").replace(/\s/g, "").replace(",", ".");
  const n = Number(s);
  if (!Number.isFinite(n)) return NaN;
  return isPercent ? n / 100 : n;
}
function parseNumberLoose(v: unknown): number {
  if (v == null) return NaN;
  if (typeof v === "number") return v;
  const s = String(v).trim().replace(/\s/g, "").replace(",", ".");
  const n = Number(s);
  return Number.isFinite(n) ? n : NaN;
}
function isFiniteNum(x: number): boolean {
  return Number.isFinite(x);
}

// Traces Plotly génériques (typage compatible)
function hist(
  x: number[],
  name: string,
  nbins = 24
): Partial<Plotly.PlotData> & { nbinsx?: number } {
  return { x, type: "histogram" as const, name, nbinsx: nbins, opacity: 0.9 };
}
function boxY(
  y: number[],
  name: string
): Partial<Plotly.PlotData> & { boxpoints?: false | "all" | "outliers" | "suspectedoutliers" } {
  return { y, type: "box" as const, name, boxpoints: false };
}

/* ───────────────────── Enrichissement stations (fallback lat/lon) ───────────────────── */

type StationMeta = {
  station_id: string;
  name?: string | null;
  lat?: number | null;
  lon?: number | null;
  capacity?: number | null;
};
async function fetchStationsIndex(): Promise<Record<string, StationMeta>> {
  // ⚠️ adapte ce chemin à ton API si besoin (ex: "/monitoring/stations" ou "/api/stations")
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

/* ───────────────────── Mini MapView (clusters, Carto light_nolabels) ───────────────────── */

type ClusterRow = {
  station_id: string;
  name?: string;
  lat: number;
  lon: number;
  capacity_est?: number;
  cluster?: number | null;
};

type MapViewProps = {
  rows: ClusterRow[]; // doit contenir lat/lon/cluster
};

const MapView = dynamic(async () => {
  const RL = await import("react-leaflet");
  const { MapContainer, TileLayer, CircleMarker, Tooltip, useMap } = RL as any;

  const palette = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
  ];

  function colorOfCluster(c: number | null | undefined, uniq: number[]) {
    if (c === -1) return "#9e9e9e"; // bruit/outliers
    const idx = uniq.indexOf(Number(c));
    return idx >= 0 ? palette[idx % palette.length] : "#4c78a8";
  }

  function FitBounds({ rows }: { rows: ClusterRow[] }) {
    const map = useMap();
    useEffect(() => {
      if (!rows.length) return;
      // @ts-ignore Leaflet global
      const b = (window as any).L.latLngBounds(rows.map(r => [r.lat, r.lon]));
      map.fitBounds(b, { padding: [20, 20] });
    }, [rows, map]);
    return null;
  }

  function MapViewInner({ rows, sizeByCapacity, autoFit }: { rows: ClusterRow[]; sizeByCapacity: boolean; autoFit: boolean; }) {
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

    const uniq = Array.from(
      new Set(valid.map((r) => r.cluster).filter((v) => v != null))
    ) as number[];

    const [tileUrl, setTileUrl] = useState(
      "https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png"
    );
    useEffect(() => {
      const img = new Image();
      img.onerror = () => setTileUrl("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png");
      img.src = "https://a.basemaps.cartocdn.com/light_nolabels/3/4/2.png";
    }, []);

    return (
      <div style={{ position: "relative", height: "100%", width: "100%" }}>
        <MapContainer
          center={[latMed, lonMed]}
          zoom={12}
          style={{ height: "100%", width: "100%", background: "#fff" }}
        >
          <TileLayer
            url={tileUrl}
            attribution='&copy; OpenStreetMap, &copy; <a href="https://carto.com/">CARTO</a>'
            detectRetina={true}
          />
          {autoFit && !!valid.length && <FitBounds rows={valid} />}

          {valid.map((r) => {
            const cap = Number(r.capacity_est);
            const rad = sizeByCapacity
              ? (Number.isFinite(cap) && cap > 0 ? Math.max(2, Math.min(9, Math.sqrt(cap) / 2.4)) : 3)
              : 4;
            const col = colorOfCluster(r.cluster ?? null, uniq);
            const title = `${r.name ?? r.station_id} • ${Number.isFinite(cap) ? `cap≈${cap | 0} • ` : ""}cluster=${r.cluster ?? "—"}`;

            return (
              <CircleMarker
                key={r.station_id}
                center={[r.lat, r.lon]}
                radius={rad}
                pathOptions={{ color: col, weight: 0.8, fillColor: col, fillOpacity: 0.8 }}
              >
                <Tooltip>
                  <div style={{display:"grid", gap:4}}>
                    <div><b>{r.name ?? r.station_id}</b></div>
                    {Number.isFinite(cap) && <div>cap≈{cap | 0}</div>}
                    <div>cluster: {r.cluster ?? "—"}</div>
                    <a
                      href={`/monitoring/network/dynamics?station_id=${encodeURIComponent(r.station_id)}`}
                      style={{textDecoration:"underline"}}
                    >
                      Voir dynamique →
                    </a>
                  </div>
                </Tooltip>
              </CircleMarker>
            );
          })}
        </MapContainer>

        {/* Légende clusters */}
        <div
          style={{
            position: "absolute", right: 8, bottom: 8,
            background: "rgba(255,255,255,0.92)", color: "#111",
            borderRadius: 10, padding: "8px 10px", fontSize: 12,
            boxShadow: "0 2px 10px rgba(0,0,0,0.15)"
          }}
        >
          <div style={{ fontWeight: 700, marginBottom: 6 }}>Clusters</div>
          {uniq.map((c) => (
            <div key={String(c)} style={{ display: "flex", alignItems: "center", gap: 6, margin: "2px 0" }}>
              <span style={{
                width: 12, height: 12, borderRadius: 999,
                background: colorOfCluster(c, uniq), border: "1px solid #0002"
              }}/>
              <span>Cluster {c}</span>
            </div>
          ))}
          <div style={{ opacity: .7, marginTop: 6 }}>{valid.length} stations</div>
        </div>
      </div>
    );
  }

  return MapViewInner;
}, { ssr: false });


/* ───────────────────────── Types (API docs MONITORING) ───────────────────────── */

/** KPIs globaux du clustering */
export type KpisDoc = {
  schema_version: string;
  generated_at: string;         // ISO UTC ("2025-10-13T01:00:00Z")
  n_stations: number | null;    // nombre de stations analysées
  k_effective: number | null;   // nombre de clusters retenus
  silhouette: number | null;    // métrique silhouette (0..1)
  davies_bouldin: number | null;// métrique Davies–Bouldin (faible = mieux)
  window_days: number;          // nombre de jours pris en compte
};

/** Profil moyen par cluster (ou global si un seul) */
export type CentroidsDoc = {
  schema_version: string;
  generated_at: string;         // ISO UTC
  x_labels: string[];           // ["00:00", "01:00", ..., "23:00"]
  centroids: Array<{
    cluster: number;            // identifiant de cluster
    y: (number | null)[];       // profil moyen horaire (24 valeurs)
  }>;
};

/** Projection PCA (scatter) : 1 point par station */
export type PcaScatterDoc = {
  schema_version: string;
  generated_at: string;
  var_ratio: [number, number] | null; // variance expliquée par PC1/PC2
  points: Array<{
    station_id: string;
    name?: string | null;
    cluster?: number | null;
    PC1: number;
    PC2: number;
  }>;
};

/** Cercle PCA : composantes + variance */
export type PcaCircleDoc = {
  schema_version: string;
  generated_at: string;
  feature_names: string[];        // mêmes que x_labels
  components: [number[], number[]] | null; // deux vecteurs (PC1, PC2)
  var_ratio: [number, number] | null;      // variance expliquée
};

/** Tableau résumé par station (≤1000 lignes) */
export type Stats7Doc = {
  schema_version: string;
  generated_at: string;
  rows: Array<{
    station_id: string;
    name?: string | null;
    lat?: number | null;            // pour la carte
    lon?: number | null;            // pour la carte
    capacity_est?: number | null;
    volatility?: number | null;
    penury_rate?: number | null;
    saturation_rate?: number | null;
    coverage_pct?: number | null;
    cluster?: number | null;
  }>;
};

/* ───────────────────────── Page ───────────────────────── */

export default function NetworkStationsPage() {
  // State monitoring
  const [kpis, setKpis] = useState<KpisDoc | null>(null);
  const [centroids, setCentroids] = useState<CentroidsDoc | null>(null);
  const [stats7Doc, setStats7Doc] = useState<Stats7Doc | null>(null);

  // State carte (clusters)
  const [clusterRows, setClusterRows] = useState<ClusterRow[]>([]);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // ── Filtres / options carte
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

  // Chargement monitoring + data pour carte (clusters)
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setLoading(true);

        // Monitoring
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

        // Carte clusters — utiliser stats7; si lat/lon absents, enrichir via /stations
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
            haveCoords = rows.some(r => Number.isFinite(r.lat) && Number.isFinite(r.lon));
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
    return () => {
      alive = false;
    };
  }, []);

  // Derived
  const generatedAt =
    kpis?.generated_at ?? centroids?.generated_at ?? stats7Doc?.generated_at;

  // Lignes filtrées pour la carte
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

  // Centroids
  const xLabels =
    centroids?.x_labels ??
    Array.from({ length: 24 }, (_, h) => `${String(h).padStart(2, "0")}:00`);
  const centroidTraces: Partial<Plotly.PlotData>[] = useMemo(() => {
    if (!centroids?.centroids) return [];
    return centroids.centroids
      .sort((a, b) => a.cluster - b.cluster)
      .map((c) => ({
        x: xLabels,
        y: (c.y ?? []).map((v) =>
          Number.isFinite(Number(v)) ? Number(v) : null
        ),
        type: "scatter" as const,
        mode: "lines" as const,
        name: `Cluster ${c.cluster}`,
        connectgaps: false,
      }));
  }, [centroids, xLabels]);

  // Stats — distributions (parse robuste + filtrage)
  const rows = stats7Doc?.rows ?? [];

  // Couverture (%) — on convertit en %
  const coverageAll = rows
    .map((r) => parseRatioLoose(r["coverage_pct"]))
    .filter(isFiniteNum)
    .map((v) => v * 100);

  /** Volatilité (écart-type du ratio d’occupation) */
  const volatilityAll = rows
    .map((r) => parseNumberLoose(r["volatility"]))
    .filter(isFiniteNum);

  /** Taux de pénurie (%) */
  const penuryPct = rows
    .map((r) => parseRatioLoose(r["penury_rate"]) * 100)
    .filter(isFiniteNum);

  /** Taux de saturation (%) */
  const saturationPct = rows
    .map((r) => parseRatioLoose(r["saturation_rate"]) * 100)
    .filter(isFiniteNum);

  return (
    <>
      <Head>
        <title>Monitoring — Network / Stations</title>
        <meta
          name="description"
          content="Carte (clusters) + profils 24 h (centroids) et distributions récentes."
        />
        {/* Leaflet CSS */}
        <link
          rel="stylesheet"
          href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
          integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
          crossOrigin=""
        />
        {/* Fond blanc pour la carte + réactiver le scroll global si bloqué ailleurs */}
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
            <h1 style={{ margin: 0 }}>Network — Stations</h1>
            <div className="small" style={{ opacity: 0.7 }}>
              {generatedAt ? `Généré : ${new Date(generatedAt).toLocaleString("fr-FR")}` : "—"}
            </div>
          </div>

          <nav style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <Link
              href="/monitoring"
              style={{
                padding: "6px 10px",
                borderRadius: 8,
                border: "1px solid #d1d5db",
                background: "#111827",
                color: "white",
                textDecoration: "none",
              }}
            >
              Overview
            </Link>
            <Link
              href="/monitoring/network/dynamics"
              style={{
                padding: "6px 10px",
                borderRadius: 8,
                border: "1px solid #d1d5db",
                background: "white",
                color: "#111827",
                textDecoration: "none",
              }}
            >
              Dynamics
            </Link>
          </nav>
        </header>

        {/* Loading / Error */}
        {loading && <Banner kind="info">Chargement…</Banner>}
        {error && <Banner kind="error">{error}</Banner>}

        {/* KPIs */}
        <section style={{ marginTop: 24 }}>
          <h2 style={{ margin: "12px 0" }}>Résumé clustering</h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: 12 }}>
            <Kpi label="Stations" value={kpis?.n_stations} />
            <Kpi label="Clusters (k)" value={kpis?.k_effective} />
            <Kpi label="Silhouette" value={kpis?.silhouette} fmt={fmt3} />
            <Kpi label="Davies-Bouldin" value={kpis?.davies_bouldin} fmt={fmt3} />
          </div>
          <div style={{ fontSize: 12, opacity: 0.7, marginTop: 6 }}>
            Fenêtre: {kpis?.window_days ?? "—"} jours · Schema v{kpis?.schema_version ?? "—"}
          </div>
        </section>

        {/* Carte — Clusters */}
        <section style={{ marginTop: 24 }}>
          <h2 style={{ margin: "12px 0" }}>Carte des stations — Clusters</h2>
          {/* Filtres carte */}
          <div style={{display:"flex", gap:8, flexWrap:"wrap", alignItems:"center", margin: "4px 0 12px"}}>
            <input
              placeholder="Rechercher une station…"
              value={query}
              onChange={(e)=>setQuery(e.target.value)}
              style={{padding:"6px 8px", borderRadius:8, border:"1px solid #374151", background:"#0b1220", color:"#e5e7eb"}}
            />
            <select
              value={clusterFilter ?? ""}
              onChange={(e)=>setClusterFilter(e.target.value===""?null:Number(e.target.value))}
              style={{padding:"6px 8px", borderRadius:8, border:"1px solid #374151", background:"#0b1220", color:"#e5e7eb"}}
            >
              <option value="">Tous clusters</option>
              {Array.from(new Set(clusterRows.map(r=>r.cluster).filter(v=>v!=null)))
                .sort((a:any,b:any)=>Number(a)-Number(b))
                .map((c:any)=><option key={String(c)} value={String(c)}>Cluster {String(c)}</option>)}
            </select>
            <label style={{display:"flex", gap:6, alignItems:"center"}}>
              <input type="checkbox" checked={sizeByCapacity} onChange={(e)=>setSizeByCapacity(e.target.checked)} />
              Taille = capacité
            </label>
            <label style={{display:"flex", gap:6, alignItems:"center"}}>
              <input type="checkbox" checked={autoFit} onChange={(e)=>setAutoFit(e.target.checked)} />
              Auto-centrage
            </label>
            <button onClick={exportCSV}
              style={{padding:"6px 10px", borderRadius:8, border:"1px solid #374151", background:"white", color:"#111827"}}>
              Export CSV
            </button>
          </div>
          <div
            style={{
              border: "1px solid #374151",
              background: "rgba(15, 23, 42, 0.5)",
              borderRadius: 12,
              overflow: "hidden",
            }}
          >
            <div style={{ width: "100%", height: 520 }}>
              {filteredRows.length ? (
                <MapView rows={filteredRows} sizeByCapacity={sizeByCapacity} autoFit={autoFit} />
              ) : (
                <div style={{ padding: 16, opacity: 0.7 }}>
                  Pas de données de cluster/coordonnées pour la carte.
                </div>
              )}
            </div>
          </div>
          <div style={{ fontSize: 12, opacity: 0.7, marginTop: 6 }}>
            Fond: Carto Light (no labels) · Couleur = cluster · Taille ≈ capacité estimée.
          </div>
        </section>

        {/* Profils 24h — Centroids */}
        <section style={{ marginTop: 24 }}>
          <h2 style={{ margin: "12px 0" }}>Profils 24h — Centroids</h2>
          {centroidTraces.length ? (
            <Plot
              data={centroidTraces as Plotly.Data[]}
              layout={{
                autosize: true,
                height: 380,
                margin: { l: 52, r: 10, t: 30, b: 40 },
                xaxis: { title: { text: "Heure" } },
                yaxis: { title: { text: "Occupation (ratio)" }, rangemode: "tozero" },
                legend: { orientation: "h" },
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                hovermode: "x unified",
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: "100%" }}
            />
          ) : (
            <Empty>Centroids non disponibles.</Empty>
          )}
        </section>

        {/* Stats — distributions (7 jours) */}
        <section style={{ marginTop: 24 }}>
          <h2 style={{ margin: "12px 0" }}>Indicateurs récents — distributions</h2>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 16 }}>
            <Card>
              <h3 style={{ margin: "6px 0 10px 0", fontSize: 16 }}>Couverture (7 jours)</h3>
              {coverageAll.length ? (
                <Plot
                  data={[hist(coverageAll, "coverage", 24)] as Plotly.Data[]}
                  layout={{
                    autosize: true,
                    height: 280,
                    margin: { l: 48, r: 10, t: 10, b: 36 },
                    xaxis: { title: { text: "Couverture (%)" }, range: [0, 100] },
                    yaxis: { title: { text: "Stations" } },
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: "100%" }}
                />
              ) : (
                <Empty>Pas de données couverture.</Empty>
              )}
            </Card>

            <Card>
              <h3 style={{ margin: "6px 0 10px 0", fontSize: 16 }}>Volatilité des vélos (σ, 7 jours)</h3>
              {volatilityAll.length ? (
                <Plot
                  data={[boxY(volatilityAll, "σ(bikes)")] as Plotly.Data[]}
                  layout={{
                    autosize: true,
                    height: 280,
                    margin: { l: 48, r: 10, t: 10, b: 36 },
                    yaxis: { title: { text: "σ vélos / station (7 j)" }, rangemode: "tozero" },
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: "100%" }}
                />
              ) : (
                <Empty>Pas de données de volatilité.</Empty>
              )}
            </Card>

            <Card>
              <h3 style={{ margin: "6px 0 10px 0", fontSize: 16 }}>% pénurie (7 jours)</h3>
              {penuryPct.length ? (
                <Plot
                  data={[hist(penuryPct, "% pénurie", 24)] as Plotly.Data[]}
                  layout={{
                    autosize: true,
                    height: 280,
                    margin: { l: 48, r: 10, t: 10, b: 36 },
                    xaxis: { title: { text: "Pourcentage du temps (%)" }, range: [0, 100] },
                    yaxis: { title: { text: "Stations" } },
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: "100%" }}
                />
              ) : (
                <Empty>Pas de données pénurie.</Empty>
              )}
            </Card>

            <Card>
              <h3 style={{ margin: "6px 0 10px 0", fontSize: 16 }}>% saturation (7 jours)</h3>
              {saturationPct.length ? (
                <Plot
                  data={[hist(saturationPct, "% saturation", 24)] as Plotly.Data[]}
                  layout={{
                    autosize: true,
                    height: 280,
                    margin: { l: 48, r: 10, t: 10, b: 36 },
                    xaxis: { title: { text: "Pourcentage du temps (%)" }, range: [0, 100] },
                    yaxis: { title: { text: "Stations" } },
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: "100%" }}
                />
              ) : (
                <Empty>Pas de données saturation.</Empty>
              )}
            </Card>
          </div>

          {stats7Doc && (
            <div style={{ marginTop: 8, fontSize: 12, opacity: 0.7 }}>
              Schema v{stats7Doc.schema_version} — generated {stats7Doc.generated_at}
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

function Banner({ kind, children }: { kind: "info" | "error"; children: React.ReactNode }) {
  const style =
    kind === "error"
      ? { border: "1px solid #DC2626", background: "rgba(220, 38, 38, 0.08)", color: "#F87171" }
      : { border: "1px solid #374151", background: "rgba(15, 23, 42, 0.5)", color: "inherit", opacity: 0.85 };
  return (
    <div style={{ marginTop: 16, borderRadius: 10, padding: "10px 12px", fontSize: 13, ...style }}>{children}</div>
  );
}

function Card({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ border: "1px solid #374151", background: "rgba(15, 23, 42, 0.5)", borderRadius: 12, padding: 12 }}>
      {children}
    </div>
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
