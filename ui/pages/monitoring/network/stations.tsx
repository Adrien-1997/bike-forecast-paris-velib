// ui/pages/monitoring/network/stations.tsx
//
// -----------------------------------------------------------------------------
// Monitoring — Réseau / Stations
//
// Page de monitoring dédiée au "profil" des stations :
//   - vue d’ensemble du clustering des stations (K-means sur profils d’occupation),
//   - carte Leaflet avec couleurs = cluster et taille ≈ capacité,
//   - distributions de couverture, volatilité, pénurie et saturation sur 7 jours.
//
// Données consommées (services /monitoring/network_stations) :
//   - getStationsKpis()       → métriques globales de clustering (silhouette, k effectif, etc.)
//   - getStationsCentroids()  → profils 24 h moyens par cluster (centroïdes)
//   - getStationsStats7()     → indicateurs par station sur les 7 derniers jours
//   - fetchStationsIndex()    → (fallback) méta station (nom, lat/lon, capacité) si absent de stats7
//
// Design / UX :
//   - même header MonitoringNav + KpiBar + LoadingBar que les autres pages Monitoring,
//   - carte Leaflet avec légende "Clusters" (bubble en bas à droite),
//   - histogrammes et boxplots Plotly pour les distributions,
//   - aucun lien cliquable vers une page de station (règle : stations non cliquables).
//
// Important :
//   - Cette page est purement "read-only" : aucun état serveur modifié,
//   - tous les filtres (cluster, recherche, options carte) sont gérés en mémoire côté client,
//   - le code JSX / logique ne doit pas être modifié par d’autres fichiers ; seule la doc
//     (commentaires) peut être enrichie si besoin.
//
// -----------------------------------------------------------------------------

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
/**
 * Plotly React est chargé dynamiquement côté client uniquement (ssr: false),
 * ce qui évite d’embarquer la librairie dans le bundle serveur Next.js.
 */
const Plot = dynamic(() => import("react-plotly.js").then((m) => m.default), {
  ssr: false,
  loading: () => (
    <div style={{ height: 320, display: "grid", placeItems: "center", opacity: 0.7 }}>
      Chargement du graphique…
    </div>
  ),
});

/* ───────────────── Helpers ───────────────── */
/**
 * Raccourci pour récupérer la valeur d’un PromiseSettledResult<T>.
 * Retourne `null` si la promesse est rejetée.
 */
function ok<T>(r: PromiseSettledResult<T>): T | null {
  return r.status === "fulfilled" ? r.value : null;
}

/**
 * Formate un nombre avec 3 décimales ou renvoie "—" si non numérique.
 * Utilisé pour les métriques de clustering (silhouette, Davies–Bouldin).
 */
function fmt3(x?: number | null) {
  const v = Number(x);
  return Number.isFinite(v) ? v.toFixed(3) : "—";
}

/**
 * Parse "souple" d’un ratio (0–1 ou "42%" ou "0,42") → nombre en [0,1] (NaN si invalide).
 * Permet de robustifier le traitement des colonnes venant des exports parquet/JSON.
 */
function parseRatioLoose(v: unknown): number {
  if (v == null) return NaN;
  if (typeof v === "number") return v;
  let s = String(v).trim();
  const isPercent = s.endsWith("%");
  s = s.replace("%", "").replace(/\s/g, "").replace(",", ".");
  const n = Number(s);
  return Number.isFinite(n) ? (isPercent ? n / 100 : n) : NaN;
}

/**
 * Parse "souple" d’un nombre flottant (gère espaces, virgules) → number (NaN si invalide).
 */
function parseNumberLoose(v: unknown): number {
  if (v == null) return NaN;
  if (typeof v === "number") return v;
  const n = Number(String(v).trim().replace(/\s/g, "").replace(",", "."));
  return Number.isFinite(n) ? n : NaN;
}

const isFiniteNum = (x: number) => Number.isFinite(x);

/**
 * Helper pour construire un histogramme Plotly à partir d’un tableau de valeurs.
 * - `x`: valeurs numériques
 * - `name`: libellé de la série
 * - `nbins`: nombre de bins (24 par défaut)
 */
function hist(
  x: number[],
  name: string,
  nbins = 24
): Partial<Plotly.PlotData> & { nbinsx?: number } {
  return { x, type: "histogram" as const, name, nbinsx: nbins, opacity: 0.9, hovertemplate: "%{x:.1f}<extra></extra>" };
}

/**
 * Helper pour construire un boxplot Plotly vertical sur l’axe Y.
 */
function boxY(
  y: number[],
  name: string
): Partial<Plotly.PlotData> & {
  boxpoints?: false | "all" | "outliers" | "suspectedoutliers";
} {
  return { y, type: "box" as const, name, boxpoints: false, hovertemplate: "%{y:.2f}<extra></extra>" };
}

/* ───────────────── Mini MapView (Leaflet) ───────────────── */
/**
 * Ligne de données pour la carte des stations :
 *   - station_id : identifiant unique de la station
 *   - name       : nom éventuel (si non fourni par stats7, backfill via stationsIndex)
 *   - lat / lon  : géolocalisation
 *   - capacity_est : capacité estimée (slots vélos)
 *   - cluster    : cluster numérique (ou null)
 */
type ClusterRow = {
  station_id: string;
  name?: string;
  lat: number;
  lon: number;
  capacity_est?: number;
  cluster?: number | null;
};

/**
 * Carte Leaflet affichant toutes les stations filtrées :
 *   - couleur = cluster,
 *   - taille ≈ capacité estimée (optionnelle),
 *   - légende en bas à droite "Clusters".
 *
 * La carte prend :
 *   - rows          : lignes de stations (ClusterRow[])
 *   - sizeByCapacity: booléen pour activer/désactiver la taille ≈ capacité
 *   - autoFit       : booléen pour activer/désactiver le fitBounds automatique
 */
const MapView = dynamic(async () => {
  const RL = await import("react-leaflet");
  const { MapContainer, TileLayer, CircleMarker, Tooltip, useMap } = RL as any;
  const { useEffect, useMemo, useState } = await import("react");

  // Palette de base pour les clusters (réutilisable côté légende)
  const palette = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
  ];

  /**
   * Retourne une couleur stable pour un numéro de cluster donné.
   * -1 est traité comme "bruit" (gris).
   */
  function colorOfCluster(c: number | null | undefined, uniq: number[]) {
    if (c === -1) return "#9e9e9e";
    const idx = uniq.indexOf(Number(c));
    return idx >= 0 ? palette[idx % palette.length] : "#4c78a8";
  }

  /**
   * Adapte le viewport de la carte aux points visibles (fitBounds).
   */
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
    // Filtre les stations pour lesquelles lat/lon sont valides
    const valid = useMemo(
      () => rows.filter((r) => Number.isFinite(Number(r.lat)) && Number.isFinite(Number(r.lon))),
      [rows]
    );

    // Centre initial (médiane des lat/lon) si pas encore fitBounds
    const latMed = valid.length
      ? valid.map((r) => r.lat).sort((a, b) => a - b)[Math.floor(valid.length / 2)]
      : 48.8566;
    const lonMed = valid.length
      ? valid.map((r) => r.lon).sort((a, b) => a - b)[Math.floor(valid.length / 2)]
      : 2.3522;

    // Liste unique des clusters présents (pour légende + palette)
    const uniq = Array.from(new Set(valid.map((r) => r.cluster).filter((v) => v != null))) as number[];

    // Basemap Carto Light (fallback OSM si indisponible)
    const [tileUrl, setTileUrl] = useState(
      "https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png"
    );
    useEffect(() => {
      const img = new Image();
      img.onerror = () => setTileUrl("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png");
      img.src = "https://a.basemaps.cartocdn.com/light_nolabels/3/4/2.png";
    }, []);

    return (
      <div className="map-wrap h-520" style={{ position: "relative", width: "100%", height: "100%" }}>
        <MapContainer
          center={[latMed, lonMed]}
          zoom={12}
          className="tile-bg"
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
                <Tooltip className="tooltip-dark">
                  <div style={{ display: "grid", gap: 4 }}>
                    <div><b>{r.name ?? r.station_id}</b></div>
                    {Number.isFinite(cap) && <div>cap≈{cap | 0}</div>}
                    <div>cluster: {r.cluster ?? "—"}</div>
                    {/* lien désactivé (règle: stations non cliquables) */}
                    <div className="small" style={{ opacity: 0.7 }}>Voir dynamique (lien désactivé)</div>
                  </div>
                </Tooltip>
              </CircleMarker>
            );
          })}
        </MapContainer>

        {/* Légende clusters (harmonisée) */}
        <div className="cluster-legend" style={{ right: 8, bottom: 8 }}>
          <div className="cluster-legend__title">Clusters</div>
          {uniq.map((c) => (
            <div key={String(c)} className="cluster-legend__row">
              <span
                className="cluster-legend__dot"
                style={{ background: colorOfCluster(c, uniq) }}
              />
              <span>Cluster {c}</span>
            </div>
          ))}
          <div className="cluster-legend__meta">{valid.length} stations</div>
        </div>
      </div>
    );
  }

  return MapViewInner;
}, { ssr: false });

/* ───────────────── Page ───────────────── */
/**
 * Page de monitoring "Réseau — Stations".
 *
 * Cycle de vie :
 *   1) Au montage, on appelle en parallèle :
 *        - getStationsKpis()
 *        - getStationsCentroids()
 *        - getStationsStats7()
 *      via Promise.allSettled pour tolérer les erreurs partielles.
 *   2) On construit `clusterRows` à partir de stats7, avec fallback éventuel
 *      sur fetchStationsIndex() pour compléter lat/lon/capacité manquants.
 *   3) Les filtres (cluster, recherche, taille, auto-fit) sont purement client.
 *
 * Rendu :
 *   - KpiBar (résumé clustering),
 *   - carte Leaflet (clusters + filters),
 *   - profils 24 h (centroids),
 *   - distributions (coverage, volatilité, pénurie, saturation).
 */
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

  // ✅ LoadingBar status (aligné avec overview.tsx / dynamics.tsx)
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

        // Fetch des 3 documents principaux en parallèle (KPIs, centroids, stats7)
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

        // Gestion d’erreur : agrège les messages si des appels ont échoué
        const failures = [rKpis, rCent, rStats].filter(
          (r): r is PromiseRejectedResult => r.status === "rejected"
        );
        setError(
          failures.length
            ? failures
                .map((f) => String((f.reason && (f.reason.message ?? f.reason)) || "request failed"))
                .join(" | ")
            : null
        );

        // Construction des lignes pour la carte à partir de stats7 (7 derniers jours)
        if (s7 && Array.isArray(s7.rows)) {
          let rows = s7.rows.map((x: any) => ({
            station_id: String(x.station_id),
            name: x.name ?? undefined,
            lat: Number.isFinite(Number(x.lat)) ? Number(x.lat) : NaN,
            lon: Number.isFinite(Number(x.lon)) ? Number(x.lon) : NaN,
            capacity_est: Number.isFinite(Number(x.capacity_est)) ? Number(x.capacity_est) : NaN,
            cluster: x.cluster != null ? Number(x.cluster) : null,
          })) as ClusterRow[];

          // Fallback : si aucune station n’a de coordonnées valides, on tente de backfiller
          // via fetchStationsIndex() (meta statique).
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

  // Timestamp de génération le plus récent parmi les différentes sources
  const generatedAt = kpis?.generated_at ?? centroids?.generated_at ?? stats7Doc?.generated_at;

  /**
   * Applique les filtres "cluster" + "recherche" sur les lignes de stations.
   * Les filtres n’affectent que l’affichage de la carte.
   */
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

  // Labels X pour les centroïdes (24 points HH:00 par défaut)
  const xLabels =
    (centroids?.x_labels as string[] | undefined) ??
    Array.from({ length: 24 }, (_, h) => `${String(h).padStart(2, "0")}:00`);

  /**
   * Séries Plotly pour les profils 24 h par cluster (centroids).
   */
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

  // Raccourcis pour les distributions 7j
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
        <title>Monitoring — Réseau / Stations</title>
        <meta name="description" content="Clusters, carte et distributions par station." />
      </Head>

      {/* Main content (header/footer injectés par _app.tsx) */}
      <main className="page" style={{ paddingTop: "calc(var(--header-h, 70px) + 12px)" }}>
        <MonitoringNav
          title="Réseau — Stations"
          subtitle="Clusters, carte et distributions"
          generatedAt={generatedAt}
          extraActions={[
            { label: "Aperçu", href: "/monitoring/network/overview" },
            { label: "Dynamiques", href: "/monitoring/network/dynamics" },
          ]}
        />

        <LoadingBar status={barStatus} />
        {error && <div className="alert error" style={{ marginTop: 8 }}>{error}</div>}

        {/* ───────────────── KPIs (KpiBar) ───────────────── */}
        <section className="mt-4">
          <h2>Résumé du clustering</h2>

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
            Fenêtre : {kpis?.window_days ?? "—"} j · Schéma v{kpis?.schema_version ?? "—"}
          </div>
        </section>

        {/* Map */}
        <section className="mt-6">
          <h2>Carte des stations — Clusters</h2>
          <div className="filters">
            <input
              className="input"
              placeholder="Rechercher une station…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <select
              className="select"
              value={clusterFilter ?? ""}
              onChange={(e) => setClusterFilter(e.target.value === "" ? null : Number(e.target.value))}
            >
              <option value="">Tous clusters</option>
              {Array.from(new Set(clusterRows.map((r) => r.cluster).filter((v) => v != null)))
                .sort((a: any, b: any) => Number(a) - Number(b))
                .map((c: any) => (
                  <option key={String(c)} value={String(c)}>Cluster {String(c)}</option>
                ))}
            </select>
            <label className="check">
              <input type="checkbox" checked={sizeByCapacity} onChange={(e) => setSizeByCapacity(e.target.checked)} />
              Taille = capacité
            </label>
            <label className="check">
              <input type="checkbox" checked={autoFit} onChange={(e) => setAutoFit(e.target.checked)} />
              Auto-fit
            </label>
            {/* <button className="btn btn--primary" onClick={exportCSV}>Export CSV</button> */}
          </div>

          <div className="map-block" style={{ height: 520 }}>
            <MapView rows={filteredRows} sizeByCapacity={sizeByCapacity} autoFit={autoFit} />
          </div>
          <div className="figure-note small">
            Basemap : Carto Light (no labels). La couleur encode le cluster ; la taille ≈ capacité estimée.
          </div>
        </section>

        {/* Centroids */}
        <section className="mt-6">
          <h2>Profils 24 h — Centroids</h2>
          <div className="card plot-card">
            <h3>Profil moyen d’occupation 24 h par cluster (ratio)</h3>
            {centroidTraces.length ? (
              <Plot
                data={centroidTraces as Plotly.Data[]}
                layout={chartLayout({
                  height: 380,
                  xaxis: { title: { text: "Heure (HH:MM)" } },
                  yaxis: { title: { text: "Occupation (ratio)" }, rangemode: "tozero" },
                })}
                config={chartConfig}
                className="plot plot--lg"
              />
            ) : (
              <div className="empty">Centroids indisponibles.</div>
            )}
          </div>
          <div className="figure-note small">
            Chaque série est le centroïde du cluster sur 24 h ; les labels X proviennent de `x_labels` ou par défaut HH:00.
          </div>
        </section>

        {/* Distributions */}
        <section className="mt-6">
          <h2>Indicateurs récents — distributions</h2>
          <div className="grid-2">
            <div className="card plot-card">
              <h3>Couverture sur 7 jours — part du temps (%)</h3>
              {coverageAll.length ? (
                <Plot
                  data={[hist(coverageAll, "coverage", 24)] as Plotly.Data[]}
                  layout={chartLayout({
                    height: 280,
                    xaxis: { title: { text: "Couverture (%)" }, range: [0, 100] },
                    yaxis: { title: { text: "Stations (nombre)" } },
                  })}
                  config={chartConfig}
                  className="plot plot--sm"
                />
              ) : (
                <div className="empty">Pas de données de couverture.</div>
              )}
            </div>

            <div className="card plot-card">
              <h3>Volatilité σ des vélos — 7 derniers jours</h3>
              {volatilityAll.length ? (
                <Plot
                  data={[boxY(volatilityAll, "σ(vélos)")] as Plotly.Data[]}
                  layout={chartLayout({
                    height: 280,
                    yaxis: { title: { text: "σ vélos / station (7 j)" }, rangemode: "tozero" },
                  })}
                  config={chartConfig}
                  className="plot plot--sm"
                />
              ) : (
                <div className="empty">Pas de données de volatilité.</div>
              )}
            </div>

            <div className="card plot-card">
              <h3>Pénurie — part du temps (%) sur 7 jours</h3>
              {penuryPct.length ? (
                <Plot
                  data={[hist(penuryPct, "% pénurie", 24)] as Plotly.Data[]}
                  layout={chartLayout({
                    height: 280,
                    xaxis: { title: { text: "Pourcentage du temps (%)" }, range: [0, 100] },
                    yaxis: { title: { text: "Stations (nombre)" } },
                  })}
                  config={chartConfig}
                  className="plot plot--sm"
                />
              ) : (
                <div className="empty">Pas de données de pénurie.</div>
              )}
            </div>

            <div className="card plot-card">
              <h3>Saturation — part du temps (%) sur 7 jours</h3>
              {saturationPct.length ? (
                <Plot
                  data={[hist(saturationPct, "% saturation", 24)] as Plotly.Data[]}
                  layout={chartLayout({
                    height: 280,
                    xaxis: { title: { text: "Pourcentage du temps (%)" }, range: [0, 100] },
                    yaxis: { title: { text: "Stations (nombre)" } },
                  })}
                  config={chartConfig}
                  className="plot plot--sm"
                />
              ) : (
                <div className="empty">Pas de données de saturation.</div>
              )}
            </div>
          </div>

          {stats7Doc && (
            <div className="figure-note small">
              Schéma v{stats7Doc.schema_version} — généré {stats7Doc.generated_at}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}