// ui/pages/monitoring/data/drift.tsx
//
// =============================================================================
// Page Monitoring — Data Drift
// -----------------------------------------------------------------------------
// Cette page affiche le suivi du "data drift" entre la distribution de
// référence et les données récentes, via différents indicateurs :
//   - PSI global + top variables par PSI
//   - PSI / KS / deltas (moyenne, variance) par variable
//   - EMA temporelle du PSI global (tendance)
//   - Carte des zones géographiques avec PSI par zone
//
// Elle consomme les endpoints exposés par le backend de monitoring :
//   - /monitoring/data/drift/summary
//   - /monitoring/data/drift/psi
//   - /monitoring/data/drift/ks
//   - /monitoring/data/drift/deltas
//   - /monitoring/data/drift/psi_ema
//   - /monitoring/data/drift/zones
//
// Les composants UI (MonitoringNav, KpiBar, etc.) sont stylés par
// monitoring.css, et les graphiques utilisent react-plotly.js + notre thème
// Plotly custom (chartLayout, chartConfig).
// =============================================================================

import Head from "next/head";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import dynamic from "next/dynamic";
import type * as Plotly from "plotly.js";
import MonitoringNav from "@/components/monitoring/MonitoringNav";
import LoadingBar, { type LoadingBarStatus } from "@/components/common/LoadingBar";
import KpiBar from "@/components/monitoring/KpiBar";
import { chartConfig, chartLayout } from "@/lib/plotlyTheme";

import {
  getDataDriftSummary,
  getDataDriftPsiByFeature,
  getDataDriftKsByFeature,
  getDataDriftDeltasByFeature,
  getDataDriftPsiGlobalDailyEma,
  getDataDriftZones,
  type DriftSummary,
  type RowPSI,
  type RowKS,
  type RowDelta,
  type EmaPoint,
  type ZonesDoc,
} from "@/lib/services/monitoring/data_drift";

/* ───────── Plotly (client only) ───────── */
// Graphique rendu uniquement côté client (ssr: false) pour éviter les problèmes
// liés à window / document pendant le rendu serveur.
const Plot = dynamic(() => import("react-plotly.js").then((m) => m.default), {
  ssr: false,
  loading: () => <div className="empty">Chargement du graphique…</div>,
});

/* ───────── Helpers ───────── */

/**
 * Helper pour extraire la valeur d’un PromiseSettledResult si "fulfilled",
 * sinon null. Permet de consommer Promise.allSettled() proprement.
 */
function ok<T>(r: PromiseSettledResult<T>): T | null {
  return r.status === "fulfilled" ? r.value : null;
}

/**
 * Format numérique simple : x → "x.xxx" (ou "—" si non numérique).
 */
const fmt = (x: number | null | undefined, d = 3) =>
  Number.isFinite(Number(x)) ? Number(x).toFixed(d) : "—";

/* ───────── Zones Map (fond clair + fallback) ───────── */
// Carte Leaflet dynamique pour les zones (PSI par zone).
// Le dynamic import évite de charger Leaflet côté serveur.
const ZonesMap = dynamic(async () => {
  const RL = await import("react-leaflet");
  const { MapContainer, TileLayer, CircleMarker, Tooltip, useMap } = RL as any;
  const { useEffect, useMemo, useState } = await import("react");

  /**
   * Couleur HSL en fonction du PSI de la zone.
   * - PSI faible → teintes bleues / violettes
   * - PSI élevé → teintes tirant vers l’orange/rouge
   */
  function psiColor(psi: number | null | undefined): string {
    const v = Number(psi);
    if (!Number.isFinite(v)) return "#94a3b8"; // gris neutre si PSI invalide
    const x = Math.max(0, Math.min(0.35, v)) / 0.35;
    const h = x < 0.5 ? 210 + (300 - 210) * (x / 0.5) : 300 + (30 - 300) * ((x - 0.5) / 0.5);
    const s = x < 0.5 ? 60 : 85;
    const l = 55;
    return `hsl(${((h % 360) + 360) % 360}deg ${s}% ${l}%)`;
  }

  /**
   * Rayon des cercles en fonction du PSI (plus le PSI est élevé, plus c’est gros).
   */
  function psiRadius(psi: number | null | undefined): number {
    const v = Number(psi);
    if (!Number.isFinite(v)) return 3.5;
    return Math.max(2.5, Math.min(7, 2.5 + (v / 0.35) * 4.5));
  }

  /**
   * Composant interne chargé d’ajuster la vue de la carte pour englober
   * l’ensemble des points valides.
   */
  function Fit({ rows }: { rows: ZonesDoc["rows"] }) {
    const map = useMap();
    useEffect(() => {
      const pts = rows.filter((r) => Number.isFinite(Number(r.lat)) && Number.isFinite(Number(r.lon)));
      if (!pts.length) return;
      // @ts-ignore Leaflet global "L" exposé sur window par react-leaflet
      const b = (window as any).L?.latLngBounds
        ? (window as any).L.latLngBounds(pts.map((r) => [Number(r.lat), Number(r.lon)]))
        : null;
      if (b) map.fitBounds(b, { padding: [20, 20] });
    }, [rows, map]);
    return null;
  }

  /**
   * Carte interne :
   *   - fond Carto "light_nolabels" avec fallback OpenStreetMap,
   *   - cercles proportionnels au PSI,
   *   - tooltip sombre avec nom de zone + PSI.
   */
  function MapInner({ rows }: { rows: ZonesDoc["rows"] }) {
    const valid = useMemo(
      () => rows.filter((r) => Number.isFinite(Number(r.lat)) && Number.isFinite(Number(r.lon))),
      [rows]
    );
    const lat0 = valid.length ? Number(valid[0].lat) : 48.8566;
    const lon0 = valid.length ? Number(valid[0].lon) : 2.3522;

    // Test d’accessibilité du fond Carto ; fallback OSM si échec.
    const [tileUrl, setTileUrl] = useState(
      "https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png"
    );
    useEffect(() => {
      const img = new Image();
      img.onerror = () => setTileUrl("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png");
      img.src = "https://a.basemaps.cartocdn.com/light_nolabels/3/4/2.png";
    }, []);

    return (
      <div style={{ position: "relative", width: "100%", height: "100%" }}>
        <MapContainer center={[lat0, lon0]} zoom={12} className="leaflet-container">
          <TileLayer
            url={tileUrl}
            attribution='&copy; OpenStreetMap, &copy; <a href="https://carto.com/">CARTO</a>'
            detectRetina
          />
          <Fit rows={valid} />
          {valid.map((r, i) => {
            const c = psiColor(r.psi);
            const rad = psiRadius(r.psi);
            return (
              <CircleMarker
                key={r.zone ?? String(i)}
                center={[Number(r.lat), Number(r.lon)]}
                radius={psiRadius(r.psi)}
                pathOptions={{
                  color: "rgba(0,0,0,0.25)",
                  weight: 0.5,
                  fillColor: c,
                  fillOpacity: 0.85,
                }}
              >
                <Tooltip direction="top" offset={[0, -8]} opacity={1} className="tooltip-dark">
                  <div className="small">
                    <div><b>Zone</b> : {r.zone ?? "—"}</div>
                    <div><b>PSI</b> : {Number.isFinite(Number(r.psi)) ? Number(r.psi).toFixed(3) : "—"}</div>
                  </div>
                </Tooltip>
              </CircleMarker>
            );
          })}
        </MapContainer>
      </div>
    );
  }

  return MapInner;
}, { ssr: false });

/* ───────── Page ───────── */
export default function DataDriftPage() {
  // Données brutes récupérées via les services de monitoring
  const [summary, setSummary] = useState<DriftSummary | null>(null);
  const [psi, setPsi] = useState<RowPSI[] | null>(null);
  const [ks, setKs] = useState<RowKS[] | null>(null);
  const [deltas, setDeltas] = useState<RowDelta[] | null>(null);
  const [ema, setEma] = useState<EmaPoint[] | null>(null);
  const [zones, setZones] = useState<ZonesDoc | null>(null);

  // État de chargement / erreur global pour la page
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // État des filtres UI
  const [query, setQuery] = useState("");
  const [minPsi, setMinPsi] = useState<number>(0);

  // ---------------------------------------------------------------------------
  // Chargement initial de toutes les sources (summary, PSI, KS, deltas, EMA, zones)
  // ---------------------------------------------------------------------------
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setLoading(true);
        const res = await Promise.allSettled([
          getDataDriftSummary(),
          getDataDriftPsiByFeature(),
          getDataDriftKsByFeature(),
          getDataDriftDeltasByFeature(),
          getDataDriftPsiGlobalDailyEma(),
          getDataDriftZones(),
        ]);
        if (!alive) return;

        setSummary(ok(res[0]));
        setPsi(ok(res[1]));
        setKs(ok(res[2]));
        setDeltas(ok(res[3]));
        setEma(ok(res[4]));
        setZones(ok(res[5]));

        // Agrégation des erreurs éventuelles (au moins un appel rejeté)
        const failures = res.filter(
          (r): r is PromiseRejectedResult => r.status === "rejected"
        );
        if (failures.length > 0) {
          const msg =
            failures
              .map((f) => String((f.reason && (f.reason.message ?? f.reason)) || "request failed"))
              .join(" | ") || "Erreur API";
          setError(msg);
        } else {
          setError(null);
        }
      } catch (e: any) {
        if (alive) setError(String(e?.message ?? e));
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, []);

  // Fusion PSI + KS + deltas sur la clé "feature"
  const merged = useMemo(() => {
    const p = psi ?? [];
    const k = new Map((ks ?? []).map((r) => [r.feature, r.ks]));
    const d = new Map((deltas ?? []).map((r) => [r.feature, { dm: r.delta_mean, dv: r.delta_var }]));
    return p
      .map((r) => ({
        feature: r.feature,
        psi: Number(r.psi),
        ks: Number(k.get(r.feature)),
        delta_mean: Number(d.get(r.feature)?.dm),
        delta_var: Number(d.get(r.feature)?.dv),
      }))
      .filter((r) => r.feature && (Number.isFinite(r.psi) || Number.isFinite(r.ks)));
  }, [psi, ks, deltas]);

  // Filtrage (search + PSI min) + tri décroissant sur PSI
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return merged
      .filter((r) => (q ? r.feature.toLowerCase().includes(q) : true))
      .filter((r) => (Number.isFinite(r.psi) ? Number(r.psi) >= (minPsi || 0) : true))
      .sort((a, b) => Number(b.psi ?? -Infinity) - Number(a.psi ?? -Infinity));
  }, [merged, query, minPsi]);

  // Données Plotly pour le bar chart des top variables par PSI
  const topBars: Partial<Plotly.PlotData>[] = useMemo(() => {
    const top = filtered.slice(0, 15);
    if (!top.length) return [];
    return [{
      x: top.map((r) => r.feature),
      y: top.map((r) => Number(r.psi)),
      type: "bar" as const,
      name: "PSI",
      hovertemplate: "%{x} — %{y:.3f}<extra></extra>",
    }];
  }, [filtered]);

  // Données Plotly pour la série EMA (PSI global)
  const emaLine: Partial<Plotly.PlotData> | null = useMemo(() => {
    const arr = ema ?? [];
    if (!arr.length) return null;
    return {
      x: arr.map((p) => p.date_local),
      y: arr.map((p) => (Number.isFinite(Number(p.psi_ema)) ? Number(p.psi_ema) : null)),
      type: "scatter",
      mode: "lines",
      name: "PSI EMA",
      connectgaps: false,
      hovertemplate: "%{x}<br>%{y:.3f}<extra>PSI EMA</extra>",
    };
  }, [ema]);

  const generatedAt = summary?.generated_at ?? undefined;
  const barStatus: LoadingBarStatus = loading ? "loading" : error ? "error" : "success";

  // Items pour la barre de KPI (PSI global, nb de variables, top feature, etc.)
  const kpiItems = useMemo(() => {
    const featuresCount = (psi ?? []).length;
    return [
      { label: "PSI global", value: fmt(summary?.psi_global, 3) },
      { label: "Variables (PSI)", value: Number.isFinite(featuresCount) ? String(featuresCount) : "—" },
      { label: "Variable la plus affectée", value: summary?.top_feature ?? "—" },
      { label: "PSI (variable en tête)", value: fmt(summary?.top_feature_psi, 3) },
    ];
  }, [summary, psi]);

  // Extraction de quelques métadonnées optionnelles (fenêtre, version de schéma…)
  const windowDays =
    (summary as any)?.last_days ??
    (summary as any)?.window_days ??
    undefined;

  const schemaVersion = summary?.schema_version ?? undefined;

  const metaParts: string[] = [];
  if (windowDays != null) metaParts.push(`Fenêtre : ${windowDays} j`);
  if (schemaVersion != null) metaParts.push(`Schéma v${schemaVersion}`);
  if (generatedAt) metaParts.push(`généré ${generatedAt}`);
  const metaLine = metaParts.join(" · ");

  return (
    <div className="monitoring">
      <Head>
        <title>Monitoring — Données / Drift</title>
        <meta name="description" content="Suivi du data drift (PSI, KS, deltas, zones, EMA)." />
      </Head>

      <main className="page" style={{ paddingTop: "calc(var(--header-h, 70px) + 12px)" }}>
        <MonitoringNav
          title="Données — Drift"
          subtitle="PSI, KS, deltas, zones et EMA"
          generatedAt={generatedAt}
          extraActions={[{ label: "Santé", href: "/monitoring/data/health" }]}
        />

        <LoadingBar status={barStatus} />

        {/* KPI + méta (bandeau synthétique en haut de page) */}
        <section className="mt-4">
          <KpiBar items={kpiItems} dense />
          {metaLine && <div className="kpi-bar-meta">{metaLine}</div>}
        </section>

        {/* Filtres (recherche texte + seuil PSI mini) */}
        <section className="mt-4">
          <div className="card">
            <div className="row">
              <div className="filters">
                <input
                  className="input"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Filtrer une variable…"
                />
                <input
                  className="input"
                  type="number"
                  step="0.01"
                  min={0}
                  value={minPsi}
                  onChange={(e) => setMinPsi(Number(e.target.value))}
                  placeholder="PSI min"
                  style={{ width: 140 }}
                />
              </div>
            </div>
          </div>
        </section>

        {/* Top PSI (bar chart) */}
        <section className="mt-6">
          <h2>Top PSI (variables)</h2>
          <div className="card plot-card">
            {topBars.length ? (
              <Plot
                data={topBars as Plotly.Data[]}
                layout={chartLayout({
                  autosize: true,
                  height: 360,
                  margin: { l: 60, r: 10, t: 10, b: 120 },
                  xaxis: { tickangle: -35, title: { text: "" } },
                  yaxis: { title: { text: "PSI" } },
                })}
                config={chartConfig}
                className="plot plot--sm"
              />
            ) : (
              <div className="empty">Pas de PSI disponible.</div>
            )}
          </div>
          <div className="figure-note small">Lecture : top 15 variables par PSI (plus haut = drift plus marqué).</div>
        </section>

        {/* EMA temporelle du PSI global */}
        <section className="mt-6">
          <h2>Tendance — PSI global (EMA)</h2>
          <div className="card plot-card">
            {emaLine ? (
              <Plot
                data={[emaLine] as Plotly.Data[]}
                layout={chartLayout({
                  autosize: true,
                  height: 320,
                  margin: { l: 60, r: 10, t: 10, b: 40 },
                  xaxis: { title: { text: "Date (locale)" } },
                  yaxis: { title: { text: "PSI EMA" } },
                  hovermode: "x unified",
                })}
                config={chartConfig}
                className="plot plot--lg"
              />
            ) : (
              <div className="empty">Aucune série EMA.</div>
            )}
          </div>
        </section>

        {/* Tableau complet des features (PSI / KS / deltas) */}
        <section className="mt-6">
          <h2>Toutes les variables — PSI / KS / Δ</h2>
          <div className="card">
            {filtered.length ? (
              <div className="table-scroll">
                <div
                  className="table-grid"
                  style={{ ["--cols" as any]: "minmax(220px,1.2fr) 120px 120px 140px 160px" }}
                >
                  <HeaderCell>Variable</HeaderCell>
                  <HeaderCell>PSI</HeaderCell>
                  <HeaderCell>KS</HeaderCell>
                  <HeaderCell>Δ moyenne (z)</HeaderCell>
                  <HeaderCell>Δ variance (rel)</HeaderCell>

                  {filtered.map((r) => (
                    <Row key={r.feature}>
                      <div
                        className="table-cell--ellipsis"
                        style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" }}
                        title={r.feature}
                      >
                        {r.feature}
                      </div>
                      <div style={{ fontVariantNumeric: "tabular-nums" }}>{fmt(r.psi, 3)}</div>
                      <div style={{ fontVariantNumeric: "tabular-nums" }}>{fmt(r.ks, 3)}</div>
                      <div style={{ fontVariantNumeric: "tabular-nums" }}>{fmt(r.delta_mean, 3)}</div>
                      <div style={{ fontVariantNumeric: "tabular-nums" }}>{fmt(r.delta_var, 3)}</div>
                    </Row>
                  ))}
                </div>
              </div>
            ) : (
              <div className="empty">Aucune variable ne correspond aux filtres.</div>
            )}
          </div>
        </section>

        {/* Carte des zones (PSI agrégé par zone) */}
        <section className="mt-6">
          <h2>PSI par zones</h2>
          <div className="map-block">
            <div className="map-wrap h-360">
              {zones?.rows?.length ? (
                <ZonesMap rows={zones.rows} />
              ) : (
                <div className="empty">Pas de données zones.</div>
              )}
            </div>
          </div>
          <div className="figure-note small">Taille/couleur proportionnelles au PSI de la zone.</div>
        </section>
      </main>
    </div>
  );
}

/* ───────── UI atoms (branchés sur monitoring.css) ───────── */

/**
 * Ligne de tableau générique :
 * - aligne à droite les cellules numériques (dernières colonnes),
 * - conserve la première colonne alignée à gauche.
 */
function Row({ children }: { children: ReactNode }) {
  const items = Array.isArray(children) ? children : [children];
  return (
    <div className="table-row">
      {items.map((child, i) => {
        const alignRight = items.length >= 5 && i >= items.length - 4;
        return (
          <div key={i} className={`table-cell ${alignRight ? "table-cell--right" : ""}`}>
            {child}
          </div>
        );
      })}
    </div>
  );
}

/**
 * Cellule d’en-tête de tableau (ligne sticky en haut du scroll).
 */
function HeaderCell({ children }: { children: ReactNode }) {
  return <div className="table-head table-head--sticky">{children}</div>;
}