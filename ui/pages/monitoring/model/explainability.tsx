// ui/pages/monitoring/model/explainability.tsx
//
// =============================================================================
// Page Monitoring — Modèle / Explicabilité
// -----------------------------------------------------------------------------
// Cette page affiche les vues d’explicabilité du modèle de prévision :
//   - structure des résidus (histogramme, QQ-plot, ACF, hétéroscédasticité),
//   - calibration (globale + par heure + par niveau),
//   - importance des variables (part + cumul),
//   - épisodes d’erreurs par station (carte + tableau),
//   - biais moyen par station.
// 
// Elle consomme les endpoints back-end suivants (par horizon h):
//   - /monitoring/model/explainability/overview
//   - /monitoring/model/explainability/residuals
//   - /monitoring/model/explainability/calibration
//   - /monitoring/model/explainability/uncertainty
//   - /monitoring/model/explainability/feature_importance
//
// L’horizon de prévision (h en minutes) est piloté par HorizonToggle et
// synchronisé avec le paramètre de query ?h=… (hook useQueryParamH).
// =============================================================================

import Head from "next/head";
import { useEffect, useMemo, useState, useCallback } from "react";
import dynamic from "next/dynamic";
import type { ScatterData } from "plotly.js";
import type * as Plotly from "plotly.js";
import MonitoringNav from "@/components/monitoring/MonitoringNav";
import LoadingBar, { type LoadingBarStatus } from "@/components/common/LoadingBar";
import KpiBar, { type KpiItem } from "@/components/monitoring/KpiBar";
import { chartConfig, chartLayout } from "@/lib/plotlyTheme";
import HorizonToggle from "@/components/common/HorizonToggle";

import {
  getExplainOverview,
  getExplainResiduals,
  getExplainCalibration,
  getExplainUncertainty,
  getExplainFeatureImportance,
  type Overview,
  type ResidualsDoc,
  type CalibrationDoc,
  type UncertaintyDoc,
  type FeatureImportanceDoc,
} from "@/lib/services/monitoring/model_explainability";

// ⬇️ Index stations (JSON local: public/data/stations.index.json)
//    Permet de décoder station_id → métadonnées (nom, lat/lon, etc.)
import { loadStationsIndexFromArrayJson, type StationMeta } from "@/lib/local/stationsIndex";

/* ───────────────── Plotly (client only) ───────────────── */
// Chargement dynamique de react-plotly.js uniquement côté client (ssr: false)
// pour éviter les accès à window/document côté serveur.
const Plot = dynamic(() => import("react-plotly.js").then((m) => m.default), {
  ssr: false,
  loading: () => (
    <div className="empty" style={{ minHeight: 320 }}>
      Chargement du graphique…
    </div>
  ),
});

/* ───────────────── Map — identique perf mais avec épisodes d’erreurs ───────────────── */
/**
 * Point sur la carte :
 *   - station_id / name : identifiants station,
 *   - lat / lon         : position géographique,
 *   - color             : couleur en fonction de la gravité,
 *   - max_run / n       : métriques sur les épisodes d’erreurs.
 */
type MapPoint = {
  station_id: string;
  name?: string;
  lat: number;
  lon: number;
  color: string;
  max_run?: number | null;
  n?: number | null;
};

// Carte Leaflet dynamique utilisée pour visualiser les épisodes d’erreurs par station.
const SnapshotMap = dynamic(async () => {
  const RL = await import("react-leaflet");
  const { MapContainer, TileLayer, CircleMarker, Tooltip, useMap } = RL as any;
  const React = await import("react");
  const { useEffect, useMemo, useState } = React;

  /**
   * Centre/zoom la vue pour englober l’ensemble des points valides.
   */
  function FitBounds({ rows }: { rows: Array<{ lat: number; lon: number }> }) {
    const map = useMap();
    useEffect(() => {
      const pts = rows.filter((r) => Number.isFinite(Number(r.lat)) && Number.isFinite(Number(r.lon)));
      if (!pts.length) return;
      let minLat = 90, maxLat = -90, minLon = 180, maxLon = -180;
      for (const r of pts) {
        const la = Number(r.lat);
        const lo = Number(r.lon);
        if (la < minLat) minLat = la;
        if (la > maxLat) maxLat = la;
        if (lo < minLon) minLon = lo;
        if (lo > maxLon) maxLon = lo;
      }
      if (minLat <= maxLat && minLon <= maxLon) {
        map.fitBounds(
          [
            [minLat, minLon],
            [maxLat, maxLon],
          ],
          { padding: [20, 20] }
        );
      }
    }, [rows, map]);
    return null;
  }

  /**
   * Carte interne :
   *   - fond Carto "light_nolabels" avec fallback OSM,
   *   - cercles colorés (rouge/vert) selon max_run,
   *   - tooltip détaillé par station.
   */
  function MapInner({ rows }: { rows: MapPoint[] }) {
    const valid = useMemo(
      () => rows.filter((r) => Number.isFinite(Number(r.lat)) && Number.isFinite(Number(r.lon))),
      [rows]
    );

    // Centre initial = médiane des lat/lon ou fallback sur Paris.
    const latMed = valid.length
      ? valid.map((r) => Number(r.lat)).sort((a, b) => a - b)[Math.floor(valid.length / 2)]
      : 48.8566;
    const lonMed = valid.length
      ? valid.map((r) => Number(r.lon)).sort((a, b) => a - b)[Math.floor(valid.length / 2)]
      : 2.3522;

    // Sélection dynamique du fond de carte (Carto → fallback OSM si indisponible).
    const [tileUrl, setTileUrl] = useState(
      "https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png"
    );
    useEffect(() => {
      const img = new Image();
      img.onerror = () => setTileUrl("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png");
      img.src = "https://a.basemaps.cartocdn.com/light_nolabels/3/4/2.png";
    }, []);

    return (
      <div className="map-wrap h-360">
        <MapContainer center={[latMed, lonMed]} zoom={12} className="tile-bg" style={{ width: "100%", height: "100%" }}>
          <TileLayer
            url={tileUrl}
            attribution='&copy; OpenStreetMap, &copy; <a href="https://carto.com/">CARTO</a>'
            detectRetina
          />
          <FitBounds rows={valid} />
          {valid.map((r) => (
            <CircleMarker
              key={r.station_id}
              center={[Number(r.lat), Number(r.lon)]}
              radius={6}
              pathOptions={{
                color: r.color,
                weight: 0.8,
                fillColor: r.color,
                fillOpacity: 0.85,
              }}
            >
              <Tooltip className="tooltip-dark">
                <div style={{ display: "grid", gap: 4 }}>
                  <div><b>{r.name ?? r.station_id}</b></div>
                  {r.n != null && (
                    <div className="mono" style={{ fontSize: 12, opacity: 0.85 }}>
                      {r.station_id} · n={r.n.toLocaleString("fr-FR")}
                    </div>
                  )}
                  {Number.isFinite(Number(r.max_run)) && (
                    <div>
                      Épisode max (|résidu|≥4) : <b>{Number(r.max_run).toFixed(0)}</b>
                    </div>
                  )}
                </div>
              </Tooltip>
            </CircleMarker>
          ))}
        </MapContainer>

        {/* Légende harmonisée CSS (style perf) */}
        <div className="cluster-legend">
          <div className="cluster-legend__title">Épisodes d’erreurs</div>
          <div className="cluster-legend__row">
            <span className="cluster-legend__dot" style={{ background: "#ef4444" }} />
            <span>Épisodes longs (max_run élevé)</span>
          </div>
          <div className="cluster-legend__row">
            <span className="cluster-legend__dot" style={{ background: "#10b981" }} />
            <span>Épisodes courts</span>
          </div>
        </div>
      </div>
    );
  }

  return MapInner;
}, { ssr: false });

/* ───────────────── Helpers ───────────────── */
/**
 * Helper pour Promise.allSettled :
 *   - renvoie r.value si status === "fulfilled",
 *   - sinon null.
 */
function ok<T>(r: PromiseSettledResult<T>): T | null {
  return r.status === "fulfilled" ? r.value : null;
}

/** Formatage numérique générique : x → "x.xx" ou "—". */
const fmtNum = (x?: number | null, d = 2) => (Number.isFinite(Number(x)) ? Number(x).toFixed(d) : "—");

/** Formatage entier français : x → "12 345" ou "—". */
const fmtInt = (x?: number | null) => (Number.isFinite(Number(x)) ? Number(x).toLocaleString("fr-FR") : "—");

/** Contrainte d’une valeur v dans [lo, hi]. */
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

/**
 * safeMinMax : calcule min / max sur un ensemble de tableaux numériques.
 * Fallback (0,0) si aucun nombre valide.
 */
const safeMinMax = (arrs: number[][]) => {
  let min = Number.POSITIVE_INFINITY, max = Number.NEGATIVE_INFINITY;
  for (const a of arrs) for (const v of a) if (Number.isFinite(v)) { if (v < min) min = v; if (v > max) max = v; }
  if (!Number.isFinite(min)) min = 0;
  if (!Number.isFinite(max)) max = 0;
  return { min, max };
};

/**
 * getLatLng : récupère un couple [lat, lon] à partir d’un StationMeta
 * en tolérant plusieurs schémas de champs (lat/lon, latitude/longitude, lng…).
 */
function getLatLng(meta?: StationMeta | null): [number, number] | null {
  if (!meta) return null;
  const lat = (meta as any).lat ?? (meta as any).latitude;
  const lon = (meta as any).lon ?? (meta as any).lng ?? (meta as any).longitude;
  const la = Number(lat), lo = Number(lon);
  if (!Number.isFinite(la) || !Number.isFinite(lo)) return null;
  return [la, lo];
}

/* ───────────────── Horizon (SSR-safe) ───────────────── */
/**
 * Hook pour piloter l’horizon h depuis la query string (?h=…) :
 *   - lit ?h=… au montage (client only),
 *   - met à jour l’URL avec history.replaceState à chaque changement,
 *   - renvoie un flag `mounted` indiquant que l’init client est fait.
 */
function useQueryParamH(defaultH = 60): [number, (h: number) => void, boolean] {
  const [h, setH] = useState<number>(defaultH);
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(true);
    try {
      const u = new URL(window.location.href);
      const qh = Number(u.searchParams.get("h"));
      if (Number.isFinite(qh) && qh > 0) setH(qh);
    } catch {}
  }, []);
  useEffect(() => {
    if (!mounted) return;
    try {
      const u = new URL(window.location.href);
      u.searchParams.set("h", String(h));
      window.history.replaceState({}, "", u.toString());
    } catch {}
  }, [h, mounted]);
  const setter = useCallback((v: number) => setH(v), []);
  return [h, setter, mounted];
}

/* ───────── Normalizer FI (gère XGB natif / scikit / custom) ───────── */
/**
 * Ligne d’importance de variable normalisée :
 *   - feature    : nom de la variable,
 *   - importance : score brut,
 *   - std        : incertitude éventuelle.
 */
type FIRow = { feature: string; importance: number; std: number | null };

/**
 * normalizeFeatureImportance :
 *   - unifie plusieurs schémas possibles envoyés par le backend,
 *   - renvoie une liste normalisée de lignes + métadonnées.
 *
 * Schémas gérés :
 *   0) XGBoost natif : rows[].gain_share / rows[].gain
 *   1) Schéma "officiel" : rows[].{feature, importance, std}
 *   2) scikit permutation_importance : feature_names[] + importances_mean[]
 *   3) Schéma plat : importances[].{feature, importance, std}
 */
function normalizeFeatureImportance(fi: any): { rows: FIRow[]; method: string | undefined; meta: string[] } {
  const method = typeof fi?.method === "string" ? fi.method : undefined;
  const meta: string[] = [];
  const rows: FIRow[] = [];

  // 0) XGBoost natif
  if (Array.isArray(fi?.rows) && fi.rows.length && (("gain_share" in fi.rows[0]) || ("gain" in fi.rows[0]))) {
    for (const r of fi.rows) {
      const f = String(r?.feature ?? "");
      const imp = Number.isFinite(Number(r?.gain_share)) ? Number(r.gain_share)
                : Number.isFinite(Number(r?.gain))       ? Number(r.gain)
                : null;
      if (f && imp != null) rows.push({ feature: f, importance: imp, std: null });
    }
  }
  // 1) Schéma “officiel”
  else if (Array.isArray(fi?.rows) && fi.rows.length) {
    for (const r of fi.rows) {
      const f = String(r?.feature ?? "");
      const imp = Number(r?.importance);
      const sd = Number.isFinite(Number(r?.std)) ? Number(r.std) : null;
      if (f && Number.isFinite(imp)) rows.push({ feature: f, importance: imp, std: sd });
    }
  }
  // 2) scikit permutation_importance
  else if (Array.isArray(fi?.feature_names) && Array.isArray(fi?.importances_mean)) {
    const names = fi.feature_names as any[];
    const mean = fi.importances_mean as any[];
    const stds = Array.isArray(fi?.importances_std) ? (fi.importances_std as any[]) : [];
    for (let i = 0; i < names.length; i++) {
      const f = String(names[i] ?? "");
      const imp = Number(mean[i]);
      const sd = Number.isFinite(Number(stds[i])) ? Number(stds[i]) : null;
      if (f && Number.isFinite(imp)) rows.push({ feature: f, importance: imp, std: sd });
    }
  }
  // 3) Plat
  else if (Array.isArray(fi?.importances) && fi.importances.length) {
    for (const r of fi.importances) {
      const f = String(r?.feature ?? "");
      const imp = Number(r?.importance);
      const sd = Number.isFinite(Number(r?.std)) ? Number(r.std) : null;
      if (f && Number.isFinite(imp)) rows.push({ feature: f, importance: imp, std: sd });
    }
  }

  // Métadonnées additionnelles affichées dans la légende FI.
  if (Number.isFinite(Number(fi?.horizon_min))) meta.push(`h=${Number(fi.horizon_min)} min`);
  if (Number.isFinite(Number(fi?.n_rows)))      meta.push(`n_rows=${Number(fi.n_rows)}`);
  if (Number.isFinite(Number(fi?.n_features)))  meta.push(`n_features=${Number(fi.n_features)}`);

  return { rows, method, meta };
}

/* ───────────────── Page ───────────────── */
export default function ModelExplainabilityPage() {
  // Horizon courant (15 / 60 min) et flag de montage client.
  const [h, setH, mounted] = useQueryParamH(60);

  // Documents d’explicabilité (par horizon).
  const [overview, setOverview] = useState<Overview | null>(null);
  const [residuals, setResiduals] = useState<ResidualsDoc | null>(null);
  const [calib, setCalib] = useState<CalibrationDoc | null>(null);
  const [unc, setUnc] = useState<UncertaintyDoc | null>(null);
  const [fiDoc, setFiDoc] = useState<FeatureImportanceDoc | any | null>(null);

  // Index stations: id → meta (nom, lat/lon, etc.).
  const [stationsIdx, setStationsIdx] = useState<Record<string, StationMeta>>({});

  // État global de chargement / erreur.
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Chargement de l’index stations local (JSON en public/…).
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const idx = await loadStationsIndexFromArrayJson("/data/stations.index.json").catch(() => ({}));
        if (!alive) return;
        setStationsIdx(idx as Record<string, StationMeta>);
      } catch {}
    })();
    return () => { alive = false; };
  }, []);

  // Chargement des docs d’explicabilité pour l’horizon h.
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setLoading(true);
        const res = await Promise.allSettled([
          getExplainOverview(h),
          getExplainResiduals(h),
          getExplainCalibration(h),
          getExplainUncertainty(h),
          getExplainFeatureImportance(h),
        ]);
        if (!alive) return;

        setOverview(ok(res[0]));
        setResiduals(ok(res[1]));
        setCalib(ok(res[2]));
        setUnc(ok(res[3]));
        setFiDoc(ok(res[4]));

        const failures = res.filter((r): r is PromiseRejectedResult => r.status === "rejected");
        setError(failures.length ? "Une ou plusieurs requêtes ont échoué." : null);
      } catch (e: any) {
        if (alive) setError(String(e?.message ?? e));
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, [h]);

  // Timestamp de génération : on prend le premier disponible.
  const generatedAt =
    (fiDoc as any)?.generated_at ||
    overview?.generated_at ||
    residuals?.generated_at ||
    calib?.generated_at ||
    unc?.generated_at ||
    undefined;

  const barStatus: LoadingBarStatus = loading ? "loading" : error ? "error" : "success";

  // Dictionnaire id → nom (tolère les ids sans zéros de tête).
  const nameIndex = useMemo(() => {
    const rec: Record<string, string> = {};
    for (const [id, meta] of Object.entries(stationsIdx)) {
      const nm = (meta?.name ?? "").trim();
      if (!nm) continue;
      rec[id] = nm;
      const noZeros = id.replace(/^0+/, "");
      if (!(noZeros in rec)) rec[noZeros] = nm;
    }
    return rec;
  }, [stationsIdx]);

  /* ───────── KPI BAR ───────── */
  // Construction des KPI synthétiques de la page (perf, y_pred, incertitude, β, α…)
  const kpiItems: KpiItem[] = useMemo(() => {
    const yesNo = (v?: boolean | null) => (v ? "Oui" : "Non");
    return [
      { label: "Stations (perf)", value: overview?.perf_stations ?? null, fmt: (v) => fmtInt(Number(v)) },
      { label: "Lignes (n)",      value: overview?.perf_rows ?? null,     fmt: (v) => fmtInt(Number(v)) },
      { label: "Prédictions",     value: overview?.has_y_pred ? 1 : 0,    fmt: (v) => yesNo(Boolean(v)) },
      { label: "Incertitude",     value: overview?.has_uncertainty ? 1:0, fmt: (v) => yesNo(Boolean(v)) },
      { label: "β global",        value: calib?.fit?.beta ?? null,        fmt: (v) => fmtNum(Number(v), 3) },
      { label: "α global",        value: calib?.fit?.alpha ?? null,       fmt: (v) => fmtNum(Number(v), 3) },
    ];
  }, [overview, calib]);

  // Sous-titre dynamique incluant l’horizon (h) une fois monté côté client.
  const subtitleText = useMemo(
    () => `Résidus, QQ, ACF, hétéroscédasticité, calibration, incertitude et importance des variables (h=${mounted ? h : 15} min)`,
    [h, mounted],
  );

  /* ───────── Data: résidus / calib / incertitude ───────── */

  // Histogramme des résidus (bin_left / bin_right → centre du bin).
  const histData: Partial<Plotly.PlotData>[] = useMemo(() => {
    const bins = residuals?.hist ?? [];
    if (!bins.length) return [];
    const x = bins.map((b) => (b.bin_left + b.bin_right) / 2);
    const y = bins.map((b) => b.count);
    return [{ x, y, type: "bar" as const, name: "Résidus", hovertemplate: "x=%{x:.2f}<br>n=%{y}<extra></extra>" }];
  }, [residuals]);

  // QQ-plot : quantiles théoriques vs quantiles empiriques.
  const qqData = useMemo<Partial<ScatterData>[]>(() => {
    const th = residuals?.qq?.th ?? [];
    const emp = residuals?.qq?.emp ?? [];
    if (!th.length || th.length !== emp.length) return [];
    const { min, max } = safeMinMax([th, emp]);
    return [
      { x: [min, max], y: [min, max], type: "scatter", mode: "lines", name: "y = x", hoverinfo: "none" },
      { x: th, y: emp, type: "scatter", mode: "markers", name: "Quantiles", hovertemplate: "th=%{x:.2f}<br>emp=%{y:.2f}<extra></extra>" },
    ];
  }, [residuals]);

  // ACF des résidus.
  const acfData: Partial<Plotly.PlotData>[] = useMemo(() => {
    const a = residuals?.acf ?? [];
    if (!a.length) return [];
    const x = Array.from({ length: a.length }, (_, i) => i);
    return [{ x, y: a, type: "bar" as const, name: "ACF", hovertemplate: "lag=%{x}<br>ρ=%{y:.2f}<extra></extra>" }];
  }, [residuals]);

  // Hétéroscédasticité : MAE par quantile de y_true.
  const heteroData: Partial<Plotly.PlotData>[] = useMemo(() => {
    const rows = residuals?.hetero ?? [];
    if (!rows.length) return [];
    return [{
      x: rows.map((r) => r.quantile),
      y: rows.map((r) => r.mae),
      type: "bar" as const,
      name: "MAE",
      hovertemplate: "q=%{x}<br>MAE=%{y:.2f}<extra></extra>",
    }];
  }, [residuals]);

  // Calibration : E[y_true] vs E[y_pred] par bin.
  const calibBinning = useMemo<Partial<ScatterData>[]>(() => {
    const rows = calib?.binned ?? [];
    if (!rows.length) return [];
    const xp = rows.map((r) => r.y_pred_mean);
    const yt = rows.map((r) => r.y_true_mean);
    const { min, max } = safeMinMax([xp, yt]);
    return [
      { x: [min, max], y: [min, max], type: "scatter", mode: "lines", name: "y = x", hoverinfo: "none" },
      { x: xp, y: yt, type: "scatter", mode: "markers", name: "Moyennes par bin", hovertemplate: "E[ŷ]=%{x:.2f}<br>E[y]=%{y:.2f}<extra></extra>" },
    ];
  }, [calib]);

  // β (pente) par heure locale.
  const betaByHour = useMemo<Partial<ScatterData>[]>(() => {
    const rows = calib?.by_hour ?? [];
    if (!rows.length) return [];
    return [{
      x: rows.map((r) => r.hour),
      y: rows.map((r) => (Number.isFinite(Number(r.beta)) ? Number(r.beta) : null)),
      type: "scatter",
      mode: "lines+markers",
      name: "β vs heure",
      hovertemplate: "h=%{x}<br>β=%{y:.3f}<extra></extra>",
    }];
  }, [calib]);

  // Erreur relative (type MAPE) par niveau de y_true.
  const relErrBars: Partial<Plotly.PlotData>[] = useMemo(() => {
    const rows = calib?.rel_error_levels ?? [];
    if (!rows.length) return [];
    return [{
      x: rows.map((r) => r.level),
      y: rows.map((r) => r.mape_like * 100),
      type: "bar" as const,
      name: "MAPE-like (%)",
      hovertemplate: "niveau=%{x}<br>%=%{y:.1f}<extra></extra>",
    }];
  }, [calib]);

  /* ───────── Importance des variables (part + cumul) ───────── */

  // Normalisation de la FI brute en tableau de FIRow.
  const { rows: fiRowsRaw, method: fiMethod, meta: fiMeta } = useMemo(
    () => normalizeFeatureImportance(fiDoc),
    [fiDoc]
  );

  // Calcul de la part (%) et du cumul (%) d’importance pour chaque variable.
  const fiAll = useMemo(() => {
    const rows = (fiRowsRaw ?? []).filter((r) => Number.isFinite(Number(r.importance)));
    if (!rows.length) return { rows: [] as Array<FIRow & { share: number; cum: number }>, total: 0 };
    const total = rows.reduce((s, r) => s + Math.max(0, Number(r.importance)), 0);
    const sorted = [...rows].sort((a, b) => Number(b.importance) - Number(a.importance));
    let cum = 0;
    const withPct = sorted.map((r) => {
      const share = total > 0 ? (Number(r.importance) / total) * 100 : 0;
      cum += share;
      return { ...r, share, cum };
    });
    return { rows: withPct, total };
  }, [fiRowsRaw]);

  // Intitulé de l’axe X en fonction de la méthode FI.
  const xTitleFI = useMemo(() => {
    if (fiMethod?.includes("get_score")) return "Part d’importance (gain_share, %)";
    return "Part d’importance (%)";
  }, [fiMethod]);

  // Hauteur dynamique du graphique d’importance (toutes variables visibles).
  const fiHeight = useMemo(() => {
    const n = fiAll.rows.length;
    return Math.max(380, Math.min(2000, 24 * n + 160));
  }, [fiAll.rows.length]);

  // Barres horizontales + texte part/cumul.
  const fiBarData: Partial<Plotly.PlotData>[] = useMemo(() => {
    if (!fiAll.rows.length) return [];
    const rows = fiAll.rows;
    const x = rows.map((r) => r.share);
    const y = rows.map((r) => r.feature);
    const err = rows.map((r) => (Number.isFinite(Number(r.std)) ? Number(r.std) : 0));
    const custom = rows.map((r) => [r.share, r.cum]);

    return [
      {
        x: x.reverse(),
        y: y.reverse(),
        type: "bar" as const,
        orientation: "h",
        name: "Part",
        error_x: {
          type: "data",
          array: err.reverse(),
          visible: err.some((v) => v > 0),
        },
        text: custom.reverse().map(([p, c]) => `${(p as number).toFixed(1)}% • cum ${(c as number).toFixed(1)}%`),
        textposition: "auto",
        insidetextanchor: "end",
        hovertemplate: "%{y}<br>part=%{x:.2f}%<br>cumul=%{text.split('cum ')[1]}<extra></extra>",
        customdata: custom.reverse(),
      },
    ];
  }, [fiAll]);

  // Relayout auto sur resize (via variable CSS observée côté Plotly).
  useEffect(() => {
    if (typeof window === "undefined") return;
    const onResize = () => {
      document.documentElement.style.setProperty("--fi-resize-tick", String(Date.now() % 1000));
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  /* ───────── Données carte (épisodes d’erreurs) ───────── */
  // Agrégation des épisodes par station + filtrage n ≥ 30 + coloration rouge/vert.
  const mapRows = useMemo<MapPoint[]>(() => {
    const rows: MapPoint[] = [];
    const src = residuals?.episodes ?? [];
    if (!src.length) return rows;

    // Agrégation par station : max_run (épisode le plus long) + total n.
    const aggregated = new Map<string, { station_id: string; max_run: number; n: number }>();
    for (const r of src) {
      const id = String(r.station_id);
      const cur = aggregated.get(id);
      const mr = Number(r.max_run ?? 0);
      const nn = Number(r.n ?? 0);
      if (!cur) aggregated.set(id, { station_id: id, max_run: mr, n: nn });
      else aggregated.set(id, { station_id: id, max_run: Math.max(cur.max_run, mr), n: cur.n + nn });
    }

    const list = Array.from(aggregated.values()).filter((r) => Number(r.n) >= 30);
    if (!list.length) return rows;

    const vals = list.map((t) => Number(t.max_run ?? 0)).filter((v) => Number.isFinite(v)) as number[];
    const median = vals.length ? [...vals].sort((a, b) => a - b)[Math.floor(vals.length / 2)] : 0;

    for (const r of list) {
      const meta = stationsIdx[r.station_id] as StationMeta | undefined;
      const ll = getLatLng(meta);
      if (!ll) continue;
      const color = (Number(r.max_run) >= median ? "#ef4444" : "#10b981");
      rows.push({
        station_id: r.station_id,
        name: (meta as any)?.name ?? r.station_id,
        lat: ll[0],
        lon: ll[1],
        color,
        max_run: Number(r.max_run),
        n: Number(r.n),
      });
    }
    return rows;
  }, [residuals, stationsIdx]);

  /* ───────── RENDER ───────── */
  return (
    <div className="monitoring">
      <Head>
        <title>Monitoring — Modèle / Explicabilité</title>
        <meta name="description" content="Résidus, QQ, ACF, hétéroscédasticité, calibration, incertitude et importance des variables." />
      </Head>

      <main className="page" style={{ paddingTop: "calc(var(--header-h, 70px) + 12px)" }}>
        <MonitoringNav
          title="Modèle — Explicabilité"
          subtitle={subtitleText}
          generatedAt={generatedAt}
          extraActions={[{ label: "Performance", href: "/monitoring/model/performance" }]}
        />

        <LoadingBar status={barStatus} />
        {error && <div className="alert error" style={{ marginTop: 8 }}>{error}</div>}

        {/* Toolbar : choix d’horizon 15 / 60 min (propagé dans l’URL). */}
        <section className="mt-3">
          <div className="card" style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 10 }}>
            <HorizonToggle
              value={h}
              onChange={(v: number) => setH(v)}
              leftValue={15}
              rightValue={60}
              leftLabel="15 min"
              rightLabel="60 min"
              ariaLabel="Choix de l’horizon de prévision"
            />
          </div>
        </section>

        {/* KPIs synthétiques (overview + calibration). */}
        <section className="mt-4">
          <h2>Résumé — KPIs</h2>
          <KpiBar items={kpiItems} dense />
          {(() => {
            const parts: string[] = [];
            if (overview?.schema_version != null) parts.push(`Schéma v${overview.schema_version}`);
            if (overview?.ts_min_perf || overview?.ts_max_perf)
              parts.push(`Intervalle : ${overview?.ts_min_perf ?? "—"} → ${overview?.ts_max_perf ?? "—"} (UTC)`);
            if (typeof (overview as any)?.horizon_min === "number") parts.push(`h=${(overview as any).horizon_min} min`);
            if ((overview as any)?.window_days) parts.push(`fenêtre=${(overview as any).window_days} j`);
            return parts.length ? <div className="kpi-bar-meta">{parts.join(" · ")}</div> : null;
          })()}
        </section>

        {/* Résidus : histogramme + QQ-plot. */}
        <section className="mt-6">
          <h2>Résidus</h2>
          <div className="grid-2">
            <div className="plot-card">
              <h3>Histogramme</h3>
              {histData.length ? (
                <Plot
                  data={histData as Plotly.Data[]}
                  layout={chartLayout({
                    height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 40 },
                    xaxis: { title: { text: "Erreur (y_true - y_pred)" } },
                    yaxis: { title: { text: "Comptes" } },
                  })}
                  config={chartConfig}
                  className="plot plot--sm"
                />
              ) : (
                <div className="empty">—</div>
              )}
            </div>

            <div className="plot-card">
              <h3>QQ-plot</h3>
              {qqData.length ? (
                <Plot
                  data={qqData as Plotly.Data[]}
                  layout={chartLayout({
                    height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 40 },
                    xaxis: { title: { text: "Théorique" } },
                    yaxis: { title: { text: "Empirique" } },
                    hovermode: "closest",
                  })}
                  config={chartConfig}
                  className="plot plot--sm"
                />
              ) : (
                <div className="empty">—</div>
              )}
            </div>
          </div>
          <div className="figure-note small">
            Lecture : histogramme des résidus sur la fenêtre courante ; QQ-plot pour vérifier la normalité (la droite y=x indique une gaussienne idéale).
          </div>
        </section>

        {/* Structure des erreurs : ACF / hétéroscédasticité. */}
        <section className="mt-6">
          <h2>Structure des erreurs</h2>
          <div className="grid-2">
            <div className="plot-card">
              <h3>ACF du résidu (lag 5 min)</h3>
              {acfData.length ? (
                <Plot
                  data={acfData as Plotly.Data[]}
                  layout={chartLayout({
                    height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 36 },
                    xaxis: { title: { text: "Lag (5 min)" } },
                    yaxis: { title: { text: "Corrélation" } },
                  })}
                  config={chartConfig}
                  className="plot plot--sm"
                />
              ) : (
                <div className="empty">—</div>
              )}
            </div>

            <div className="plot-card">
              <h3>Hétéroscédasticité</h3>
              {heteroData.length ? (
                <Plot
                  data={heteroData as Plotly.Data[]}
                  layout={chartLayout({
                    height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 36 },
                    xaxis: { title: { text: "Quantiles (y_true)" }, tickangle: -30 },
                    yaxis: { title: { text: "MAE" } },
                  })}
                  config={chartConfig}
                  className="plot plot--sm"
                />
              ) : (
                <div className="empty">—</div>
              )}
            </div>
          </div>
          <div className="figure-note small">
            Lecture : l’ACF montre la dépendance temporelle résiduelle ; l’hétéroscédasticité trace la MAE selon des tranches de y_true.
          </div>
        </section>

        {/* Calibration : binned + β par heure + erreur relative. */}
        <section className="mt-6">
          <h2>Calibration</h2>

          <div className="plot-card">
            <h3>Moyennes par bin (ŷ → y)</h3>
            {calibBinning.length ? (
              <Plot
                data={calibBinning as Plotly.Data[]}
                layout={chartLayout({
                  height: 320,
                  margin: { l: 54, r: 10, t: 10, b: 40 },
                  xaxis: { title: { text: "E[ŷ]" } },
                  yaxis: { title: { text: "E[y]" } },
                })}
                config={chartConfig}
                className="plot plot--sm"
              />
            ) : (
              <div className="empty">—</div>
            )}
          </div>

          <div className="grid-2 mt-4">
            <div className="plot-card">
              <h3>β par heure (locale)</h3>
              {betaByHour.length ? (
                <Plot
                  data={betaByHour as Plotly.Data[]}
                  layout={chartLayout({
                    height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 36 },
                    xaxis: { title: { text: "Heure" } },
                    yaxis: { title: { text: "β" } },
                  })}
                  config={chartConfig}
                  className="plot plot--sm"
                />
              ) : (
                <div className="empty">—</div>
              )}
            </div>

            <div className="plot-card">
              <h3>Erreur relative par niveau</h3>
              {relErrBars.length ? (
                <Plot
                  data={relErrBars as Plotly.Data[]}
                  layout={chartLayout({
                    height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 36 },
                    xaxis: { title: { text: "Niveau (quantiles y_true)" } },
                    yaxis: { title: { text: "MAPE-like (%)" } },
                  })}
                  config={chartConfig}
                  className="plot plot--sm"
                />
              ) : (
                <div className="empty">—</div>
              )}
            </div>
          </div>
        </section>

        {/* Importance des variables — part & cumul. */}
        <section className="mt-6">
          <h2>Importance des variables — part & cumul</h2>
          <div className="plot-card" style={{ overflow: "visible" }}>
            <div className="row" style={{ justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <div className="small muted">
                {fiMethod
                  ? `Méthode : ${fiMethod}${fiMeta.length ? " · " + fiMeta.join(" · ") : ""}`
                  : (fiDoc ? "Méthode inconnue" : "Aucune donnée disponible")}
              </div>
              {fiMethod?.includes("disabled") && (
                <div className="small" style={{ color: "var(--warn)" }}>
                  L’explicabilité par permutation est désactivée côté backend.
                </div>
              )}
            </div>

            {fiBarData.length ? (
              <Plot
                data={fiBarData as Plotly.Data[]}
                layout={chartLayout({
                  height: fiHeight,
                  margin: { l: 280, r: 12, t: 10, b: 48 },
                  xaxis: { title: { text: xTitleFI }, ticksuffix: "%", rangemode: "tozero" },
                  yaxis: { automargin: true },
                  hovermode: "closest",
                })}
                config={chartConfig}
                className="plot plot--lg"
                style={{ minHeight: fiHeight, overflow: "visible" }}
              />
            ) : (
              <div className="empty">
                {fiMethod?.includes("disabled")
                  ? "Désactivé côté backend."
                  : "Aucune importance calculée / données vides."}
              </div>
            )}
          </div>
          {!!(fiDoc as any)?.notes?.length && <div className="figure-note small">{(fiDoc as any).notes.join(" · ")}</div>}
        </section>

        {/* Carte — intégration identique à Performance, basée sur épisodes d’erreurs. */}
        <section className="mt-6">
          <h2>Carte — épisodes d’erreurs par station</h2>
          <div className="map-block">
            {mapRows.length ? <SnapshotMap rows={mapRows} /> : <div className="empty">Aucune donnée carte.</div>}
          </div>
          <div className="small mt-2">Couleur ~ longueur maximale d’un épisode |résidu| ≥ 4 (rouge = long, vert = court). Filtre n ≥ 30.</div>
        </section>

        {/* Tableaux : épisodes d’erreurs. */}
        <section className="mt-6">
          <h2>Épisodes d’erreurs (|résidu| ≥ 4)</h2>
          <div className="card">
            {residuals?.episodes?.length ? (
              <>
                <div className="table-scroll">
                  <div className="table-grid" style={{ ["--cols" as any]: "minmax(0,1fr) 1fr 78px" }}>
                    <div className="table-head table-head--sticky">Station</div>
                    <div className="table-head table-head--sticky">Durée max (pas)</div>
                    <div className="table-head table-head--sticky">n épisodes</div>

                    {(() => {
                      const rows = residuals.episodes.slice(0, 160);
                      const maxRun = Math.max(...rows.map((r) => Number(r.max_run ?? 0)), 1);

                      return rows.map((r) => {
                        const val = Number(r.max_run ?? 0);
                        const widthPct = clamp((val / maxRun) * 100, 0, 100);
                        const nm = nameIndex[r.station_id] ?? `#${r.station_id}`;
                        return (
                          <div className="table-row" key={r.station_id}>
                            <div className="table-cell">
                              <div className="table-cell--ellipsis" style={{ fontWeight: 600 }} title={nm}>
                                {nm}
                              </div>
                              <div className="mono" style={{ fontSize: 12, opacity: 0.7 }}>#{r.station_id} · max run={fmtInt(r.max_run)}</div>
                            </div>

                            <div className="table-cell">
                              <div style={{ display: "grid", gridTemplateColumns: "1fr 64px", alignItems: "center", gap: 10 }}>
                                <div className="bar">
                                  <div className="bar__fill bar__fill--danger" style={{ width: `${widthPct}%` }} aria-hidden />
                                </div>
                                <div className="table-cell--right" style={{ fontWeight: 700 }}>{fmtInt(val)}</div>
                              </div>
                            </div>

                            <div className="table-cell table-cell--right" style={{ fontWeight: 700 }}>
                              {fmtInt(r.n)}
                            </div>
                          </div>
                        );
                      });
                    })()}
                  </div>
                </div>
                <div className="figure-note small">
                  Lecture : pour chaque station, on affiche la plus longue séquence consécutive où |résidu| ≥ 4, normalisée par le maximum du tableau.
                </div>
              </>
            ) : (
              <div className="empty">Aucun épisode détecté.</div>
            )}
          </div>
        </section>

        {/* Stations — biais moyen. */}
        <section className="mt-6" style={{ marginBottom: 40 }}>
          <h2>Stations — biais moyen (référence)</h2>
          <div className="card">
            {(() => {
              const src = calib?.bias_by_station ?? [];
              if (!src.length) return <div className="empty">Aucune station à afficher.</div>;

              const filtered = src.filter((r) => Number(r.n) >= 30);
              if (!filtered.length) return <div className="empty">Aucune station après filtre n ≥ 30.</div>;

              const maxAbs = Math.max(...filtered.map((r) => Math.abs(Number(r.bias ?? 0))), 1);
              const rows = filtered
                .map((r) => ({ ...r, abs: Math.abs(Number(r.bias ?? 0)) }))
                .sort((a, b) => b.abs - a.abs)
                .slice(0, 160);

              return (
                <>
                  <div className="table-scroll">
                    <div className="table-grid" style={{ ["--cols" as any]: "minmax(0,1fr) 1fr 78px" }}>
                      <div className="table-head table-head--sticky">Station</div>
                      <div className="table-head table-head--sticky">|biais|</div>
                      <div className="table-head table-head--sticky">n</div>

                      {rows.map((r) => {
                        const absBias = Math.abs(Number(r.bias ?? 0));
                        const widthPct = clamp((absBias / maxAbs) * 100, 0, 100);
                        const nm = nameIndex[r.station_id] ?? r.name ?? `#${r.station_id}`;
                        return (
                          <div className="table-row" key={r.station_id}>
                            <div className="table-cell">
                              <div className="table-cell--ellipsis" style={{ fontWeight: 600 }} title={nm}>
                                {nm}
                              </div>
                              <div className="mono" style={{ fontSize: 12, opacity: 0.7 }}>#{r.station_id}</div>
                              <div className="bar" style={{ height: 5, marginTop: 5, width: "min(75%, 320px)" }}>
                                <div className="bar__fill bar__fill--danger" style={{ width: `${widthPct}%` }} aria-hidden />
                              </div>
                            </div>

                            <div className="table-cell table-cell--right" style={{ fontWeight: 700 }}>
                              {fmtNum(absBias, 2)}
                            </div>

                            <div className="table-cell table-cell--right">{fmtInt(r.n)}</div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                  <div className="figure-note small">
                    Lecture : |biais| = |E[y_true − y_pred]| par station (filtre n ≥ 30). Barres normalisées par le |biais| max affiché.
                  </div>
                </>
              );
            })()}
          </div>
        </section>
      </main>
    </div>
  );
}