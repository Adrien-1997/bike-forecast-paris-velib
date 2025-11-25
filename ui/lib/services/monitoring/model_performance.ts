// ui/lib/services/monitoring/model_performance.ts
// Service pour /monitoring/model/performance (ETag + base/token gérés par lib/http)
//
// Rôle : exposer des helpers typés pour récupérer toutes les données de la
// section "Performance modèle" du Monitoring. Ce module ne gère pas l’UI,
// uniquement le contrat de données avec le backend.
//
// Endpoints consommés :
//   - /monitoring/model/performance/manifest
//   - /monitoring/model/performance/kpis?h=...
//   - /monitoring/model/performance/daily_metrics?h=...
//   - /monitoring/model/performance/by_hour?h=...
//   - /monitoring/model/performance/by_dow?h=...
//   - /monitoring/model/performance/by_station?h=...
//   - /monitoring/model/performance/lift_curve?h=...
//   - /monitoring/model/performance/hist_residuals?h=...
//   - /monitoring/model/performance/station_timeseries?h=...&at=...
//
// Les détails HTTP (base URL, token, ETag, gestion d’erreurs) sont délégués
// à fetchJsonWithEtag dans lib/http.

import { fetchJsonWithEtag } from "@/lib/http";

/* ── Types ─────────────────────────────────────────────── */

/**
 * Manifest global de la zone "performance" :
 *   - permet de connaître la fenêtre temporelle couverte,
 *   - liste les horizons disponibles,
 *   - fournit le préfixe "latest" utilisé côté backend.
 */
export type Manifest = {
  schema_version: string;
  generated_at: string;
  latest_prefix: string;
  window_days: number;
  horizons: number[];
};

/**
 * KPIs globaux de performance pour un horizon donné.
 * Tous les indicateurs sont calculés sur la même fenêtre temporelle.
 * La plupart sont optionnels (null) pour rester robustes à l’évolution
 * du backend.
 */
export type KPIs = {
  schema_version: string;
  generated_at: string;
  window_days: number;
  horizon_min: number | null;
  /** Couverture des prédictions (0..100, % de lignes avec prédiction). */
  coverage_pred_pct: number | null;
  /** MAE du modèle. */
  mae_model: number | null;
  /** RMSE du modèle. */
  rmse_model: number | null;
  /** Biais moyen du modèle (ME). */
  me_model: number | null;
  /** MAE de la baseline (persistance). */
  mae_baseline: number | null;
  /** RMSE de la baseline. */
  rmse_baseline: number | null;
  /** Biais moyen de la baseline. */
  me_baseline: number | null;
  /**
   * Gain vs baseline : (score_baseline - score_model) / score_baseline
   * attendu dans [0,1] (null si non calculable).
   */
  lift_vs_baseline: number | null; // 0..1
  /** Nombre de lignes utilisées. */
  n_rows: number;
  /** Nombre de stations distinctes. */
  n_stations: number;
  /** Début de la fenêtre (UTC). */
  ts_min_utc: string | null;
  /** Fin de la fenêtre (UTC). */
  ts_max_utc: string | null;
};

/**
 * Métriques agrégées par jour (date locale).
 */
export type DailyRow = {
  date: string;
  mae_model: number | null;
  mae_baseline: number | null;
  rmse_model: number | null;
  rmse_baseline: number | null;
  coverage_pred_pct: number | null;
  lift_vs_baseline: number | null;
  n: number;
};

/**
 * Document de performance quotidienne (par horizon).
 */
export type DailyMetrics = { schema_version: string; horizon_min: number; rows: DailyRow[] };

/**
 * Métriques agrégées par heure locale (0..23).
 */
export type HourRow = {
  hour: number;
  mae_model: number | null;
  mae_baseline: number | null;
  coverage_pred_pct: number | null;
  n: number;
};

/**
 * Document de performance par heure (par horizon).
 */
export type ByHour = { schema_version: string; horizon_min: number; rows: HourRow[] };

/**
 * Métriques agrégées par jour de semaine (0 = lundi, 6 = dimanche).
 */
export type DOWRow = {
  dow: number;
  mae_model: number | null;
  mae_baseline: number | null;
  coverage_pred_pct: number | null;
  n: number;
};

/**
 * Document de performance par jour de semaine (par horizon).
 */
export type ByDow = { schema_version: string; horizon_min: number; rows: DOWRow[] };

/**
 * Métriques agrégées par station sur la fenêtre courante.
 */
export type StationRow = {
  station_id: string;
  mae_model: number | null;
  mae_baseline: number | null;
  coverage_pred_pct: number | null;
  n: number;
  /** Gain vs baseline pour cette station (0..1). */
  lift_vs_baseline: number | null;
};

/**
 * Document de performance par station (par horizon).
 */
export type ByStation = { schema_version: string; horizon_min: number; rows: StationRow[] };

/**
 * Courbe de lift globale vs baseline (par date).
 */
export type LiftCurve = {
  schema_version: string;
  /** Horizon associé (en minutes). */
  horizon_min?: number;
  /** Points journaliers de lift vs baseline. */
  points: Array<{ date: string; lift_vs_baseline: number | null }>;
};

/**
 * Histogramme des résidus (erreurs modèle) pour un horizon donné.
 */
export type HistResiduals = {
  schema_version: string;
  horizon_min?: number;
  /** Bords des bins. */
  bins: number[];
  /** Comptes par bin. */
  counts: number[];
  /** Nombre total de points. */
  n: number;
};

/* ⬇️ NEW — Série 24 h Observé / Modèle / Baseline pour une station */
/**
 * Série temporelle détaillée pour une station (fenêtre courte, typiquement 24h) :
 *   - ts      : timestamps (UTC),
 *   - y_true  : observations,
 *   - y_pred  : prédictions modèle,
 *   - y_base  : baseline.
 */
export type StationTimeseries = {
  schema_version: string;
  generated_at: string;
  /** Horizon utilisé (en minutes). */
  h: number;
  /** Identifiant de la station. */
  station_id: string;
  /** Nom de la station (optionnel). */
  name?: string | null;
  /** Timezone locale utilisée côté backend (ex: "Europe/Paris"). */
  tz?: string;
  ts: string[];       // ISO8601 UTC
  y_true: number[];   // Observé
  y_pred: number[];   // Modèle
  y_base: number[];   // Baseline
};

/* ── Helpers ───────────────────────────────────────────── */

/**
 * Construit le chemin relatif pour les endpoints de performance modèle.
 * Exemple : path("/kpis") → "/monitoring/model/performance/kpis"
 */
const path = (s: string) => `/monitoring/model/performance${s}`;

/* ── API (ETag) ────────────────────────────────────────── */
/**
 * Récupère le manifest global de la zone "performance".
 */
export const getPerformanceManifest = () =>
  fetchJsonWithEtag<Manifest>(path("/manifest"));

/**
 * Récupère les KPIs globaux de performance pour un horizon donné.
 */
export const getPerformanceKpis = (h: number) =>
  fetchJsonWithEtag<KPIs>(path(`/kpis?h=${encodeURIComponent(h)}`));

/**
 * Récupère les métriques quotidiennes (DailyMetrics) pour un horizon donné.
 */
export const getPerformanceDailyMetrics = (h: number) =>
  fetchJsonWithEtag<DailyMetrics>(path(`/daily_metrics?h=${encodeURIComponent(h)}`));

/**
 * Récupère les métriques agrégées par heure locale pour un horizon donné.
 */
export const getPerformanceByHour = (h: number) =>
  fetchJsonWithEtag<ByHour>(path(`/by_hour?h=${encodeURIComponent(h)}`));

/**
 * Récupère les métriques agrégées par jour de semaine pour un horizon donné.
 */
export const getPerformanceByDow = (h: number) =>
  fetchJsonWithEtag<ByDow>(path(`/by_dow?h=${encodeURIComponent(h)}`));

/**
 * Récupère les métriques agrégées par station pour un horizon donné.
 */
export const getPerformanceByStation = (h: number) =>
  fetchJsonWithEtag<ByStation>(path(`/by_station?h=${encodeURIComponent(h)}`));

/**
 * Récupère la courbe de lift globale vs baseline pour un horizon donné.
 */
export const getPerformanceLiftCurve = (h: number) =>
  fetchJsonWithEtag<LiftCurve>(path(`/lift_curve?h=${encodeURIComponent(h)}`));

/**
 * Récupère l’histogramme des résidus pour un horizon donné.
 */
export const getPerformanceHistResiduals = (h: number) =>
  fetchJsonWithEtag<HistResiduals>(path(`/hist_residuals?h=${encodeURIComponent(h)}`));

/* ⬇️ NEW */
/**
 * Récupère une série temporelle détaillée Observé / Modèle / Baseline
 * pour une station donnée, sur une fenêtre courte (ex : 24 h).
 *
 * Paramètres :
 *   - h  : horizon (minutes),
 *   - at : timestamp de référence optionnel (UTC, ISO8601) permettant
 *          de centrer la fenêtre sur un instant donné.
 */
export const getPerformanceStationTimeseries = (h: number, at?: string | null) => {
  const q = new URLSearchParams({ h: String(h) });
  if (at) q.set("at", at);
  return fetchJsonWithEtag<StationTimeseries>(path(`/station_timeseries?${q.toString()}`));
};
