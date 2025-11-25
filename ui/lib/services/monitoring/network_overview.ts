// ui/lib/services/monitoring/network_overview.ts
//
// =============================================================================
// Service front pour la page /monitoring/network/overview.
//
// Rôle :
// - Récupérer les artefacts JSON produits par le job
//   `service/jobs/build_network_overview.py` (bucket GCS → API → UI).
// - Exposer des types stricts pour :
//     • les KPI globaux réseau,
//     • la distribution snapshot (pénurie / saturation / vélos / docks),
//     • les courbes “aujourd’hui vs médiane de référence”,
//     • les barres de comparaison J-7 / J-14 / J-21,
//     • la carte snapshot,
//     • la tension par station.
// - Fournir des helpers pour indexer les stations côté UI.
//
// Endpoints consommés :
//   GET /monitoring/network/overview/kpis
//   GET /monitoring/network/overview/snapshot_distribution
//   GET /monitoring/network/overview/today_curve
//   GET /monitoring/network/overview/ref_median_curve
//   GET /monitoring/network/overview/kpis_today_vs_lags
//   GET /monitoring/network/overview/snapshot_map
//   GET /monitoring/network/overview/stations_tension
//
// Convention :
// - Tous les appels passent par `fetchJsonWithEtag` (base URL, token, ETag).
// - Les docs sont versionnés via `schema_version` et timestampés via
//   `generated_at`.
// =============================================================================

import { fetchJsonWithEtag, getJSON } from "@/lib/http";

/* ───────────────────────── Types ───────────────────────── */

/**
 * KPI globaux de la page overview.
 *
 * Correspond à `kpis.json` (un seul document) :
 * - snapshot “du jour”,
// - agrégats sur les LAST_DAYS et REF_DAYS utilisés par le job backend.
 */
export type OverviewKpis = {
  /** Version de schéma JSON. */
  schema_version: string;
  /** Timestamp de génération (UTC, ISO). */
  generated_at: string;

  /** Timestamp du snapshot (UTC). */
  snapshot_ts_utc: string;
  /** Timestamp du snapshot en heure locale (Europe/Paris). */
  snapshot_ts_local: string;

  /** Nombre de stations dans l’univers (fenêtre d’analyse). */
  stations_universe: number;
  /** Nombre de stations actives au snapshot. */
  stations_active: number;
  /** Nombre de stations hors-ligne au snapshot. */
  stations_offline: number;

  /** Disponibilité globale en vélos (0–100). */
  availability_bike_pct: number | null;
  /** Disponibilité globale en docks libres (0–100). */
  availability_dock_pct: number | null;

  /** Taux moyen de pénurie (0–100). */
  penury_pct: number | null;
  /** Taux moyen de saturation (0–100). */
  saturation_pct: number | null;

  /** Couverture efficace (stations avec données suffisantes, 0–100). */
  coverage_pct: number | null;

  /**
   * Volatilité intra-jour (mesure synthétique de la variation d’occupation).
   * La définition exacte est côté backend, mais la valeur est normalisée
   * (souvent 0–100 ou 0–1).
   */
  volatility_today: number | null;

  /** Nombre de jours utilisés pour la fenêtre “last_days”. */
  last_days: number;
  /** Nombre de jours utilisés pour la référence “ref_days”. */
  ref_days: number;
};

/**
 * Une ligne de distribution snapshot :
 * - on projette le réseau à l’instant T dans des catégories
 *   (pénurie / saturation / vélos / docks).
 */
export type OverviewSnapshotDistributionItem = {
  /** Type de métrique du bin. */
  metric: "bike_avail" | "dock_avail" | "penury" | "saturation";
  /** Nombre de stations dans cette catégorie. */
  count: number;
  /** Nombre total de stations actives. */
  total_active: number;
  /** Part de cette catégorie dans le total (0–100). */
  pct: number; // 0..100
};

/** Distribution snapshot complète (table d’items). */
export type OverviewSnapshotDistribution = OverviewSnapshotDistributionItem[];

/**
 * Courbe “aujourd’hui” (today_curve.json) :
 * - points en 5 min,
 * - pourcentage de stations avec ≥1 vélo.
 */
export type OverviewTodayCurve = {
  schema_version: string;
  generated_at: string;
  /** Points horodatés (hh:mm local ou UTC selon backend, ici on ne tranche pas). */
  points: Array<{ hhmm: string; pct: number | null }>;
};

/**
 * Courbe de référence (ref_median_curve.json) :
 * - médiane des % de stations avec ≥1 vélo,
 * - sur REF_DAYS jours de même weekday.
 */
export type OverviewRefMedianCurve = {
  schema_version: string;
  generated_at: string;
  /** Médiane par tranche hh:mm. */
  median: Array<{ hhmm: string; pct_median: number | null }>;
};

/**
 * Comparaison “Aujourd’hui vs J-7 / J-14 / J-21” :
 * - pour chaque jour de référence, on stocke 4 KPI :
 *     • vélos disponibles,
 *     • docks disponibles,
 *     • pénurie,
//     • saturation.
 */
export type OverviewKpisTodayVsLags = {
  schema_version: string;
  generated_at: string;

  /** KPI agrégés pour le jour courant. */
  today: {
    avail_bike: number | null;
    avail_dock: number | null;
    pen: number | null;
    sat: number | null;
  };

  /**
   * Jours de référence :
   * - J-7, J-14, J-21 (relative au jour courant).
   */
  lags: {
    "J-7": OverviewKpisTodayVsLags["today"];
    "J-14": OverviewKpisTodayVsLags["today"];
    "J-21": OverviewKpisTodayVsLags["today"];
  };
};

/**
 * Carte snapshot :
 * - une ligne par station,
 * - attributs minimaux pour la carte et des badges côté UI.
 */
export type OverviewSnapshotMap = {
  schema_version: string;
  generated_at: string;
  rows: Array<{
    station_id: string;
    /** Nom public de la station. */
    name: string;
    /** Coordonnées géographiques. */
    lat: number | null;
    lon: number | null;
    /** Nombre de vélos présents. */
    bikes: number | null;
    /** Nombre de docks libres. */
    docks_avail: number | null;
    /** Station en pénurie (0/1). */
    is_penury: 0 | 1;
    /** Station en saturation (0/1). */
    is_saturation: 0 | 1;
  }>;
};

/**
 * Tension par station (sur la fenêtre LAST_DAYS) :
 * - pourcentage de temps en pénurie / saturation.
 */
export type OverviewStationsTension = {
  schema_version: string;
  generated_at: string;
  rows: Array<{
    station_id: string;
    /** Taux de pénurie (0–100) sur la fenêtre. */
    penury_rate: number | null;
    /** Taux de saturation (0–100) sur la fenêtre. */
    saturation_rate: number | null;
  }>;
};

/**
 * Métadonnées station récupérées via l’API `/stations`.
 * Utilisé côté UI pour faire la jointure id → nom / coords.
 */
export type StationMeta = {
  station_id: string;
  name?: string | null;
  lat?: number | null;
  lon?: number | null;
};

/* ───────────────────────── Helpers ───────────────────────── */

/**
 * Construit le chemin relatif à `/monitoring/network/overview`.
 *
 * Exemple :
 *   path("/kpis") → "/monitoring/network/overview/kpis"
 */
const path = (suffix: string) => `/monitoring/network/overview${suffix}`;

/**
 * Index minimal des stations (id → métadonnées).
 *
 * - Appelle une route générique `/stations` (hors /monitoring),
 * - Tolère les erreurs réseau en renvoyant un index vide,
 * - Normalise les lat/lon en nombres ou null.
 */
export async function fetchStationsIndex(): Promise<Record<string, StationMeta>> {
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

/* ───────────────────────── API calls (ETag) ───────────────────────── */

/**
 * KPIs globaux overview.
 *
 * Endpoint :
 *   GET /monitoring/network/overview/kpis
 */
export const getOverviewKpis = () =>
  fetchJsonWithEtag<OverviewKpis>(path("/kpis"));

/**
 * Distribution snapshot (pénurie / saturation / vélos / docks).
 *
 * Endpoint :
 *   GET /monitoring/network/overview/snapshot_distribution
 */
export const getOverviewSnapshotDistribution = () =>
  fetchJsonWithEtag<OverviewSnapshotDistribution>(path("/snapshot_distribution"));

/**
 * Courbe “aujourd’hui” (% stations avec ≥1 vélo par tranche 5 min).
 *
 * Endpoint :
 *   GET /monitoring/network/overview/today_curve
 */
export const getOverviewTodayCurve = () =>
  fetchJsonWithEtag<OverviewTodayCurve>(path("/today_curve"));

/**
 * Courbe “médiane de référence” sur REF_DAYS jours de même weekday.
 *
 * Endpoint :
 *   GET /monitoring/network/overview/ref_median_curve
 */
export const getOverviewRefMedianCurve = () =>
  fetchJsonWithEtag<OverviewRefMedianCurve>(path("/ref_median_curve"));

/**
 * Barres KPI “Aujourd’hui vs J-7 / J-14 / J-21”.
 *
 * Endpoint :
 *   GET /monitoring/network/overview/kpis_today_vs_lags
 */
export const getOverviewKpisTodayVsLags = () =>
  fetchJsonWithEtag<OverviewKpisTodayVsLags>(path("/kpis_today_vs_lags"));

/**
 * Carte snapshot (une ligne par station avec vélos/docks & flags).
 *
 * Endpoint :
 *   GET /monitoring/network/overview/snapshot_map
 */
export const getOverviewSnapshotMap = () =>
  fetchJsonWithEtag<OverviewSnapshotMap>(path("/snapshot_map"));

/**
 * Tension agrégée par station (pénurie / saturation sur la fenêtre).
 *
 * Endpoint :
 *   GET /monitoring/network/overview/stations_tension
 */
export const getOverviewStationsTension = () =>
  fetchJsonWithEtag<OverviewStationsTension>(path("/stations_tension"));
