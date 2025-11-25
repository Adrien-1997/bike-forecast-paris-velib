// ui/lib/services/monitoring/data_health.ts
// =============================================================================
// Service de monitoring — Santé des données
// -----------------------------------------------------------------------------
// Rôle : fournir à la page /monitoring/data/health toutes les données nécessaires
// via des appels HTTP typés vers le backend de monitoring.
// 
// Endpoints consommés :
//   - /monitoring/data/health/kpis            → DataKpis
//   - /monitoring/data/health/station_health  → StationHealthRow[]
//   - /monitoring/data/health/coverage_by_hour→ CoverageByHourRow[]
//   - /monitoring/data/health/anomalies       → Anomaly[]
//   - /monitoring/data/health/alerts          → AlertsDoc
//
// Les détails d’authentification (token, baseUrl, ETag, etc.) sont gérés dans
// lib/http.ts via fetchJsonWithEtag. Ce fichier se contente de typer et de
// exposer des helpers métiers pour la couche UI.
// =============================================================================

import { fetchJsonWithEtag } from "@/lib/http";

/* ───────────────────────── Types ───────────────────────── */

/**
 * KPI globaux de "santé" des données sur la fenêtre courante.
 * La plupart des champs sont optionnels pour tolérer des versions plus
 * anciennes du backend ou des champs non remplis.
 */
export type DataKpis = {
  /** Version de schéma de ce document (pour compat UI / backend). */
  schema_version: string;
  /** Horodatage de génération (UTC, ISO8601). */
  generated_at: string;
  /** Nombre total de lignes (bins × stations) dans la fenêtre courante. */
  rows: number;
  /** Nombre de stations distinctes couvertes. */
  stations: number;
  /** Intervalle temporel (UTC) couvert par les données (min, max). */
  span: [string | null, string | null];
  /** Pas de temps (bin) en minutes. */
  bin_min: number;
  /** Longueur de la fenêtre courante en jours (current_days). */
  current_days: number;
  /** Timezone utilisée côté backend pour les agrégations locales. */
  tz: string;

  /** Horodatage "now" côté backend (UTC, ISO8601). */
  now_utc?: string | null;
  /** Horodatage du dernier bin observé (UTC). */
  ts_global_max?: string | null;

  /** Fraîcheur médiane (p50) en minutes (now_utc - ts_global_max station). */
  freshness_age_p50_min?: number | null;
  /** Fraîcheur p95 en minutes. */
  freshness_age_p95_min?: number | null;
  /** SLO de fraîcheur (en minutes) configuré côté backend. */
  freshness_slo_min?: number | null;
  /** true si freshness_age_p95_min <= freshness_slo_min. */
  freshness_p95_ok?: boolean | null;

  /** Couverture moyenne globale (0..100, en pourcentage de bins présents). */
  coverage_global_pct?: number | null; // 0..100

  /** Latence médiane d’arrivée des données (minutes). */
  latency_p50_min?: number | null;
  /** Latence p95 (minutes). */
  latency_p95_min?: number | null;

  /** Part de doublons détectés (0..100, pourcentage). */
  dups_pct?: number | null; // 0..100

  /** Nombre de stations complètement manquantes dans la fenêtre. */
  missing_stations?: number | null;

  /** Seuils utilisés côté backend pour certaines alertes. */
  thresholds?: {
    /** SLO de fraîcheur (en minutes). */
    fresh_slo_min?: number;
    /** Seuil de couverture (%) sous lequel on déclenche une alerte. */
    compl_alert_pct?: number;
    /** Seuil de doublons (%) déclenchant une alerte. */
    dup_alert_pct?: number;
    /** Longueur (en pas) à partir de laquelle une séquence plate est signalée. */
    flat_steps?: number;
  };

  /**
   * Rappel des alertes synthétiques (même structure que AlertsDoc),
   * répliquées ici pour un accès direct depuis le même document.
   */
  alerts?: {
    freshness_p95_ok?: boolean | null;
    coverage_ok?: boolean;
    duplication_alert?: boolean;
    flat_sequences_found?: boolean;
  };
};

/**
 * Vue "santé" par station :
 *   - complétude (coverage_pct),
 *   - nombre de bins observés / attendus,
 *   - indications pour les stations les plus dégradées.
 */
export type StationHealthRow = {
  /** Identifiant de station (stringifié). */
  station_id: string;
  /** Nom lisible de la station (facultatif). */
  name?: string;
  /** Nombre de bins effectivement observés (peut être manquant). */
  obs?: number;
  /** Nombre de bins attendus sur la fenêtre. */
  expected?: number;
  /** Complétude (%) sur la fenêtre (0..100). */
  coverage_pct?: number; // 0..100
  /** Nombre de bins manquants (expected - obs). */
  missing?: number;
};

/**
 * Couverture moyenne par heure locale.
 * Utilisé pour tracer "couverture vs heure" sur la fenêtre courante.
 */
export type CoverageByHourRow = {
  /** Heure locale (0..23). */
  hour: number; // 0..23 (local)
  /** Couverture moyenne pour cette heure (0..100). */
  coverage_pct: number; // 0..100
};

/**
 * Anomalies détectées dans la fenêtre courante.
 * Trois familles :
 *   - "flat_sequence" : valeurs constantes pendant trop longtemps,
 *   - "duplicates"    : doublons (bins répétés pour une même station),
 *   - "missing_bins"  : bins manquants par rapport au calendrier attendu.
 */
export type Anomaly =
  | {
      type: "flat_sequence";
      station_id: string;
      name?: string;
      /** Début de la séquence plate (UTC, ISO8601). */
      start: string;
      /** Fin de la séquence plate (UTC, ISO8601). */
      end: string;
      /** Longueur de la séquence en nombre de pas. */
      steps: number;
      /** Durée en minutes (steps × bin_min). */
      duration_min: number;
    }
  | {
      type: "duplicates";
      station_id: string;
      name?: string;
      /** Nombre de doublons détectés sur la fenêtre. */
      dups: number;
    }
  | {
      type: "missing_bins";
      station_id: string;
      name?: string;
      /** Nombre de bins manquants. */
      missing: number;
      /** Nombre de bins attendus. */
      expected: number;
    };

/**
 * Document d’alertes agrégées :
 * flags globaux dérivés des KPI pour savoir si on est "dans le vert".
 */
export type AlertsDoc = {
  /** Fraîcheur p95 dans le SLO (true) ou non (false). */
  freshness_p95_ok?: boolean | null;
  /** Couverture globale considérée comme correcte (true) ou basse (false). */
  coverage_ok?: boolean;
  /** true si un niveau de doublons anormal est détecté. */
  duplication_alert?: boolean;
  /** true si au moins une séquence plate suspecte a été trouvée. */
  flat_sequences_found?: boolean;
};

/* ───────────────────────── Helpers ───────────────────────── */

/**
 * Construit le chemin relatif pour un suffixe donné.
 * Exemple : path("/kpis") → "/monitoring/data/health/kpis"
 */
const path = (suffix: string) => `/monitoring/data/health${suffix}`;

/**
 * Certains endpoints peuvent renvoyer un objet enveloppé
 * (ex: {kpis: {...}}). Ce helper uniformise la forme en extrayant
 * la clé demandée ou en renvoyant l’objet tel quel si la clé n’existe pas.
 */
function unwrap<T = unknown>(obj: any, key: string): T {
  return (obj?.[key] ?? obj) as T;
}

/* ───────────────────────── API calls (ETag) ───────────────────────── */

/**
 * Récupère les KPI globaux de santé des données.
 * Endpoint : GET /monitoring/data/health/kpis
 */
export async function getDataHealthKpis(): Promise<DataKpis> {
  const raw = await fetchJsonWithEtag<any>(path("/kpis"));
  return unwrap<DataKpis>(raw, "kpis");
}

/**
 * Récupère la vue "santé" par station :
 * complétude, bins observés/attendus, etc.
 * Endpoint : GET /monitoring/data/health/station_health
 */
export async function getDataHealthStationHealth(): Promise<StationHealthRow[]> {
  const raw = await fetchJsonWithEtag<any>(path("/station_health"));
  return unwrap<StationHealthRow[]>(raw, "station_health");
}

/**
 * Récupère la couverture moyenne par heure locale.
 * Endpoint : GET /monitoring/data/health/coverage_by_hour
 */
export async function getDataHealthCoverageByHour(): Promise<CoverageByHourRow[]> {
  const raw = await fetchJsonWithEtag<any>(path("/coverage_by_hour"));
  return unwrap<CoverageByHourRow[]>(raw, "coverage_by_hour");
}

/**
 * Récupère la liste des anomalies détectées sur la fenêtre courante.
 * Endpoint : GET /monitoring/data/health/anomalies
 */
export async function getDataHealthAnomalies(): Promise<Anomaly[]> {
  const raw = await fetchJsonWithEtag<any>(path("/anomalies"));
  return unwrap<Anomaly[]>(raw, "anomalies");
}

/**
 * Récupère les alertes agrégées de santé des données (SLO, couverture, etc.).
 * Endpoint : GET /monitoring/data/health/alerts
 */
export async function getDataHealthAlerts(): Promise<AlertsDoc> {
  const raw = await fetchJsonWithEtag<any>(path("/alerts"));
  return unwrap<AlertsDoc>(raw, "alerts");
}
