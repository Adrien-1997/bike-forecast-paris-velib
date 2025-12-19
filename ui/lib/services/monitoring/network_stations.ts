// ui/lib/services/monitoring/network_stations.ts
//
// =============================================================================
// Service front pour la page /monitoring/network/stations.
//
// Rôle :
// - Récupérer les artefacts JSON produits par le backend
//   pour la page de monitoring des stations (bucket GCS → API → UI).
// - Exposer des types stricts pour :
//     • les KPI globaux de la clusterisation des stations,
//     • les centroïdes de clusters (profil type de station),
//     • les statistiques agrégées sur 7 jours par station.
// - Fournir un helper pour construire un index léger des stations
//   (id → métadonnées) utilisable partout dans l’UI (labels, carte, etc.).
//
// Contraintes :
// - Tous les fetch passent par lib/http (base URL + token déjà gérés).
// - Les types doivent rester alignés avec le schéma JSON produit par le job.
// =============================================================================

import { fetchJsonWithEtag, getJSON } from "@/lib/http";


/* ───────────────────────── Types ───────────────────────── */

/**
 * KPI globaux issus de la clusterisation des stations sur une
 * fenêtre glissante de `window_days` jours.
 *
 * Endpoint source :
 *   GET /monitoring/network/stations/kpis
 */
export type KpisDoc = {
  schema_version: string;
  generated_at: string;

  /** Nombre de stations effectivement présentes dans le jeu de données. */
  n_stations: number | null;

  /**
   * Nombre de clusters effectivement utilisés (k “effectif” après éventuels
   * merges ou filtrages backend).
   */
  k_effective: number | null;

  /**
   * Score de silhouette global de la clusterisation.
   * Plus il est élevé, plus les clusters sont bien séparés.
   */
  silhouette: number | null;

  /**
   * Indice de Davies-Bouldin global (plus il est faible, mieux c’est).
   */
  davies_bouldin: number | null;

  /**
   * Taille de la fenêtre d’agrégation (en jours) utilisée pour construire
   * ces KPI.
   */
  window_days: number;
};


/**
 * Centroïdes des clusters de stations dans l’espace de features.
 *
 * - `x_labels` : noms des dimensions (features standardisées).
 * - `centroids` : pour chaque cluster, la valeur moyenne sur chaque feature.
 *
 * Endpoint source :
 *   GET /monitoring/network/stations/centroids
 */
export type CentroidsDoc = {
  schema_version: string;
  generated_at: string;
  x_labels: string[];
  centroids: Array<{ cluster: number; y: (number | null)[] }>;
};


/**
 * Statistiques agrégées sur 7 jours par station.
 *
 * Chaque ligne correspond à une station et contient les indicateurs
 * utilisés pour la carte, la table et les clusters :
 *   - volatilité d’usage,
 *   - taux de pénurie / saturation,
 *   - couverture des données,
 *   - cluster affecté, etc.
 *
 * Endpoint source :
 *   GET /monitoring/network/stations/stats7
 */
export type Stats7Doc = {
  schema_version: string;
  generated_at: string;
  rows: Array<{
    /** Identifiant interne de la station (clé primaire). */
    station_id: string;

    /** Nom lisible de la station (si disponible côté backend). */
    name?: string | null;

    /** Latitude (WGS84) de la station. */
    lat?: number | null;

    /** Longitude (WGS84) de la station. */
    lon?: number | null;

    /**
     * Capacité estimée (nombre de bornes). Peut différer légèrement de
     * la capacité “théorique” déclarée si recalculée côté backend.
     */
    capacity_est?: number | null;

    /**
     * Mesure de volatilité d’occupation sur 7 jours (définition
     * exacte côté backend, typiquement normalisée).
     */
    volatility?: number | null;

    /** Taux de pénurie (proportion du temps avec 0 vélo disponible). */
    penury_rate?: number | null;

    /** Taux de saturation (proportion du temps avec 0 dock libre). */
    saturation_rate?: number | null;

    /** Couverture des données (proportion de bins avec données valides). */
    coverage_pct?: number | null;

    /** Cluster affecté à la station (index entier, ou null si non clusterisée). */
    cluster?: number | null;
  }>;
};


/**
 * Métadonnées minimales d’une station, suffisantes pour :
 * - afficher des libellés lisibles (nom),
 * - projeter la station sur une carte (lat/lon),
 * - donner un ordre de grandeur de capacité.
 *
 * Utilisé par `fetchStationsIndex` comme valeur dans l’index.
 */
export type StationMeta = {
  /** Identifiant de la station (stringifié). */
  station_id: string;
  /** Nom “front” de la station. */
  name?: string | null;
  /** Latitude en degrés. */
  lat?: number | null;
  /** Longitude en degrés. */
  lon?: number | null;
  /** Capacité (nombre de docks), si connue. */
  capacity?: number | null;
};



/* ───────────────────────── Helpers ───────────────────────── */

/**
 * Construit le chemin d’API pour la sous-arborescence
 * `/monitoring/network/stations`.
 *
 * On garde la logique centralisée ici pour éviter de dupliquer
 * la chaîne de base dans chaque appel.
 */
const path = (suffix: string) => `/monitoring/network/stations${suffix}`;


/**
 * Récupère un index léger des stations (id → métadonnées).
 *
 * - Appelle l’endpoint `/stations` (backend d’agrégation global).
 * - Tolère les erreurs réseau : en cas d’échec, renvoie un index vide.
 * - Normalise les types (id en string, lat/lon/capacity en number|null).
 *
 * Cet helper est pensé pour :
 *   - les cartes (Leaflet),
 *   - les listes déroulantes de sélection de station,
 *   - toute autre UI ayant besoin d’un accès rapide à la méta station.
 */
export async function fetchStationsIndex(): Promise<Record<string, StationMeta>> {
  const arr = await getJSON<StationMeta[]>("serving/stations").catch(() => []);
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


/* ───────────────────────── API calls (ETag) ───────────────────────── */

// Tous ces endpoints passent par `fetchJsonWithEtag` afin de :
// - profiter du cache HTTP (ETag) côté navigateur,
// - limiter les hits API pour les pages de monitoring,
// - garder une API typée (génériques TypeScript).

export const getStationsKpis = () =>
  fetchJsonWithEtag<KpisDoc>(path("/kpis"));

export const getStationsCentroids = () =>
  fetchJsonWithEtag<CentroidsDoc>(path("/centroids"));

export const getStationsStats7 = () =>
  fetchJsonWithEtag<Stats7Doc>(path("/stats7"));
