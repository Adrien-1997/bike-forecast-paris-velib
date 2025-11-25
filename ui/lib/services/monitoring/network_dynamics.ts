// ui/lib/services/monitoring/network_dynamics.ts
// Service pour la page /monitoring/network/dynamics
// → utilise lib/http.ts (token & base déjà gérés)
//
// Rôle de ce module :
// - exposer des fonctions typées pour récupérer les artefacts de dynamique réseau
//   (heatmaps, profils horaires, épisodes de pénurie/saturation, carte de tension);
// - masquer les détails de sérialisation (ETag, enveloppe JSON, typage).
// Les composants React consomment uniquement ces fonctions + types, jamais les
// endpoints bruts.

/* ───────────────────────── Imports ───────────────────────── */

import { fetchJsonWithEtag, getJSON } from "@/lib/http";

/* ───────────────────────── Types ───────────────────────── */
/**
 * Document combinant :
 *  - une série de heatmaps (occupation/taux de pénurie/taux de saturation),
 *  - des profils moyens d’occupation par jour de semaine.
 *
 * Conventions :
 *  - heatmap.* : matrices 2D [row][col], typiquement [jour de semaine][heure],
 *    éléments null si pas de données ;
 *  - profiles_occ_by_dow : clé = dow (string) côté backend, valeur = profil
 *    24 points (ou n-points) sur une journée.
 */
export type HeatmapsProfilesDoc = {
  schema_version: string;
  generated_at: string;
  heatmap: {
    occ_mean: (number | null)[][];
    penury_rate: (number | null)[][];
    saturation_rate: (number | null)[][];
  };
  profiles_occ_by_dow: Record<string, (number | null)[]>;
};

/**
 * Document pour les agrégations horaires globales :
 *  - une ligne par heure locale,
 *  - taux moyens de pénurie / saturation sur le réseau.
 */
export type HourlyDoc = {
  schema_version: string;
  generated_at: string;
  rows: Array<{ hour: number; penury_rate: number | null; saturation_rate: number | null }>;
};

/**
 * Séquences (épisodes) de pénurie / saturation par station sur les derniers jours.
 *
 * Un "épisode" = une séquence temporelle continue de pas de temps où la condition
 * (pénurie / saturation) est vraie.
 */
export type EpisodesDoc = {
  schema_version: string;
  generated_at: string;
  /** Fenêtre d’analyse en jours (ex: 14 derniers jours) */
  last_days: number;
  rows: Array<{
    /** Identifiant de station (stringifié côté service) */
    station_id: string;
    /** Type d’épisode : pénurie ou saturation */
    type: "penury" | "saturation";
    /** Timestamp de début (UTC, ISO 8601) */
    start_utc: string;
    /** Timestamp de fin (UTC, ISO 8601) */
    end_utc: string;
    /** Nombre de pas de temps consécutifs dans l’épisode */
    steps: number;
    /** Durée en minutes (peut être null si non calculée) */
    duration_min: number | null;
  }>;
};

/**
 * Indicateurs de "tension" par station sur une fenêtre récente.
 *
 * Pour chaque station :
 *  - taux de pénurie / saturation,
 *  - occupation moyenne,
 *  - index de tension agrégé (0..2),
 *  - nombre d’observations utilisées.
 */
export type TensionByStationDoc = {
  schema_version: string;
  generated_at: string;
  /** Fenêtre utilisée pour le calcul (en jours) */
  last_days: number;
  rows: Array<{
    station_id: string;
    name?: string | null;
    lat?: number | null;
    lon?: number | null;
    penury_rate: number | null;
    saturation_rate: number | null;
    occ_mean: number | null;
    /** Index synthétique de tension : 0 (faible) → 2 (forte) */
    tension_index: number | null; // 0..2
    /** Nombre d’observations (points spatio-temporels) agrégées */
    n_obs: number;
  }>;
};

/**
 * Métadonnées minimales de station utilisées côté frontend
 * pour labelliser les tableaux/cartes et représenter les points sur Leaflet.
 */
export type StationMeta = {
  station_id: string;
  name?: string | null;
  lat?: number | null;
  lon?: number | null;
};

/* ───────────────────────── Helpers ───────────────────────── */

/**
 * Construit le chemin relatif pour les endpoints de dynamique réseau.
 * Exemple : path("/heatmaps_profiles") → "/monitoring/network/dynamics/heatmaps_profiles"
 */
const path = (suffix: string) => `/monitoring/network/dynamics${suffix}`;

/**
 * Récupère un index minimal des stations auprès de l’API "/stations"
 * et le convertit en dictionnaire { station_id → StationMeta }.
 *
 * Note :
 *  - on cast en any pour tolérer différents schémas de réponse (legacy),
 *  - lat/lon sont normalisés en number | null.
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

/**
 * Certains endpoints peuvent renvoyer un objet enveloppé
 *   ex: { heatmaps_profiles: {...} }
 * Ces helpers "unwrap" l’enveloppe pour retourner directement le document typé.
 */
function unwrapHeatmapsProfiles(p: any): HeatmapsProfilesDoc {
  return (p?.heatmaps_profiles ?? p) as HeatmapsProfilesDoc;
}
function unwrapHourly(p: any): HourlyDoc {
  return (p?.hourly ?? p) as HourlyDoc;
}
function unwrapEpisodes(p: any): EpisodesDoc {
  return (p?.episodes ?? p) as EpisodesDoc;
}
function unwrapTension(p: any): TensionByStationDoc {
  return (p?.tension_by_station ?? p) as TensionByStationDoc;
}

/* ───────────────────────── API calls (ETag) ───────────────────────── */
/**
 * Heatmaps + profils d’occupation (document principal pour la vue "cartes + profils").
 * Utilise fetchJsonWithEtag pour bénéficier de :
 *  - cache navigateur (If-None-Match),
 *  - gestion transparente du token base URL via lib/http.
 */
export const getDynamicsHeatmapsProfiles = async () => {
  const raw = await fetchJsonWithEtag<any>(path("/heatmaps_profiles"));
  return unwrapHeatmapsProfiles(raw);
};

/**
 * Agrégations horaires (taux de pénurie / saturation par heure locale).
 */
export const getDynamicsHourlyPenSat = async () => {
  const raw = await fetchJsonWithEtag<any>(path("/hourly_pen_sat"));
  return unwrapHourly(raw);
};

/**
 * Épisodes de pénurie / saturation par station (fenêtre glissante récente).
 */
export const getDynamicsEpisodes = async () => {
  const raw = await fetchJsonWithEtag<any>(path("/episodes"));
  return unwrapEpisodes(raw);
};

/**
 * Indicateurs de tension par station (pénurie, saturation, index synthétique).
 */
export const getDynamicsTensionByStation = async () => {
  const raw = await fetchJsonWithEtag<any>(path("/tension_by_station"));
  return unwrapTension(raw);
};
