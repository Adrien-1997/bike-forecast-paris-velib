// ui/lib/services/badges.ts

/**
 * Minimal shape for weather coming from your snapshot.
 * Example payload:
 * {
 *   ts_utc: "2025-10-05T15:45:00Z",
 *   temp_C: 17.3,
 *   precip_mm: 0.2,
 *   wind_mps: 3.1
 * }
 */
export type SnapshotWeather = {
  ts_utc?: string | null;
  tbin_utc?: string | null;  // some snapshots use tbin_utc
  temp_C?: number | null;
  precip_mm?: number | null;
  wind_mps?: number | null;
} | null | undefined;

/**
 * Build badges payload entirely on the client without any HTTP request.
 *
 * Params (rétro-compatibles) :
 *  - weather: météo du snapshot
 *  - tsA?: si tsB est fourni → tsA = dataLatestISO (fraîcheur des données)
 *          sinon (legacy)     → tsA = predTsISO (heure de run du modèle)
 *  - tsB?:  predTsISO (heure de run du modèle)
 *  - targetISO?: horodatage cible (optionnel, pour info)
 *
 * Reco: appelle computeBadges(weather, dataLatestISO, predTsISO, targetISO)
 */
export function computeBadges(
  weather?: SnapshotWeather,
  tsA?: string | null,            // dataLatestISO (nouveau) OU predTsISO (legacy)
  tsB?: string | null,            // predTsISO si fourni
  targetISO?: string | null       // optionnel
) {
  // Détermination des timestamps selon usage
  const hasTsB = typeof tsB === 'string' && tsB.trim().length > 0;
  const dataLatestISO = hasTsB ? (tsA ?? null) : null; // nouveau flux : tsA = dataLatest
  const predTsISO     = hasTsB ? (tsB ?? null) : (tsA ?? null); // legacy : tsA = pred_ts

  // Fraîcheur = âge basé sur dataLatestISO (tbin_latest)
  const ageMin = minutesSinceUTC(dataLatestISO);

  // Weather timestamp (pour affichage/traçage éventuel)
  const weatherTs = (weather?.ts_utc ?? weather?.tbin_utc ?? null) || null;

  return {
    weather: weather
      ? {
          ts_utc: weatherTs,
          temp_C: safeNum(weather?.temp_C),
          precip_mm: safeNum(weather?.precip_mm),
          wind_mps: safeNum(weather?.wind_mps),
        }
      : null,

    // 🟩 Fraîcheur des données (et plus celle du run modèle)
    freshness: dataLatestISO
      ? {
          data_latest_utc: toISOorNull(dataLatestISO),
          age_minutes: ageMin, // entier (minutes)
        }
      : null,

    // Métadonnées utiles pour le bandeau/tooltip
    meta: {
      pred_ts_utc: toISOorNull(predTsISO),     // heure de génération du modèle (informatif)
      target_ts_utc: toISOorNull(targetISO),   // heure cible de la prévision (informatif)
      freshness_min: ageMin,                   // alias pratique
      updated_at: toISOorNull(dataLatestISO) || toISOorNull(predTsISO) || weatherTs || null,
    },
  };
}

/* ----------------- utils ----------------- */

function safeNum(x: unknown): number | null {
  if (typeof x === 'number' && Number.isFinite(x)) return x;
  if (typeof x === 'string' && x.trim() !== '') {
    const n = Number(x);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function toISOorNull(iso?: string | null): string | null {
  if (!iso || typeof iso !== 'string' || iso.trim() === '') return null;
  const t = Date.parse(ensureZ(iso));
  return Number.isNaN(t) ? null : new Date(t).toISOString();
}

function ensureZ(iso: string): string {
  return iso.endsWith('Z') ? iso : `${iso}Z`;
}

function minutesSinceUTC(iso?: string | null): number | null {
  if (!iso || typeof iso !== 'string' || iso.trim() === '') return null;
  const t = Date.parse(ensureZ(iso));
  if (Number.isNaN(t)) return null;
  const diffMs = Date.now() - t;
  return Math.max(0, Math.round(diffMs / 60000)); // minutes entières
}