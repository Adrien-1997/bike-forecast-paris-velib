// ui/lib/services/badges.ts
//
// =============================================================================
// Service front pour la construction des "badges" d’en-tête (bandeau).
//
// Rôle :
// - Construire côté client un petit payload sérialisable, déjà prêt pour l’UI,
//   à partir :
//     • d’un snapshot météo (facultatif),
//     • de timestamps fournis par le backend (fraîcheur des données,
//       heure de run du modèle, heure cible).
// - Unifier l’ancien flux (legacy : tsA = predTsISO seul) et le nouveau flux
//   (tsA = dataLatestISO, tsB = predTsISO) dans une seule fonction.
// - Centraliser le calcul de la fraîcheur (minutes depuis dataLatestISO).
//
// Contraintes :
// - Aucune requête HTTP : tout est construit localement à partir des props.
// - Le résultat est volontairement petit, stable et sérialisable (JSON-safe).
// - Idéalement, seule cette fonction est utilisée par les composants d’UI
//   pour afficher : météo, fraîcheur, horodatages utiles, etc.
// =============================================================================

/**
 * Minimal shape for weather coming from your snapshot.
 *
 * Exemple de payload (côté snapshot) :
 * {
 *   ts_utc: "2025-10-05T15:45:00Z",
 *   temp_C: 17.3,
 *   precip_mm: 0.2,
 *   wind_mps: 3.1
 * }
 *
 * Notes :
 * - Certains snapshots utilisent `tbin_utc` plutôt que `ts_utc`.
 * - Tous les champs sont optionnels pour rester tolérant aux variations
 *   de schéma et aux données manquantes.
 */
export type SnapshotWeather = {
  ts_utc?: string | null;
  tbin_utc?: string | null;  // some snapshots use tbin_utc
  temp_C?: number | null;
  precip_mm?: number | null;
  wind_mps?: number | null;
} | null | undefined;

/**
 * Construit le payload pour les badges (météo + fraîcheur + méta).
 *
 * Params (rétro-compatibles) :
 *  - weather:
 *      météo du snapshot, telle que fournie par l’API de snapshot.
 *
 *  - tsA?:
 *      • si tsB est fourni → tsA = dataLatestISO (fraîcheur des données, nouveau flux)
 *      • sinon (legacy)     → tsA = predTsISO (heure de run du modèle)
 *
 *  - tsB?:
 *      predTsISO, heure de run du modèle (nouveau flux uniquement).
 *
 *  - targetISO?:
 *      horodatage cible de la prévision (optionnel, purement informatif).
 *
 * Recommandation d’usage :
 *  - Nouveau flux :
 *      computeBadges(weather, dataLatestISO, predTsISO, targetISO)
 *  - Legacy (sans dataLatestISO séparé) :
 *      computeBadges(weather, predTsISO)
 *
 * Retour :
 *  {
 *    weather: {
 *      ts_utc: string | null;   // ts_utc ou tbin_utc normalisé
 *      temp_C: number | null;
 *      precip_mm: number | null;
 *      wind_mps: number | null;
 *    } | null;
 *
 *    freshness: {
 *      data_latest_utc: string | null; // ISO normalisé
 *      age_minutes: number | null;     // minutes entières depuis dataLatest
 *    } | null;
 *
 *    meta: {
 *      pred_ts_utc: string | null;     // heure de génération du modèle
 *      target_ts_utc: string | null;   // heure cible de la prévision
 *      freshness_min: number | null;   // alias pratique de age_minutes
 *      updated_at: string | null;      // meilleur candidat "last updated"
 *    };
 *  }
 */
export function computeBadges(
  weather?: SnapshotWeather,
  tsA?: string | null,            // dataLatestISO (nouveau) OU predTsISO (legacy)
  tsB?: string | null,            // predTsISO si fourni
  targetISO?: string | null       // optionnel
) {
  // Détermination des timestamps selon usage
  const hasTsB = typeof tsB === "string" && tsB.trim().length > 0;
  const dataLatestISO = hasTsB ? (tsA ?? null) : null; // nouveau flux : tsA = dataLatest
  const predTsISO = hasTsB ? (tsB ?? null) : (tsA ?? null); // legacy : tsA = pred_ts

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
      pred_ts_utc: toISOorNull(predTsISO), // heure de génération du modèle (informatif)
      target_ts_utc: toISOorNull(targetISO), // heure cible de la prévision (informatif)
      freshness_min: ageMin, // alias pratique
      updated_at:
        toISOorNull(dataLatestISO) ||
        toISOorNull(predTsISO) ||
        weatherTs ||
        null,
    },
  };
}

/* ----------------- utils ----------------- */

/**
 * Cast prudent vers `number | null`.
 *
 * - Accepte un `number` fini tel quel.
 * - Accepte un `string` numérique (ex: "12.3") et tente un `Number(...)`.
 * - Retourne `null` pour tout ce qui n’est ni un nombre, ni une chaîne
 *   convertible proprement en nombre fini.
 */
function safeNum(x: unknown): number | null {
  if (typeof x === "number" && Number.isFinite(x)) return x;
  if (typeof x === "string" && x.trim() !== "") {
    const n = Number(x);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

/**
 * Normalise une chaîne ISO et retourne `null` si la date est invalide.
 *
 * - Ajoute un `Z` final si absent (assume UTC).
 * - Utilise `Date.parse` pour valider.
 * - Retourne une ISO string canonique (`toISOString()`).
 */
function toISOorNull(iso?: string | null): string | null {
  if (!iso || typeof iso !== "string" || iso.trim() === "") return null;
  const t = Date.parse(ensureZ(iso));
  return Number.isNaN(t) ? null : new Date(t).toISOString();
}

/**
 * S’assure que la chaîne ISO se termine par "Z".
 *
 * Utile quand le backend renvoie parfois des timestamps sans suffixe
 * explicite de fuseau (on normalise en UTC).
 */
function ensureZ(iso: string): string {
  return iso.endsWith("Z") ? iso : `${iso}Z`;
}

/**
 * Retourne le nombre de minutes entières écoulées depuis l’instant now()
 * jusqu’au timestamp donné (en UTC).
 *
 * - Retourne `null` si le timestamp est manquant ou invalide.
 * - Clamp à `>= 0` pour éviter les valeurs négatives (horloge locale en avance).
 */
function minutesSinceUTC(iso?: string | null): number | null {
  if (!iso || typeof iso !== "string" || iso.trim() === "") return null;
  const t = Date.parse(ensureZ(iso));
  if (Number.isNaN(t)) return null;
  const diffMs = Date.now() - t;
  return Math.max(0, Math.round(diffMs / 60000)); // minutes entières
}
