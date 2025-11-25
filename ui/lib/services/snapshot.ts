// ui/lib/services/snapshot.ts
//
// =============================================================================
// Service léger pour récupérer le snapshot "live" côté frontend.
//
// Rôle :
// - Interroger l’endpoint `/snapshot?mode=latest` exposé par l’API.
// - Renvoyer tel quel le payload JSON brut (météo + éventuels autres champs).
// - Fournir un point d’entrée unique pour tous les composants qui ont
//   besoin d’un "instantané" de l’état courant (ex : météo dans les badges).
//
// Contraintes :
// - Aucun typage strict ici : la forme exacte du snapshot est laissée au
//   backend (d’où le `Promise<any>`).
// - Aucune transformation métier : ce module ne fait que "proxy" l’API.
// - Gestion d’erreur défensive : en cas de problème réseau ou parsing,
//   la fonction loggue un warning et retourne `null`.
// =============================================================================

import { json } from '@/lib/http';

/**
 * Récupère le snapshot live directement depuis l’API.
 *
 * Endpoint :
 *   GET /snapshot?mode=latest
 *
 * Contenu attendu (exemple typique, non exhaustif) :
 * {
 *   "ts_utc": "2025-10-05T15:45:00Z",
 *   "temp_C": 17.3,
 *   "precip_mm": 0.2,
 *   "wind_mps": 3.1,
 *   ...
 * }
 *
 * Retour :
 * - En cas de succès : l’objet JSON tel que renvoyé par l’API (type `any`).
 * - En cas d’erreur (réseau, JSON invalide, etc.) :
 *     • log dans la console : `[getSnapshot] failed`
 *     • valeur de retour : `null`
 *
 * Remarque :
 * - Les appels consommateurs doivent toujours tolérer `null` et des champs
 *   optionnels / manquants dans le snapshot.
 */
export async function getSnapshot(): Promise<any> {
  try {
    return await json<any>('/snapshot?mode=latest');
  } catch (err) {
    console.warn('[getSnapshot] failed', err);
    return null;
  }
}
