// ui/lib/services/weather.ts
//
// =============================================================================
// Service frontend minimal pour récupérer la **météo en temps quasi-réel**.
//
// Rôle :
// - Appeler l’endpoint `/weather/live` exposé par l’API backend.
// - Renvoyer un objet typé `LiveWeather` (ou `null` en cas d’échec).
// - Servir de point d’entrée unique pour tous les composants qui affichent
//   la météo "live" (badges, bandeau d’info, etc.).
//
// Contraintes :
// - Aucun post-traitement métier : ce service ne fait que proxifier l’API.
// - Tolérance maximale aux erreurs réseau / JSON : on log un warning et on
//   renvoie `null` sans casser l’UI.
// - Le type `LiveWeather` reste volontairement lâche (champs optionnels)
//   pour supporter les évolutions de schéma côté backend.
// =============================================================================

import { json } from '@/lib/http';

/**
 * Météo "live" telle que renvoyée par l’API `/weather/live`.
 *
 * Champs typiques attendus :
 * - `ts_utc`   : timestamp ISO de la mesure (UTC).
 * - `temp_C`   : température en degrés Celsius.
 * - `precip_mm`: précipitations instantanées / récentes (millimètres).
 * - `wind_mps` : vitesse du vent en mètres par seconde.
 *
 * Tous les champs sont optionnels et nullables pour rester robustes
 * aux données incomplètes ou à des changements côté backend.
 */
export type LiveWeather = {
  ts_utc?: string | null;
  temp_C?: number | null;
  precip_mm?: number | null;
  wind_mps?: number | null;
} | null;

/**
 * Récupère la météo live via l’endpoint `/weather/live`.
 *
 * Retour :
 * - En cas de succès : l’objet `LiveWeather` tel que renvoyé par l’API.
 * - En cas d’erreur (réseau, JSON invalide, etc.) :
 *     • log dans la console : `[getWeather] failed`
 *     • valeur de retour : `null`
 *
 * Les composants consommateurs doivent donc toujours tolérer `null`
 * et des champs internes potentiellement absents.
 */
export async function getWeather(): Promise<LiveWeather> {
  try {
    return await json<LiveWeather>('/serving/weather');
  } catch (e) {
    console.warn('[getWeather] failed', e);
    return null;
  }
}
