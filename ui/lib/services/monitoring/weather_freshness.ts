// ui/lib/services/monitoring/weather_freshness.ts
//
// =============================================================================
// Service front pour la page /monitoring/weather (freshness).
//
// Rôle :
// - Interroger l’endpoint backend qui expose la fraîcheur des données météo,
//   déjà agrégée côté jobs (p95 de la latence en minutes).
// - Fournir un type strict `WeatherFreshnessDoc` afin de typer
//   les composants de monitoring (KPI bar, tooltips, etc.).
//
// Contexte :
// - Les données météo proviennent d’un fournisseur externe (ex: Open-Meteo,
//   OpenWeather, etc.) et sont ingérées régulièrement côté backend.
// - Le job de monitoring calcule une mesure de "freshness" en minutes
//   (latence entre l’heure de référence et la dernière observation connue),
//   puis en extrait le percentile 95 (p95_min).
//
// Contrats :
// - L’UI ne fait **aucun calcul** ici : elle consomme uniquement le JSON
//   renvoyé par le backend.
// - Le type reste volontairement tolérant (propriétés optionnelles) afin de
//   supporter des évolutions de schéma sans casser le front.
// - Toute modification de structure JSON doit être répercutée dans ce type
//   et dans la documentation API.
// =============================================================================

import { getJSON } from "@/lib/http";

/**
 * Document de fraîcheur météo utilisé par la page de monitoring.
 *
 * Schéma JSON attendu (exemple simplifié) :
 * {
 *   "schema_version": "1.0",
 *   "generated_at": "2025-11-20T12:00:00Z",
 *   "meta": {
 *     "bin_t_utc": "2025-11-20T11:55:00Z"
 *   },
 *   "provider": {
 *     "freshness": {
 *       "p95_min": 7.5
 *     }
 *   }
 * }
 *
 * Champs :
 * - schema_version : version du schéma JSON (pour compatibilité).
 * - generated_at   : timestamp ISO UTC de génération du document.
 * - meta.bin_t_utc : borne temporelle de référence (bin météo) côté backend.
 * - provider.freshness.p95_min :
 *     percentile 95 de la latence, en minutes, entre la référence
 *     (bin_t_utc ou "now") et la dernière observation réellement disponible.
 */
export type WeatherFreshnessDoc = {
  schema_version?: string;
  generated_at?: string;
  meta?: { bin_t_utc?: string } | null;
  provider?: {
    freshness?: {
      /** Percentile 95 de la latence météo (en minutes). */
      p95_min?: number | null;
    } | null;
  } | null;
};

/**
 * Récupère le dernier document de fraîcheur météo.
 *
 * Endpoint backend :
 *   GET /monitoring/weather/freshness/latest
 *
 * Comportement :
 * - Renvoie toujours un objet (les champs internes peuvent être null/undefined).
 * - En cas d’erreur réseau, l’erreur est propagée (à gérer côté page).
 *
 * Utilisation typique :
 * - Afficher un KPI de type
 *     "Météo : P95 fraîcheur = X min"
 *   dans la barre de statuts globales.
 */
export async function getWeatherFreshnessLatest(): Promise<WeatherFreshnessDoc> {
  return getJSON<WeatherFreshnessDoc>("/monitoring/weather/freshness/latest");
}
