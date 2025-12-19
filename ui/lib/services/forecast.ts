// ui/lib/services/forecast.ts
//
// =============================================================================
// Service centralisé pour les prévisions Vélib’ (API `/forecast`).
//
// Rôle principal :
// - Fournir une façade unique côté frontend pour récupérer les prévisions
//   de vélos via des appels **GET** uniquement (les anciens POST sont retirés).
// - Normaliser les réponses JSON du backend pour toujours exposer un
//   tableau typé `Forecast[]` à l’UI.
// - Déléguer la sélection des lignes (horizon, structure) au helper
//   `selectForecastRows` de `lib/http`.
//
// Ce module est utilisé par :
// - la page principale de l’app (liste + carte des stations),
// - les composants qui consomment des prévisions filtrées par station,
// - certains écrans de monitoring qui réemploient le type `Forecast`.
//
// Contraintes :
// - Aucune logique d’affichage (pas de React, pas de formatage UI).
// - Toute la logique réseau passe par `lib/http` (baseURL, token, gestion
//   d’erreurs HTTP standard).
// - Le code doit rester tolérant à des réponses JSON légèrement non standard
//   (content-type incorrect, body vide, etc.) grâce à `safeJson`.
// =============================================================================

import { json as httpJson, selectForecastRows } from '@/lib/http'
import type { Forecast } from '@/lib/types/types'

// ───────────────────────────────
// Safe JSON fetch wrapper
// ───────────────────────────────

/**
 * Safe JSON fetch utilitaire.
 *
 * Comportement :
 * - tente d’abord `httpJson<T>(url, init)` (wrapper standard de `fetch`);
 * - en cas d’échec (erreur réseau, content-type incorrect, body vide), refait
 *   un `fetch` brut et essaie de parser `res.text()` avec `JSON.parse(...)`;
 * - si tout échoue, renvoie `fallback` (par défaut un tableau vide).
 *
 * Objectif :
 * - éviter qu’un incident ponctuel côté API (content-type mal configuré,
 *   JSON vide, etc.) ne casse toute la page ;
 * - garder un point unique pour ce comportement "defensif".
 */
async function safeJson<T = any>(
  url: string,
  init?: RequestInit,
  fallback: T = [] as unknown as T
): Promise<T> {
  try {
    return await httpJson<T>(url, init)
  } catch {
    try {
      const res = await fetch(url, {
        ...init,
        headers: {
          'content-type': 'application/json',
          ...(init?.headers || {}),
        },
      })
      const txt = await res.text().catch(() => '')
      if (!txt) return fallback
      try {
        return JSON.parse(txt) as T
      } catch {
        return fallback
      }
    } catch {
      return fallback
    }
  }
}

// ───────────────────────────────
// Core: GET-only forecast
// ───────────────────────────────

/**
 * Récupère la dernière prévision disponible pour un horizon donné.
 *
 * Paramètres :
 * - `h` : horizon en minutes (typiquement 15 ou 60).
 *
 * Détails :
 * - appelle l’endpoint `/serving/forecast?h=${h}` en GET ;
 * - passe la réponse brute à `selectForecastRows(payload, h)` qui :
 *     • sait extraire le bon tableau de lignes selon la forme JSON courante,
 *     • normalise la structure pour coller au type `Forecast`;
 * - en cas de réponse non exploitable, renvoie un tableau vide.
 *
 * Retour :
 * - `Promise<Forecast[]>` toujours défini (jamais `null` ou `undefined`).
 */
export async function getLatestForecast(h = 60): Promise<Forecast[]> {
  const payload = await safeJson<any>(`/serving/forecast/?h=${h}`, { method: 'GET' })
  const rows = selectForecastRows(payload, h)
  return Array.isArray(rows) ? (rows as Forecast[]) : []
}

/**
 * Récupère la dernière prévision pour un horizon donné, puis filtre
 * côté client sur un sous-ensemble de stations.
 *
 * Rôle :
 * - remplace l’ancien endpoint `POST /forecast/batch` (déprécié) ;
 * - permet à l’UI de ne faire qu’un seul appel réseau pour un horizon donné,
 *   puis d’appliquer un filtrage local par `station_id`.
 *
 * Paramètres :
 * - `stationIds` :
 *     • `null` / `undefined` / tableau vide → renvoie **toutes** les stations
 *       de la prévision (comportement “full batch”) ;
 *     • tableau non vide → ne garde que les lignes dont l’identifiant
 *       de station correspond (après `String(...)`).
 * - `h` : horizon en minutes (par défaut `15`), transmis à `getLatestForecast`.
 *
 * Détails d’implémentation :
 * - s’appuie sur `getLatestForecast(h)` pour récupérer le batch complet ;
 * - normalise l’identifiant de station via :
 *       (r as any)?.station_id ?? (r as any)?.stationcode
 *   afin de rester compatible avec d’anciens schémas de payload.
 *
 * Retour :
 * - un tableau `Forecast[]` directement exploitable par l’UI (liste, carte).
 */
export async function getForecastFiltered(
  stationIds: string[] | null | undefined,
  h = 15
): Promise<Forecast[]> {
  const all = await getLatestForecast(h)
  if (!stationIds?.length) return all
  const wanted = new Set(stationIds.map(String))
  return all.filter(r => {
    const sid = (r as any)?.station_id ?? (r as any)?.stationcode
    return sid != null && wanted.has(String(sid))
  })
}
