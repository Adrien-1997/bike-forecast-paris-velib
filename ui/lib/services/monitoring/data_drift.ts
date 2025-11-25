// ui/lib/services/monitoring/data_drift.ts
//
// =============================================================================
// Service front pour la page /monitoring/data/drift.
//
// Rôle :
// - Fournir des fonctions typées pour appeler les endpoints de monitoring
//   "Data Drift" côté API, avec gestion d’ETag via `fetchJsonWithEtag`.
// - Centraliser les types utilisés par l’UI : résumé global, PSI par feature,
//   KS par feature, deltas de moyenne/variance, courbe EMA, zones spatiales.
//
// Convention :
// - Tous les appels retournent une promesse typée (`Promise<T>`), avec `T`
//   correspondant exactement à la forme JSON servie par l’API.
// - Le helper `path()` préfixe systématiquement les routes par
//   `/monitoring/data/drift` pour réduire les risques d’incohérence.
// =============================================================================

import { fetchJsonWithEtag } from "@/lib/http";

/* ───────────────────────── Types ───────────────────────── */

/**
 * Résumé global de la dérive :
 * - version du schéma JSON,
 * - timestamp de génération des artefacts,
 * - PSI global (synthèse) sur une fenêtre donnée,
 * - feature la plus "dérivante" (PSI max) et sa valeur.
 */
export type DriftSummary = {
  schema_version: string;
  generated_at: string;
  psi_global: number | null;
  top_feature?: string | null;
  top_feature_psi?: number | null;
};

/**
 * PSI (Population Stability Index) par variable.
 */
export type RowPSI = { feature: string; psi: number | null };

/**
 * KS (Kolmogorov-Smirnov) par variable.
 */
export type RowKS = { feature: string; ks: number | null };

/**
 * Deltas de moyenne/variance entre train et production.
 * - delta_mean : différence de moyenne,
 * - delta_var  : différence de variance.
 */
export type RowDelta = {
  feature: string;
  delta_mean: number | null;
  delta_var: number | null;
};

/**
 * Point d’EMA (moyenne exponentielle) journalière du PSI global.
 * Utilisé pour tracer l’évolution de la dérive dans le temps.
 */
export type EmaPoint = {
  date_local: string;
  psi_ema: number | null;
};

/**
 * Documentation de la dérive par "zones" (spatiales, clusters, etc.).
 * Chaque ligne représente une zone géographique ou logique.
 */
export type ZonesDoc = {
  rows: Array<{
    zone: string | null;
    psi: number | null;
    lat?: number | null;
    lon?: number | null;
  }>;
};

/* ───────────────────────── Helpers ───────────────────────── */

/**
 * Helper pour construire le chemin d’endpoint relatif à
 * `/monitoring/data/drift`.
 *
 * Exemple :
 *   path("")            → "/monitoring/data/drift"
 *   path("/zones")      → "/monitoring/data/drift/zones"
 */
const path = (suffix: string) => `/monitoring/data/drift${suffix}`;

/* ───────────────────────── API (ETag) ───────────────────────── */
/**
 * Résumé global de la dérive (PSI global, feature top, timestamp).
 * GET /monitoring/data/drift
 */
export const getDataDriftSummary = () =>
  fetchJsonWithEtag<DriftSummary>(path(""));

/**
 * PSI par feature.
 * GET /monitoring/data/drift/psi_by_feature
 */
export const getDataDriftPsiByFeature = () =>
  fetchJsonWithEtag<RowPSI[]>(path("/psi_by_feature"));

/**
 * KS par feature.
 * GET /monitoring/data/drift/ks_by_feature
 */
export const getDataDriftKsByFeature = () =>
  fetchJsonWithEtag<RowKS[]>(path("/ks_by_feature"));

/**
 * Deltas de moyenne/variance par feature.
 * GET /monitoring/data/drift/deltas_by_feature
 */
export const getDataDriftDeltasByFeature = () =>
  fetchJsonWithEtag<RowDelta[]>(path("/deltas_by_feature"));

/**
 * EMA journalière du PSI global.
 * GET /monitoring/data/drift/psi_global_daily_ema
 */
export const getDataDriftPsiGlobalDailyEma = () =>
  fetchJsonWithEtag<EmaPoint[]>(path("/psi_global_daily_ema"));

/**
 * Dérive par zones (spatiales, clusters…).
 * GET /monitoring/data/drift/zones
 */
export const getDataDriftZones = () =>
  fetchJsonWithEtag<ZonesDoc>(path("/zones"));
