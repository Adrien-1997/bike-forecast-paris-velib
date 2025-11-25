// ui/lib/services/monitoring/data_freshness.ts
//
// =============================================================================
// Service front pour la page /monitoring/data/freshness.
//
// Rôle :
// - Fournir une fonction typée pour récupérer le dernier snapshot de fraîcheur
//   des données côté API (`/monitoring/data/freshness`), avec gestion d’ETag.
// - Exposer quelques sélecteurs utilitaires pour extraire les indicateurs clés
//   (p50, p95, méta du bin) depuis le document retourné.
// - Laisser à l’UI (composants monitoring) la responsabilité d’afficher ces
//   valeurs sous forme de KPI ou de badges.
//
// Convention :
// - Les types ici reflètent la structure JSON exposée par le backend de
//   monitoring (build_data_health / freshness).
// - `fetchJsonWithEtag` gère la mise en cache ETag + token d’authentification.
// =============================================================================

import { fetchJsonWithEtag } from "@/lib/http";

/* ───────────────────────── Types ───────────────────────── */
/**
 * Document de fraîcheur de données renvoyé par l’API.
 *
 * Champs visibles dans cet extrait :
 * - now_utc  : horodatage courant au moment du build (ISO UTC).
 * - stations : bloc de fraîcheur au niveau des stations, contenant :
 *    • count     : nombre de stations considérées,
 *    • freshness : distribution de fraîcheur en minutes (p50/p95/max),
 *    • top_oldest: stations les plus "anciennes" (snapshot le plus vieux),
 *                  avec pour chacune : station_id + freshness_min.
 *
 * NB : La structure complète peut contenir d’autres champs (ex: `meta` avec
 *      la fenêtre temporelle de binning). Ils restent inchangés ici.
 */
export type FreshnessDoc = {
  now_utc: string;
  stations: {
    count: number;
    freshness: {
      p50_min: number | null;
      p95_min: number | null;
      max_min: number | null;
    };
    top_oldest?: Array<{ station_id: number | string; freshness_min: number }>;
  };
  weather?: { freshness_min: number | null };
  meta?: { bin_t_utc?: string; schema?: string };
};

/* ───────────────────────── Helpers ───────────────────────── */
/**
 * Construit le chemin d’endpoint relatif à `/monitoring/data/freshness`.
 *
 * Exemple :
 *   path()         → "/monitoring/data/freshness"
 *   path("/debug") → "/monitoring/data/freshness/debug"
 */
const path = (suffix = "") => `/monitoring/data/freshness${suffix}`;

/* ───────────────────────── API (ETag) ───────────────────────── */
/**
 * Récupère le dernier snapshot de fraîcheur des données :
 * GET /monitoring/data/freshness
 *
 * Retour :
 * - Promesse résolue avec un `FreshnessDoc` typé,
 * - gérée via `fetchJsonWithEtag` (ETag + auth transparents).
 */
export const getDataFreshnessLatest = () =>
  fetchJsonWithEtag<FreshnessDoc>(path());

/* ───────────────────────── Selectors utiles ───────────────────────── */
/**
 * Sélecteur : p95 de fraîcheur en minutes (stations).
 * - Retourne `null` si le document ou la valeur est absente.
 */
export const selectFreshnessP95 = (d?: FreshnessDoc | null) =>
  (d?.stations?.freshness?.p95_min ?? null);

/**
 * Sélecteur : p50 (médiane) de fraîcheur en minutes (stations).
 * - Retourne `null` si le document ou la valeur est absente.
 */
export const selectFreshnessP50 = (d?: FreshnessDoc | null) =>
  (d?.stations?.freshness?.p50_min ?? null);

/**
 * Sélecteur : timestamp du bin courant (UTC) si fourni dans la méta.
 * - Délègue à `d.meta.bin_t_utc` sans imposer la structure complète de `meta`.
 */
export const selectFreshnessMetaBin = (d?: FreshnessDoc | null) =>
  d?.meta?.bin_t_utc ?? null;
