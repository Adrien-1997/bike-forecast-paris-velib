// ui/lib/services/badges.ts
//
// =============================================================================
// Service front pour les "badges" d’en-tête.
//
// 🔁 Refactor API (serving)
// - L’API expose désormais un payload consolidé via `GET /serving/badges` :
//     {
//       weather: { ts_utc, temp_C, precip_mm, wind_mps } | null,
//       freshness: { forecast_generated_at, age_minutes, h? } | null,
//       meta: { updated_at, freshness_min }
//     }
// - L’UI peut donc consommer directement ce document (source of truth).
//
// Notes
// - On conserve `computeBadges()` pour compat (au cas où certaines pages
//   construisent encore localement), mais le chemin recommandé est `getBadges()`.
// =============================================================================

import { json } from '@/lib/http';

/** Weather payload returned by `/serving/badges`. */
export type BadgeWeather = {
  ts_utc?: string | null;
  temp_C?: number | null;
  precip_mm?: number | null;
  wind_mps?: number | null;
} | null;

/** Freshness payload returned by `/serving/badges`. */
export type BadgeFreshness = {
  forecast_generated_at?: string | null;
  age_minutes?: number | null;
  h?: number | null;
} | null;

/** Meta payload returned by `/serving/badges`. */
export type BadgeMeta = {
  updated_at?: string | null;
  freshness_min?: number | null;
} | null;

export type BadgesPayload = {
  weather: BadgeWeather;
  freshness: BadgeFreshness;
  meta: {
    updated_at: string | null;
    freshness_min: number | null;
  };
};

/**
 * ✅ Source of truth: fetch badges from the API.
 * Endpoint: GET `/serving/badges`
 */
export async function getBadges(): Promise<BadgesPayload | null> {
  try {
    return await json<BadgesPayload>('/serving/badges');
  } catch (e) {
    console.warn('[getBadges] failed', e);
    return null;
  }
}

/* -------------------------------------------------------------------------- */
/* Legacy / local computation (kept for compatibility)                         */
/* -------------------------------------------------------------------------- */

/**
 * Minimal shape for weather coming from older flows.
 */
export type SnapshotWeather = {
  ts_utc?: string | null;
  tbin_utc?: string | null; // some snapshots use tbin_utc
  temp_C?: number | null;
  precip_mm?: number | null;
  wind_mps?: number | null;
} | null | undefined;

/**
 * Legacy/local builder (kept).
 * Prefer using `getBadges()` which reads `/serving/badges`.
 */
export function computeBadges(
  weather?: SnapshotWeather,
  tsA?: string | null,
  tsB?: string | null,
  targetISO?: string | null
) {
  const hasTsB = typeof tsB === 'string' && tsB.trim().length > 0;
  const dataLatestISO = hasTsB ? (tsA ?? null) : null;
  const predTsISO = hasTsB ? (tsB ?? null) : (tsA ?? null);

  const ageMin = minutesSinceUTC(dataLatestISO);
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

    freshness: dataLatestISO
      ? {
          data_latest_utc: toISOorNull(dataLatestISO),
          age_minutes: ageMin,
        }
      : null,

    meta: {
      pred_ts_utc: toISOorNull(predTsISO),
      target_ts_utc: toISOorNull(targetISO),
      freshness_min: ageMin,
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
  return Math.max(0, Math.round(diffMs / 60000));
}
