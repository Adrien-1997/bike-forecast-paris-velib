// ui/lib/services/forecast.ts
// Centralized forecast service: GET-only + normalization

import { json as httpJson, selectForecastRows } from '@/lib/http'
import type { Forecast } from '@/lib/types/types'

// ───────────────────────────────
// Safe JSON fetch wrapper
// ───────────────────────────────

/**
 * Safe JSON fetch:
 * - returns fallback [] on non-JSON / empty body
 * - retries by parsing .text() if content-type is wrong
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
 * Fetch the latest forecast for a given horizon (15 or 60).
 * Returns the normalized array of rows.
 */
export async function getLatestForecast(h = 15): Promise<Forecast[]> {
  const payload = await safeJson<any>(`/forecast/latest?h=${h}`, { method: 'GET' })
  const rows = selectForecastRows(payload, h)
  return Array.isArray(rows) ? (rows as Forecast[]) : []
}

/**
 * Fetch latest forecast once and filter client-side by station_ids (if provided).
 * This replaces the old POST /forecast/batch behavior.
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
