// ui/lib/services/forecast.ts
// Centralized forecast service: batch fetching + normalization

import { json, selectForecastRows } from '@/lib/http'
import type { Forecast } from '@/lib/types'

// ───────────────────────────────
// Helpers
// ───────────────────────────────

/**
 * Split a large array into chunks (max n elements per chunk).
 */
const chunk = <T,>(arr: T[], n = 600) =>
  Array.from({ length: Math.ceil(arr.length / n) }, (_, i) =>
    arr.slice(i * n, i * n + n)
  )

// ───────────────────────────────
// Core batch fetcher
// ───────────────────────────────

/**
 * Fetch forecast predictions for a list of station_ids at a given horizon.
 * Calls the API in chunks of 600 stations max per request.
 *
 * @param stationIds - list of station IDs
 * @param h - forecast horizon in minutes (default 15)
 */
export async function getForecastBatch(
  stationIds: string[],
  h = 15
): Promise<Forecast[]> {
  const ids = Array.from(new Set(stationIds.map(String)))
  if (!ids.length) return []

  const parts = chunk(ids, 600)
  const out: Forecast[] = []

  for (const part of parts) {
    try {
      // ✅ POST + query ?h=...
      const payload = await json<any>(`/forecast/batch?h=${h}`, {
        method: 'POST',
        body: JSON.stringify({ station_ids: part }), // correct field name
      })

      // ✅ Normalize payload regardless of shape
      const rows = selectForecastRows(payload, h)
      if (Array.isArray(rows)) out.push(...(rows as Forecast[]))
    } catch (err) {
      console.error('[getForecastBatch] failed for chunk', err)
      throw err
    }
  }

  return out
}

// ───────────────────────────────
// Convenience single horizon fetcher
// ───────────────────────────────

/**
 * Shortcut to fetch the latest available forecast for a given horizon.
 */
export async function getLatestForecast(h = 15): Promise<Forecast[]> {
  try {
    const payload = await json<any>(`/forecast/latest?h=${h}`, { method: 'GET' })
    return selectForecastRows(payload, h)
  } catch (err) {
    console.error('[getLatestForecast] error', err)
    return []
  }
}
