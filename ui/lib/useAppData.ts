// ui/lib/useAppData.ts
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { getStations } from '@/lib/services/stations'
import { getForecastBatch } from '@/lib/services/forecast'
import { getWeather } from '@/lib/services/weather'
import { computeBadges } from '@/lib/services/badges'

import type { Station, Forecast } from '@/lib/types'

/* ───────── helpers ───────── */

const toNumber = (x: unknown, fallback = 0): number => {
  const n = Number(x)
  return Number.isFinite(n) ? n : fallback
}

const getPred = (f?: any): number =>
  toNumber(f?.bikes_pred_int ?? f?.bikes_pred ?? 0, 0)

// clé de jointure potentielle (station_id ou stationcode selon ce que renvoie l’API)
const keyFor = (obj: any): string | null => {
  if (!obj) return null
  const id = obj.station_id ?? obj.stationId ?? null
  const code = obj.stationcode ?? obj.code ?? null
  if (id != null && String(id).trim() !== '') return String(id)
  if (code != null && String(code).trim() !== '') return String(code)
  return null
}

// array direct OU bundle { generated_at, horizons, data: {"15":[...]} }
function normalizeForecastRows(payload: any, horizon = 15): any[] {
  if (Array.isArray(payload)) return payload
  if (payload && payload.data) {
    const k = String(horizon)
    if (Array.isArray(payload.data[k])) return payload.data[k]
  }
  return []
}

// âge en minutes par rapport à un ISO UTC
const ageMinutes = (iso?: string | null): number | null => {
  if (!iso) return null
  const t = new Date(iso).getTime()
  return Number.isFinite(t) ? Math.max(0, Math.round((Date.now() - t) / 60000)) : null
}

// format HH:mm en Europe/Paris à partir d’un ISO UTC, en tenant compte du DST
const parisHHmmAt = (iso?: string | null, addMin = 0): string => {
  if (!iso) return '—'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return '—'
  if (addMin) d.setMinutes(d.getMinutes() + addMin)
  return new Intl.DateTimeFormat('fr-FR', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
    timeZone: 'Europe/Paris',
  }).format(d)
}

/* ───────── Hook principal ───────── */

export function useAppData(horizonMin = 15, refreshMs = 300_000) {
  const [stations, setStations] = useState<Station[]>([])
  const [forecast, setForecast] = useState<Forecast[] | any[]>([])
  const [badges, setBadges] = useState<any>(null)
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)

  const timer = useRef<number | null>(null)
  const alive = useRef(true)

  const load = useCallback(async () => {
    if (!alive.current) return
    setLoading(true)
    setError(null)

    try {
      // 1) Stations
      const st = await getStations()
      if (!alive.current) return
      setStations(st ?? [])

      // 2) Forecast (on envoie stationcodes par défaut)
      const codes = Array.from(
        new Set((st ?? []).map(s => String((s as any).stationcode)).filter(Boolean))
      )
      const raw = codes.length ? await getForecastBatch(codes, horizonMin) : []
      if (!alive.current) return

      const fcRows = normalizeForecastRows(raw, horizonMin)
      setForecast(fcRows ?? [])

      // 3) Weather + badges côté client
      const weather = await getWeather().catch(() => null)

      // pred_ts_utc = horodatage UTC du run modèle
      const predBase: string | null = (fcRows.find((r: any) => r?.pred_ts_utc) || {}).pred_ts_utc ?? null

      const base = computeBadges(weather, predBase)
      const forecastHour = parisHHmmAt(predBase, horizonMin)
      const freshMin = ageMinutes(predBase)

      // structure stable pour BadgesBar
      setBadges({
        weather: base?.weather ?? null,
        freshness: base?.freshness ?? null,
        meta: {
          ...(base?.meta || {}),
          pred_ts_utc: predBase ?? null,
          forecast_hour: forecastHour,
          freshness_min: freshMin,
        },
      })
    } catch (e: any) {
      setError(e?.message || String(e))
    } finally {
      setLoading(false)
    }
  }, [horizonMin])

  // refresh aligné sur bin 5 min + sur focus/online
  useEffect(() => {
    alive.current = true

    const alignAndLoad = async () => {
      await load()
      if (!alive.current || refreshMs <= 0) return
      const now = Date.now()
      const bin = 5 * 60 * 1000
      const ms = Math.max(10_000, bin - (now % bin) + 2_000)
      if (timer.current) window.clearTimeout(timer.current)
      timer.current = window.setTimeout(alignAndLoad, ms)
    }

    alignAndLoad()

    const onVis = () => document.visibilityState === 'visible' && load()
    const onOnline = () => navigator.onLine && load()

    document.addEventListener('visibilitychange', onVis)
    window.addEventListener('online', onOnline)

    return () => {
      alive.current = false
      if (timer.current) window.clearTimeout(timer.current)
      document.removeEventListener('visibilitychange', onVis)
      window.removeEventListener('online', onOnline)
    }
  }, [load, refreshMs])

  /* ───────── dérivés ───────── */

  // index forecast par station_id ET stationcode (si présents)
  const forecastByKey = useMemo(() => {
    const m = new Map<string, any>()
    ;(forecast as any[]).forEach(f => {
      const id = (f as any)?.station_id
      const code = (f as any)?.stationcode
      if (id != null) m.set(String(id), f)
      if (code != null) m.set(String(code), f)
    })
    return m
  }, [forecast])

  const kpis = useMemo(() => {
    if (!stations.length) return { total: 0, bikes: 0, predBikes: 0 }
    const bikes = stations.reduce<number>((acc, s) => acc + toNumber((s as any).num_bikes_available, 0), 0)
    const predBikes = stations.reduce<number>((acc, s) => {
      const f = forecastByKey.get(String((s as any).stationcode))
      return acc + getPred(f)
    }, 0)
    return { total: stations.length, bikes, predBikes }
  }, [stations, forecastByKey])

  /* ───────── export ───────── */
  return {
    stations,
    forecast,          // rows normalisés
    badges,            // { weather, freshness?, meta: {pred_ts_utc, forecast_hour, freshness_min} }
    kpis,
    loading,
    error,
    refresh: load,
  }
}
