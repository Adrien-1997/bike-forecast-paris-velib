// ui/lib/useAppData.ts
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { getStations } from '@/lib/services/stations';
import { getForecastBatch } from '@/lib/services/forecast';
import { getBadges } from '@/lib/services/badges';
import type { Station, Forecast } from '@/lib/types';

const toNumber = (x: unknown, fallback = 0) => {
  const n = Number(x);
  return Number.isFinite(n) ? n : fallback;
};
const getPred = (f?: Forecast) => (f ? toNumber((f as any).bikes_pred_t15, 0) : 0);

function parseIsoToMs(v: unknown): number {
  if (!v) return 0;
  const t = Date.parse(String(v));
  return Number.isFinite(t) ? t : 0;
}
function computeFreshnessMin(updatedAt: unknown): number {
  const t = parseIsoToMs(updatedAt);
  if (!t) return 0;
  const now = Date.now();
  const diffMs = Math.max(0, now - t);
  return Math.floor(diffMs / 60000);
}
function normalizeBadges(bd: any) {
  if (!bd) return bd;
  const serverFresh =
    bd?.meta?.freshness_min ?? bd?.freshness_min ?? bd?.meta?.freshness ?? bd?.freshness ?? 0;
  const updatedAt = bd?.meta?.updated_at ?? bd?.updated_at ?? bd?.ts ?? null;
  const computed = computeFreshnessMin(updatedAt);
  const finalFresh = Number(serverFresh) > 0 ? Number(serverFresh) : computed;
  return {
    ...bd,
    meta: {
      ...(bd.meta || {}),
      updated_at: updatedAt,
      freshness_min_norm: finalFresh,
      freshness_min_server: Number(serverFresh) || 0,
      freshness_min_client: computed,
    },
  };
}

export function useAppData(horizonMin = 15, refreshMs = 300_000) {
  const [stations, setStations] = useState<Station[]>([]);
  const [forecast, setForecast] = useState<Forecast[]>([]);
  const [badges, setBadges] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);
  const alive = useRef(true);

  const load = useCallback(async () => {
    if (!alive.current) return;
    setLoading(true);
    setError(null);
    try {
      // 1) stations
      const st = await getStations();
      if (!alive.current) return;
      setStations(st ?? []);

      // 2) forecast (utilise les codes stations)
      const codes = (st ?? []).map(s => String((s as any).stationcode));
      const fc = await getForecastBatch(codes, horizonMin);
      if (!alive.current) return;
      setForecast(fc ?? []);

      // 3) badges
      const bd = await getBadges();
      if (!alive.current) return;
      setBadges(normalizeBadges(bd));
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setLoading(false);
    }
  }, [horizonMin]);

  // interval + focus + retour en ligne
  useEffect(() => {
    alive.current = true;
    load(); // initial

    if (refreshMs > 0) {
      timer.current = window.setInterval(load, refreshMs);
    }

    const onVis = () => document.visibilityState === 'visible' && load();
    const onOnline = () => navigator.onLine && load();

    document.addEventListener('visibilitychange', onVis);
    window.addEventListener('online', onOnline);

    return () => {
      alive.current = false;
      if (timer.current) window.clearInterval(timer.current);
      document.removeEventListener('visibilitychange', onVis);
      window.removeEventListener('online', onOnline);
    };
  }, [load, refreshMs]);

  // dérivés
  const forecastByCode = useMemo(
    () => new Map(forecast.map(f => [String((f as any).stationcode), f])),
    [forecast]
  );

  const kpis = useMemo(() => {
    if (!stations.length) return { total: 0, bikes: 0, predBikes: 0 };
    const bikes = stations.reduce((acc, s) => acc + toNumber((s as any).num_bikes_available, 0), 0);
    const predBikes = stations.reduce(
      (acc, s) => acc + getPred(forecastByCode.get(String((s as any).stationcode))),
      0
    );
    return { total: stations.length, bikes, predBikes };
  }, [stations, forecastByCode]);

  return {
    stations,
    forecast,
    badges,
    kpis,
    loading,
    error,
    refresh: load,
  };
}
