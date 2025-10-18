// ui/lib/useAppData.ts
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { getStations } from "@/lib/services/stations";
import { getForecastBatch } from "@/lib/services/forecast";
import { getWeather } from "@/lib/services/weather";
import { computeBadges } from "@/lib/services/badges";

import type { Station, Forecast } from "@/lib/types/types";

/* ───────── helpers ───────── */

const toNumber = (x: unknown, fallback = 0): number => {
  const n = Number(x);
  return Number.isFinite(n) ? n : fallback;
};

const getPred = (f?: any): number =>
  toNumber(f?.bikes_pred_int ?? f?.bikes_pred ?? 0, 0);

// clé possible = station_id OU stationcode (selon ce que renvoie l’API)
const keyFor = (obj: any): string | null => {
  if (!obj) return null;
  const id = obj.station_id ?? obj.stationId ?? null;
  const code = obj.stationcode ?? obj.code ?? null;
  if (id != null && String(id).trim() !== "") return String(id);
  if (code != null && String(code).trim() !== "") return String(code);
  return null;
};

// normalisation forecast: array direct OU bundle { data: {"15":[...]}, ... }
function normalizeForecastRows(payload: any, horizon = 15): any[] {
  if (Array.isArray(payload)) return payload;
  if (payload?.data?.[String(horizon)] && Array.isArray(payload.data[String(horizon)])) {
    return payload.data[String(horizon)];
  }
  if (Array.isArray(payload?.predictions)) return payload.predictions;
  return [];
}

// ISO UTC → Date (force le Z si absent)
const toUtcDate = (iso?: string | null): Date | null => {
  if (!iso) return null;
  const s = iso.endsWith("Z") ? iso : `${iso}Z`;
  const d = new Date(s);
  return Number.isNaN(d.getTime()) ? null : d;
};

// âge en minutes
const ageMinutes = (iso?: string | null): number | null => {
  const d = toUtcDate(iso);
  return d ? Math.max(0, Math.round((Date.now() - d.getTime()) / 60000)) : null;
};

// HH:mm Europe/Paris
const parisHHmmAt = (iso?: string | null): string => {
  const d = toUtcDate(iso);
  if (!d) return "—";
  return new Intl.DateTimeFormat("fr-FR", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "Europe/Paris",
  }).format(d);
};

// “plus récent” d’un champ temporel
const latestIso = (rows: any[], keyA: string, keyB?: string): string | null => {
  const vals = rows
    .map((r) => (r?.[keyA] ?? (keyB ? r?.[keyB] : null)) as string | null)
    .filter(Boolean) as string[];
  if (!vals.length) return null;
  vals.sort(
    (a, b) =>
      (toUtcDate(b)?.getTime() ?? 0) - (toUtcDate(a)?.getTime() ?? 0)
  );
  return vals[0] ?? null;
};

/* ───────── Hook principal ───────── */

export function useAppData(horizonMin = 15, refreshMs = 300_000) {
  const [stations, setStations] = useState<Station[]>([]);
  const [forecast, setForecast] = useState<Forecast[] | any[]>([]);
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
      // 1) Stations
      const st = await getStations();
      if (!alive.current) return;
      setStations(st ?? []);

      // 2) Forecast — envoie toutes les clés (id/code) disponibles
      const keys = Array.from(
        new Set((st ?? []).map((s) => keyFor(s as any)).filter(Boolean) as string[])
      );
      const raw = keys.length ? await getForecastBatch(keys, horizonMin) : [];
      if (!alive.current) return;

      const fcRows = normalizeForecastRows(raw, horizonMin);
      setForecast(fcRows ?? []);

      // 3) Weather + badges enrichis
      const weather = await getWeather().catch(() => null);

      // Références temporelles
      const latestTarget = latestIso(fcRows, "target_ts_utc");             // heure de prévision affichée
      const latestPredTs = latestIso(fcRows, "pred_ts_utc");               // horodatage du run modèle
      const latestObserved = latestIso(fcRows, "tbin_latest", "tbin_utc"); // fraîcheur des observations

      const forecastHour = parisHHmmAt(latestTarget);
      const freshMin = ageMinutes(latestObserved);

      const base = computeBadges(weather, latestPredTs);
      setBadges({
        weather: base?.weather ?? null,
        freshness: { age_minutes: freshMin ?? null },
        meta: {
          ...(base?.meta || {}),
          pred_ts_utc: latestPredTs ?? null,
          target_ts_utc: latestTarget ?? null,
          forecast_hour: forecastHour,
          freshness_min: freshMin ?? null,
        },
      });
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setLoading(false);
    }
  }, [horizonMin]);

  // refresh aligné sur bins 5 min + focus/online
  useEffect(() => {
    alive.current = true;

    const alignAndLoad = async () => {
      await load();
      if (!alive.current || refreshMs <= 0) return;

      const now = Date.now();
      const bin = 5 * 60 * 1000;
      const ms = Math.max(10_000, bin - (now % bin) + 2_000);

      if (timer.current) window.clearTimeout(timer.current);
      timer.current = window.setTimeout(alignAndLoad, ms);
    };

    alignAndLoad();

    const onVis = () => document.visibilityState === "visible" && load();
    const onOnline = () => navigator.onLine && load();

    document.addEventListener("visibilitychange", onVis);
    window.addEventListener("online", onOnline);

    return () => {
      alive.current = false;
      if (timer.current) window.clearTimeout(timer.current);
      document.removeEventListener("visibilitychange", onVis);
      window.removeEventListener("online", onOnline);
    };
  }, [load, refreshMs]);

  /* ───────── dérivés ───────── */

  // index forecast par station_id ET stationcode (si présents)
  const forecastByKey = useMemo(() => {
    const m = new Map<string, any>();
    (forecast as any[]).forEach((f) => {
      const id = (f as any)?.station_id;
      const code = (f as any)?.stationcode;
      if (id != null) m.set(String(id), f);
      if (code != null) m.set(String(code), f);
    });
    return m;
  }, [forecast]);

  const kpis = useMemo(() => {
    if (!stations.length) return { total: 0, bikes: 0, predBikes: 0 };
    const bikes = stations.reduce<number>(
      (acc, s) => acc + toNumber((s as any).num_bikes_available, 0),
      0
    );
    const predBikes = stations.reduce<number>((acc, s) => {
      const key = keyFor(s as any);
      const f = key ? forecastByKey.get(key) : undefined;
      return acc + getPred(f);
    }, 0);
    return { total: stations.length, bikes, predBikes };
  }, [stations, forecastByKey]);

  /* ───────── export ───────── */
  return {
    stations,
    forecast,          // rows normalisés
    badges,            // { weather, freshness: {age_minutes}, meta: {pred_ts_utc, target_ts_utc, forecast_hour, freshness_min} }
    kpis,
    loading,
    error,
    refresh: load,
  };
}
