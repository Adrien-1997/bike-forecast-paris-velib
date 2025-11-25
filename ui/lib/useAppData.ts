// ui/lib/useAppData.ts
//
// =============================================================================
// Hook principal de l’application Vélib’ Forecast.
//
// Rôle :
// - Centraliser la récupération des **données de l’app** :
//     • stations (métadonnées + état courant),
//     • prévisions de vélos pour un horizon donné,
//     • météo "live" et badges de bandeau,
//     • petits KPI globaux (total stations, vélos présents / prédits).
// - Gérer le **refresh automatique** aligné sur les bins 5 minutes,
//   avec reprise sur focus onglet / retour en ligne.
// - Exposer un contrat simple aux composants :
//     { stations, forecast, badges, kpis, loading, error, refresh }.
//
// Contraintes :
// - Aucun accès direct au DOM en dehors des listeners (visibility / online).
// - Aucun formatage UI (pas de JSX) : hook purement "données".
// - Tolérance aux erreurs réseau : on remonte un message d’erreur et on
//   laisse l’UI décider de l’affichage.
//
// Ce hook est pensé comme **point d’entrée unique** pour l’écran principal
// (liste + carte) et toute UI qui a besoin d’un snapshot "app" cohérent.
// =============================================================================

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { getStations } from "@/lib/services/stations";
import { getForecastFiltered } from "@/lib/services/forecast";
import { getWeather } from "@/lib/services/weather";
import { computeBadges } from "@/lib/services/badges";

import type { Station, Forecast } from "@/lib/types/types";

/* ───────── helpers ───────── */

/**
 * Cast "safe" vers number.
 *
 * - Accepte les number finis tels quels.
 * - Tente un Number(x) sur les strings.
 * - Retourne `fallback` si la valeur n’est pas convertible proprement.
 */
const toNumber = (x: unknown, fallback = 0): number => {
  const n = Number(x);
  return Number.isFinite(n) ? n : fallback;
};

/**
 * Récupère la prédiction entière à partir d’une ligne de forecast.
 *
 * Ordre de priorité :
 * - `bikes_pred_int` (entier déjà calculé côté backend),
 * - sinon `bikes_pred` castée en number,
 * - sinon 0.
 */
const getPred = (f?: any): number =>
  toNumber(f?.bikes_pred_int ?? f?.bikes_pred ?? 0, 0);

/**
 * Construit une clé stable pour identifier une station côté client.
 *
 * - Plusieurs schémas de payload coexistent :
 *     • `station_id` / `stationId`
 *     • `stationcode` / `code`
 * - On essaie d’abord id, puis code, en stringifiant.
 */
const keyFor = (obj: any): string | null => {
  if (!obj) return null;
  const id = obj.station_id ?? obj.stationId ?? null;
  const code = obj.stationcode ?? obj.code ?? null;
  if (id != null && String(id).trim() !== "") return String(id);
  if (code != null && String(code).trim() !== "") return String(code);
  return null;
};

/**
 * Normalise la forme des lignes de forecast.
 *
 * Schémas supportés :
 * - tableau direct :        `[...]`
 * - bundle par horizon :   `{ data: { "60": [...], "15": [...] } }`
 * - legacy :               `{ predictions: [...] }`
 */
function normalizeForecastRows(payload: any, horizon = 60): any[] {
  if (Array.isArray(payload)) return payload;
  if (payload?.data?.[String(horizon)] && Array.isArray(payload.data[String(horizon)])) {
    return payload.data[String(horizon)];
  }
  const hKey = `h${horizon}`;
  if (payload?.data?.[hKey] && Array.isArray(payload.data[hKey])) {
    return payload.data[hKey];
  }
  if (Array.isArray(payload?.predictions)) return payload.predictions;
  return [];
}

/**
 * Convertit un ISO UTC (avec ou sans `Z`) en Date.
 * Retourne `null` si la date est invalide.
 */
const toUtcDate = (iso?: string | null): Date | null => {
  if (!iso) return null;
  const s = iso.endsWith("Z") ? iso : `${iso}Z`;
  const d = new Date(s);
  return Number.isNaN(d.getTime()) ? null : d;
};

/**
 * Âge d’un timestamp en minutes (par rapport à maintenant).
 *
 * - Retourne `null` si le timestamp est invalide.
 * - Clamp à 0 pour éviter les valeurs négatives.
 */
const ageMinutes = (iso?: string | null): number | null => {
  const d = toUtcDate(iso);
  return d ? Math.max(0, Math.round((Date.now() - d.getTime()) / 60000)) : null;
};

/**
 * Formatte un timestamp ISO en "HH:mm" sur le fuseau Europe/Paris.
 * Retourne "—" si la date est invalide.
 */
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

/**
 * Renvoie le timestamp le plus récent parmi une liste de lignes.
 *
 * - `keyA` : champ principal à regarder.
 * - `keyB` : champ de fallback éventuel.
 */
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

/**
 * Hook principal de l’app.
 *
 * Paramètres :
 * - `horizonMin` : horizon des prévisions (en minutes), par ex. 60.
 * - `refreshMs`  : intervalle d’actualisation automatique (ms). Si ≤ 0,
 *                  désactive le refresh programmé (mais pas le refresh manuel).
 *
 * Comportement :
 * - Charge séquentiellement :
 *     1. les stations,
 *     2. les prévisions filtrées sur ces stations,
 *     3. la météo + badges (freshness + heure de prévision).
 * - Met en place un timer aligné sur les bins 5 minutes (00, 05, 10, …),
 *   et relance un chargement :
 *     • à chaque bin,
 *     • à la remise au premier plan de l’onglet,
 *     • au retour en ligne du navigateur.
 *
 * Retour :
 * - `stations` : tableau de Station.
 * - `forecast` : tableau de Forecast (normalisé).
 * - `badges`   : payload compact pour le bandeau.
 * - `kpis`     : petits indicateurs globaux (total, bikes, predBikes).
 * - `loading`  : booléen de chargement.
 * - `error`    : message d’erreur (string) ou null.
 * - `refresh`  : fonction manuelle pour relancer un fetch complet.
 */
export function useAppData(horizonMin = 60, refreshMs = 300_000) {
  const [stations, setStations] = useState<Station[]>([]);
  const [forecast, setForecast] = useState<Forecast[] | any[]>([]);
  const [badges, setBadges] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Timer pour le refresh aligné sur les bins
  const timer = useRef<number | null>(null);
  // Flag de vie pour éviter les `setState` après unmount
  const alive = useRef(true);

  /**
   * Charge l’intégralité des données app (stations + forecast + badges).
   *
   * Séquence :
   *  1. Stations
   *  2. Forecast filtré sur toutes les clés (id + code)
   *  3. Météo + badges enrichis (freshness, heures, etc.)
   */
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
      const raw = keys.length ? await getForecastFiltered(keys, horizonMin) : [];
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

      // Badges de base + enrichissements spécifiques à l’app
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

    // Cleanup : on stoppe le timer + on retire les listeners
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

  /**
   * KPI globaux pour le bandeau / header app.
   *
   * - `total`      : nombre de stations.
   * - `bikes`      : somme des vélos actuellement disponibles.
   * - `predBikes`  : somme des vélos prédits (forecast) sur les mêmes stations.
   */
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
    forecast,
    badges,
    kpis,
    loading,
    error,
    refresh: load,
  };
}