// ui/pages/app/embed.tsx
//
// =============================================================================
// Page "embed" de l'application Vélo Paris
// -----------------------------------------------------------------------------
// Cette page fournit une version intégrable de l’app Vélo Paris :
//   - pas de header / footer (noChrome = true),
//   - une carte Leaflet avec les stations Vélib’ et les prévisions de vélos,
//   - un panneau latéral avec badges, KPI globaux et liste de stations proches.
//
// Principes :
//   - La page consomme les mêmes services que l’app principale (météo, stations,
//     prévisions) mais dans un layout simplifié pour l’intégration (iframe, etc.).
//   - L’horizon de prévision (H) est commutable via un interrupteur 15 / 60 min.
//   - La géolocalisation (si disponible) sert à centrer la carte et la liste
//     des stations proches.
// =============================================================================

import Head from "next/head";
import dynamic from "next/dynamic";
import { useEffect, useMemo, useRef, useState } from "react";

// UI
import BadgesBar from "@/components/app/Badges";
import LoadingBar, { type LoadingBarStatus } from "@/components/common/LoadingBar";
import HorizonToggle from "@/components/common/HorizonToggle";

// Data
import { computeBadges } from "@/lib/services/badges";
import { getForecastFiltered } from "@/lib/services/forecast";
import { getStations } from "@/lib/services/stations";
import { getWeather } from "@/lib/services/weather";

// Types
import type { Station, Forecast } from "@/lib/types/types";
import type { Map as LeafletMap } from "leaflet";

// Map (no SSR) : la carte Leaflet ne doit être rendue que côté client
const MapView = dynamic(() => import("@/components/app/MapView"), { ssr: false });

/* ----------------- helpers ----------------- */

/**
 * Convertit une valeur quelconque en nombre, avec valeur de repli.
 *
 * @param x        Valeur à convertir.
 * @param fallback Valeur par défaut si la conversion échoue.
 */
const toNumber = (x: unknown, fallback = 0) => {
  const n = Number(x);
  return Number.isFinite(n) ? n : fallback;
};

/**
 * Extrait une prédiction de vélos à partir d’une ligne de forecast.
 * Tolère plusieurs noms de champs possibles (bikes_pred_int / bikes_pred).
 */
const getPred = (f?: any) => toNumber(f?.bikes_pred_int ?? f?.bikes_pred ?? 0, 0);

/**
 * Normalise la façon de récupérer un identifiant de station (clé unique).
 *
 * Recherche dans l’ordre :
 *   - station_id
 *   - stationId
 *   - id
 */
const keyFor = (obj: any): string | null => {
  if (!obj) return null;
  const id = obj.station_id ?? obj.stationId ?? obj.id ?? null;
  if (id != null && String(id).trim() !== "") return String(id);
  return null;
};

/**
 * Formate un timestamp ISO en heure locale Paris (HH:mm).
 *
 * @param iso Timestamp ISO (avec ou sans "Z").
 * @returns   Heure locale "HH:mm" ou "—" si invalide.
 */
const parisHHmmAt = (iso?: string | null): string => {
  if (!iso) return "—";
  const utcIso = iso.endsWith("Z") ? iso : `${iso}Z`;
  const d = new Date(utcIso);
  return new Intl.DateTimeFormat("fr-FR", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "Europe/Paris",
  }).format(d);
};

/**
 * Calcule l’âge (en minutes) d’un timestamp ISO par rapport à "maintenant".
 */
const ageMinutes = (iso?: string | null): number | null => {
  if (!iso) return null;
  const utcIso = iso.endsWith("Z") ? iso : `${iso}Z`;
  const t = new Date(utcIso).getTime();
  return Math.max(0, Math.round((Date.now() - t) / 60000));
};

/**
 * Renvoie le timestamp le plus récent dans un tableau de lignes.
 *
 * @param rows Liste de lignes quelconques.
 * @param keyA Clé principale contenant un timestamp.
 * @param keyB Clé alternative éventuelle, utilisée si keyA est absente.
 */
const latestIso = (rows: any[], keyA: string, keyB?: string): string | null => {
  const vals = rows
    .map((r) => (r?.[keyA] ?? (keyB ? r?.[keyB] : null)) as string | null)
    .filter(Boolean) as string[];
  if (!vals.length) return null;
  vals.sort(
    (a, b) =>
      new Date((b.endsWith("Z") ? b : b + "Z")).getTime() -
      new Date((a.endsWith("Z") ? a : a + "Z")).getTime()
  );
  return vals[0] ?? null;
};

/* ----------------- component ----------------- */

function AppEmbedPage() {
  // Données "brutes" de l’app
  const [stations, setStations] = useState<Station[]>([]);
  const [forecast, setForecast] = useState<Forecast[] | any[]>([]);
  const [badges, setBadges] = useState<any>(null);

  // Carte & géolocalisation
  const [center, setCenter] = useState<[number, number]>([48.8566, 2.3522]); // Paris par défaut
  const [userPos, setUserPos] = useState<[number, number] | null>(null);
  const userCenteredOnce = useRef(false); // évite de recentrer plusieurs fois
  const mapRef = useRef<LeafletMap | null>(null);

  // ⬇️ Horizon contrôlé par l'interrupteur (par défaut sur 60 min)
  const [H, setH] = useState<number>(60);

  // État de chargement / erreur pour la barre de statut
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const barStatus: LoadingBarStatus = loading ? "loading" : error ? "error" : "success";

  // ----------------------------------------------------------------------------
  // Boot initial : météo + stations (ne dépendent pas de H)
  // ----------------------------------------------------------------------------
  useEffect(() => {
    let alive = true;

    const boot = async () => {
      try {
        setLoading(true);
        setError(null);

        // On charge en parallèle la météo et les stations
        const [weather, st] = await Promise.all([getWeather(), getStations()]);
        if (!alive) return;

        setStations(st ?? []);

        // Badges météo initiaux
        // (les timestamps de prévision seront ajoutés après le fetch des forecasts)
        const base = computeBadges(weather ?? null, null);
        setBadges({
          weather: base?.weather ?? null,
          freshness: { age_minutes: null },
          meta: {
            ...(base?.meta || {}),
            pred_ts_utc: null,
            target_ts_utc: null,
            forecast_hour: "—",
            freshness_min: null,
          },
        });
      } catch (err: any) {
        if (alive) setError(String(err?.message ?? err));
      } finally {
        if (alive) setLoading(false);
      }
    };

    boot();
    return () => {
      alive = false;
    };
  }, []);

  // ----------------------------------------------------------------------------
  // Fetch des prévisions (forecast)
  //   - dépend de H (horizon) et de la liste des stations connues
  //   - rafraîchit toutes les 5 minutes
  // ----------------------------------------------------------------------------
  useEffect(() => {
    let alive = true;
    let timer: number | null = null;

    const run = async () => {
      try {
        setLoading(true);
        setError(null);

        // Liste des IDs de stations à demander côté backend
        const keys = Array.from(
          new Set((stations ?? []).map((s) => keyFor(s as any)).filter(Boolean) as string[])
        );

        const fcRows = keys.length ? await getForecastFiltered(keys, H) : [];
        if (!alive) return;
        setForecast(fcRows);

        // Mise à jour des badges avec les horodatages des prévisions
        const latestBin = latestIso(fcRows, "tbin_latest", "tbin_utc");
        const latestPredTs = latestIso(fcRows, "pred_ts_utc");
        const latestTarget = latestIso(fcRows, "target_ts_utc");
        const forecastHourParis = parisHHmmAt(latestTarget);
        const ageMin = ageMinutes(latestBin);

        setBadges((prev: any) => ({
          ...(prev || {}),
          freshness: { age_minutes: ageMin ?? null },
          meta: {
            ...(prev?.meta || {}),
            pred_ts_utc: latestPredTs ?? null,
            target_ts_utc: latestTarget ?? null,
            forecast_hour: forecastHourParis,
            freshness_min: ageMin ?? null,
          },
        }));
      } catch (err: any) {
        if (alive) setError(String(err?.message ?? err));
      } finally {
        if (alive) setLoading(false);
      }
    };

    // Exécute le fetch immédiat si on a déjà des stations
    if (stations.length) run();

    // Rafraîchit les prévisions toutes les 5 minutes à horizon constant
    timer = window.setInterval(run, 5 * 60 * 1000);

    return () => {
      alive = false;
      if (timer) window.clearInterval(timer);
    };
  }, [H, stations]);

  // ----------------------------------------------------------------------------
  // Géolocalisation utilisateur (optionnelle)
  //   - si disponible, on mémorise la position et on centre la carte une fois.
  // ----------------------------------------------------------------------------
  useEffect(() => {
    if (typeof navigator !== "undefined" && navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          const p: [number, number] = [pos.coords.latitude, pos.coords.longitude];
          setUserPos(p);
          if (!userCenteredOnce.current) {
            userCenteredOnce.current = true;
            setCenter(p);
            mapRef.current?.setView?.(p, 14);
          }
        },
        (err) => console.warn("[geoloc]", err),
        { enableHighAccuracy: true, timeout: 8000, maximumAge: 15000 }
      );
    }
  }, []);

  // ----------------------------------------------------------------------------
  // Stations avec géocoordonnées valides
  // ----------------------------------------------------------------------------
  const stationsWithGeo = useMemo(
    () =>
      stations.filter(
        (s) =>
          typeof (s as any).lat === "number" &&
          Number.isFinite((s as any).lat) &&
          typeof (s as any).lon === "number" &&
          Number.isFinite((s as any).lon)
      ),
    [stations]
  );

  // Index des prévisions par station_id
  const forecastByKey = useMemo(() => {
    const m = new Map<string, any>();
    (forecast as any[]).forEach((f) => {
      const id = f?.station_id;
      if (id != null && String(id).trim() !== "") m.set(String(id), f);
    });
    return m;
  }, [forecast]);

  // ----------------------------------------------------------------------------
  // KPI globaux : nombre de stations, vélos actuels, vélos prévus
  // ----------------------------------------------------------------------------
  const kpis = useMemo(() => {
    if (!stations.length) return { total: 0, bikes: 0, predBikes: 0 };
    const bikes = stations.reduce(
      (acc, s) => acc + toNumber((s as any).num_bikes_available, 0),
      0
    );
    const predBikes = stations.reduce((acc, s) => {
      const k = keyFor(s as any);
      const f = k ? forecastByKey.get(k) : undefined;
      return acc + getPred(f);
    }, 0);
    return { total: stations.length, bikes, predBikes };
  }, [stations, forecastByKey]);

  // Heure de prévision commune, formatée en heure Paris
  const forecastHourParis = useMemo(() => {
    const latestTarget = latestIso(forecast as any[], "target_ts_utc");
    return parisHHmmAt(latestTarget);
  }, [forecast]);

  // ----------------------------------------------------------------------------
  // Haversine : calcul de distance entre deux points géographiques (mètres)
  // ----------------------------------------------------------------------------
  const R = 6371000;
  const toRad = (deg: number) => (deg * Math.PI) / 180;
  const haversine = (lat1: number, lon1: number, lat2: number, lon2: number) => {
    const dLat = toRad(lat2 - lat1);
    const dLon = toRad(lon2 - lon1);
    const a =
      Math.sin(dLat / 2) ** 2 +
      Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
    return 2 * R * Math.asin(Math.sqrt(a));
  };

  // ----------------------------------------------------------------------------
  // Stations "proches" de l’utilisateur (ou du centre par défaut)
  //   - nécessite stations géocodées + prévisions disponibles
  //   - filtre sur prédiction ≥ 2 vélos
  //   - tri par distance croissante
  // ----------------------------------------------------------------------------
  const nearby = useMemo(() => {
    if (!stationsWithGeo.length || !Array.isArray(forecast)) return [];
    const user = userPos ?? center;
    return stationsWithGeo
      .map((s: any) => {
        const k = keyFor(s);
        const f = k ? forecastByKey.get(k) : undefined;
        const pred = getPred(f);
        const d = haversine(Number(s.lat), Number(s.lon), user[0], user[1]);
        return { ...s, pred, distance: d };
      })
      .filter((s) => s.pred >= 2)
      .sort((a, b) => a.distance - b.distance)
      .slice(0, 10);
  }, [stationsWithGeo, forecast, forecastByKey, userPos, center]);

  return (
    <>
      <Head>
        <title>Vélo Paris Embed — Carte & prévisions</title>
        <meta name="description" content="Version intégrable de Vélo Paris App (sans header ni footer)." />
      </Head>

      <main
        className="main app-main"
        style={{ display: "flex", gap: "12px", padding: "8px 12px", height: "100vh" }}
      >
        <MapView
          stations={stationsWithGeo as Station[]}
          forecast={forecast as Forecast[]}
          mode={H === 60 ? "t60" : "t15"}
          center={center}
          userPos={userPos}
          setMapInstance={(m: LeafletMap) => {
            mapRef.current = m;
          }}
        />

        <aside className="panel side-panel">
          {/* Badges (KPI étiquettes) */}
          <div className="badges" style={{ marginBottom: 8 }}>
            {badges ? <BadgesBar data={badges} /> : <LoadingBar status={barStatus} />}
          </div>

          {/* Interrupteur SOUS les badges */}
          <div className="hz-wrap" style={{ display: "flex", alignItems: "center", margin: "4px 0 12px" }}>
            <HorizonToggle
              value={H}
              onChange={setH}
              ariaLabel="Horizon de prévision"
              leftValue={15}
              rightValue={60}
              leftLabel="15 min"
              rightLabel="60 min"
              dense
            />
          </div>

          {error && (
            <div className="small" style={{ color: "#ef4444", margin: "6px 0 10px" }}>
              {error}
            </div>
          )}

          {/* KPI globaux : nombre de stations / vélos actuels / vélos prévus */}
          <div className="kpi">
            <div className="card">
              <div className="small">Stations</div>
              <div className="val">{kpis.total.toLocaleString()}</div>
            </div>
            <div className="card">
              <div className="small">Vélos actuels</div>
              <div className="val">{kpis.bikes.toLocaleString()}</div>
            </div>
            <div className="card">
              <div className="small">Vélos prévus</div>
              <div className="val">{kpis.predBikes.toLocaleString()}</div>
            </div>
          </div>

          <h3 style={{ margin: "20px 0 10px", fontSize: "1rem" }}>
            Stations proches · prévision {forecastHourParis} ({H} min)
          </h3>

          {/* Liste des stations proches avec au moins 2 vélos prévus */}
          <div className="list">
            {nearby.map((s: any) => (
              <div
                key={String(s.station_id)}
                className="row"
                style={{ background: "rgba(255,255,255,0.03)", marginBottom: 6, padding: 8 }}
              >
                <div>
                  <div style={{ fontWeight: 700 }}>{s.name ?? s.station_id}</div>
                  <div className="small">#{String(s.station_id)}</div>
                </div>
                <div style={{ textAlign: "right" }}>
                  <div className="small">{forecastHourParis}</div>
                  <div style={{ fontWeight: 700, fontSize: "1.1rem" }}>
                    {getPred(forecastByKey.get(String(s.station_id)))}
                  </div>
                </div>
              </div>
            ))}
            {!nearby.length && (
              <div className="small">Aucune station proche avec ≥2 vélos prévus</div>
            )}
          </div>
        </aside>
      </main>
    </>
  );
}

// 🚫 Désactive le header/footer pour cette page "embed"
(AppEmbedPage as any).noChrome = true;

export default AppEmbedPage;
