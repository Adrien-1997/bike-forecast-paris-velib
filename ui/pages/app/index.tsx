// ui/pages/app/index.tsx
import { useEffect, useMemo, useRef, useState } from "react";
import Head from "next/head";
import dynamic from "next/dynamic";

// Layout
import GlobalHeader from "@/components/layout/GlobalHeader";
import GlobalFooter from "@/components/layout/GlobalFooter";

// UI
import BadgesBar from "@/components/app/Badges";

// Data
import { computeBadges } from "@/lib/services/badges";
import { getForecastBatch } from "@/lib/services/forecast";
import { getStations } from "@/lib/services/stations";
import { getWeather } from "@/lib/services/weather";

// Types
import type { Station, Forecast } from "@/lib/types/types";
import type { Map as LeafletMap } from "leaflet";

// Map (no SSR)
const MapView = dynamic(() => import("@/components/app/MapView"), { ssr: false });

/* ----------------- helpers ----------------- */

const toNumber = (x: unknown, fallback = 0) => {
  const n = Number(x);
  return Number.isFinite(n) ? n : fallback;
};

const getPred = (f?: any) => toNumber(f?.bikes_pred_int ?? f?.bikes_pred ?? 0, 0);

// clé de jointure — uniquement station_id
const keyFor = (obj: any): string | null => {
  if (!obj) return null;
  const id = obj.station_id ?? obj.stationId ?? obj.id ?? null;
  if (id != null && String(id).trim() !== "") return String(id);
  return null;
};

// accepte soit un array direct, soit le bundle {generated_at, horizons, data: {"15":[...]} }
function normalizeForecastRows(payload: any, horizon = 15): any[] {
  if (Array.isArray(payload)) return payload;
  if (payload && payload.data) {
    const k = String(horizon);
    if (Array.isArray(payload.data[k])) return payload.data[k];
  }
  return [];
}

/**
 * Convertit un timestamp ISO (UTC venant de l'API) vers l'heure locale Paris HH:mm.
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

// âge en minutes entre maintenant (local) et un timestamp UTC
const ageMinutes = (iso?: string | null): number | null => {
  if (!iso) return null;
  const utcIso = iso.endsWith("Z") ? iso : `${iso}Z`;
  const t = new Date(utcIso).getTime();
  return Math.max(0, Math.round((Date.now() - t) / 60000));
};

// util: renvoie la valeur “la plus récente” d’un champ parmi les rows
const latestIso = (rows: any[], keyA: string, keyB?: string): string | null => {
  const vals = rows
    .map(r => (r?.[keyA] ?? (keyB ? r?.[keyB] : null)) as string | null)
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

export default function AppHomePage() {
  const [stations, setStations] = useState<Station[]>([]);
  const [forecast, setForecast] = useState<Forecast[] | any[]>([]);
  const [badges, setBadges] = useState<any>(null);

  const [center, setCenter] = useState<[number, number]>([48.8566, 2.3522]);
  const [userPos, setUserPos] = useState<[number, number] | null>(null);
  const userCenteredOnce = useRef(false);
  const mapRef = useRef<LeafletMap | null>(null);
  const H = 15;

  useEffect(() => {
    let alive = true;
    const loadOnce = async () => {
      try {
        const weather = await getWeather();
        if (!alive) return;

        const st = await getStations();
        if (!alive) return;
        setStations(st ?? []);

        const keys = Array.from(
          new Set((st ?? []).map(s => keyFor(s as any)).filter(Boolean) as string[])
        );
        const raw = keys.length ? await getForecastBatch(keys, H) : [];
        if (!alive) return;

        const fcRows = normalizeForecastRows(raw, H);
        setForecast(fcRows);

        const latestBin = latestIso(fcRows, "tbin_latest", "tbin_utc");
        const latestPredTs = latestIso(fcRows, "pred_ts_utc");
        const latestTarget = latestIso(fcRows, "target_ts_utc");

        const forecastHourParis = parisHHmmAt(latestTarget);
        const ageMin = ageMinutes(latestBin);

        const base = computeBadges(weather ?? null, latestPredTs);
        setBadges({
          weather: base?.weather ?? null,
          freshness: { age_minutes: ageMin ?? null },
          meta: {
            ...(base?.meta || {}),
            pred_ts_utc: latestPredTs ?? null,
            target_ts_utc: latestTarget ?? null,
            forecast_hour: forecastHourParis,
            freshness_min: ageMin ?? null,
          },
        });
      } catch (err) {
        console.error("[loadOnce]", err);
      }
    };

    loadOnce();
    const id = setInterval(loadOnce, 5 * 60 * 1000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  // geoloc + recadrage 1x
  useEffect(() => {
    if (typeof navigator !== "undefined" && navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        pos => {
          const p: [number, number] = [pos.coords.latitude, pos.coords.longitude];
          setUserPos(p);
          if (!userCenteredOnce.current) {
            userCenteredOnce.current = true;
            setCenter(p);
            mapRef.current?.setView?.(p, 14);
          }
        },
        err => console.warn("[geoloc]", err),
        { enableHighAccuracy: true, timeout: 8000, maximumAge: 15000 }
      );
    }
  }, []);

  const stationsWithGeo = useMemo(
    () =>
      stations.filter(
        s =>
          typeof (s as any).lat === "number" &&
          Number.isFinite((s as any).lat) &&
          typeof (s as any).lon === "number" &&
          Number.isFinite((s as any).lon)
      ),
    [stations]
  );

  const forecastByKey = useMemo(() => {
    const m = new Map<string, any>();
    (forecast as any[]).forEach(f => {
      const id = f?.station_id;
      if (id != null && String(id).trim() !== "") m.set(String(id), f);
    });
    return m;
  }, [forecast]);

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

  const forecastHourParis = useMemo(() => {
    const latestTarget = latestIso(forecast as any[], "target_ts_utc");
    return parisHHmmAt(latestTarget);
  }, [forecast]);

  // distance haversine (m)
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
      .filter(s => s.pred >= 2)
      .sort((a, b) => a.distance - b.distance)
      .slice(0, 10);
  }, [stationsWithGeo, forecast, forecastByKey, userPos, center]);

  // items du header (adaptés à l'app)
  const headerItems = [
    { label: "Carte", href: "/app" },
    { label: "Monitoring", href: "/monitoring" },
    { label: "Accueil", href: "/" },
  ];

  return (
    <>
      <Head>
        <title>Vélib’ Paris — Disponibilités et prévisions</title>
        <meta name="description" content="Disponibilités temps réel et prévisions courtes." />
      </Head>

      {/* Header global */}
      <GlobalHeader items={headerItems} brandHref="/" />

      {/* Contenu */}
      <div className="container" style={{ paddingTop: "16px" }}>

        <main className="main">
          <div className="panel map-card">
            <div className="map-fill">
              <MapView
                stations={stationsWithGeo as Station[]}
                forecast={forecast as Forecast[]}
                mode="t15"
                center={center}
                userPos={userPos}
                setMapInstance={(m: LeafletMap) => {
                  mapRef.current = m;
                }}
              />
            </div>
          </div>

          <aside className="panel side-panel">
            <div className="badges" style={{ marginBottom: 8 }}>
              {badges ? <BadgesBar data={badges} /> : <div className="small">Chargement…</div>}
            </div>

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
              Stations proches · prévision {forecastHourParis}
            </h3>

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
      </div>

      {/* Footer global */}
      <GlobalFooter />
    </>
  );
}
