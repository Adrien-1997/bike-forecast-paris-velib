import { useEffect, useMemo, useRef, useState } from 'react';
import Head from 'next/head';
import dynamic from 'next/dynamic';

// ─────────────── UI Components ───────────────
import BadgesBar from '@/components/Badges';

// ─────────────── Data Layer ───────────────
import { getBadges } from '@/lib/services/badges';
import { getForecastBatch } from '@/lib/services/forecast';
import { getStations } from '@/lib/services/stations';

// ─────────────── Types ───────────────
import type { Station, Forecast } from '@/lib/types';
import type { Map as LeafletMap } from 'leaflet';

// ─────────────── Map Component (no SSR) ───────────────
const MapView = dynamic(() => import('../components/MapView'), { ssr: false });


// ─────────────── Helpers ───────────────
const toNumber = (x: unknown, fallback = 0) => {
  const n = Number(x);
  return Number.isFinite(n) ? n : fallback;
};
const getPred = (f?: Forecast) => (f ? toNumber((f as any).bikes_pred_t15, 0) : 0);

// Parse timestamp
const parseIsoToMs = (v: unknown): number => {
  if (!v) return 0;
  const t = Date.parse(String(v));
  return Number.isFinite(t) ? t : 0;
};

// Compute freshness (minutes)
const computeFreshnessMin = (updatedAt: unknown): number => {
  const t = parseIsoToMs(updatedAt);
  if (!t) return 0;
  const now = Date.now();
  const diffMs = Math.max(0, now - t); // avoid negative when TZ in future
  return Math.floor(diffMs / 60000);
};

// Normalize badges freshness
// Normalize badges freshness
const normalizeBadges = (bd: any) => {
  if (!bd) return bd;

  const serverFresh =
    bd?.meta?.freshness_min ??
    bd?.freshness?.age_minutes ??
    bd?.freshness_min ??
    0;

  // ← on prend le parquet_ts comme updated_at par défaut (cohérent avec age_minutes)
  const updatedAt =
    bd?.meta?.updated_at ??
    bd?.freshness?.parquet_ts_utc ??
    bd?.weather?.ts_utc ??
    bd?.updated_at ??
    bd?.ts ??
    null;

  const computed = computeFreshnessMin(updatedAt);
  const finalFresh = Number(serverFresh) > 0 ? Number(serverFresh) : computed;

  // petit check : si l’écart client/serveur > 2 min, log pour enquête
  if (Number.isFinite(finalFresh) && Number.isFinite(computed)) {
    const delta = Math.abs(finalFresh - computed);
    if (delta > 2) {
      console.warn('[badges] freshness mismatch', { server: finalFresh, client: computed, updatedAt });
    }
  }

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
};


export default function HomePage() {
  const [stations, setStations] = useState<Station[]>([]);
  const [forecast, setForecast] = useState<Forecast[]>([]);
  const [badges, setBadges] = useState<any>(null);
  const [center] = useState<[number, number]>([48.8566, 2.3522]);
  const [userPos, setUserPos] = useState<[number, number] | null>(null);
  const mapRef = useRef<LeafletMap | null>(null);
  const NEARBY_LIMIT = 10;

  // ─────────────── Chargement et refresh ───────────────
  useEffect(() => {
    let alive = true;

      const loadOnce = async () => {
        try {
          console.debug('[loadOnce] start');

          // 1) Badges d'abord (affichage immédiat)
          try {
            const bd = await getBadges();
            const next = normalizeBadges(bd);
            console.debug('[badges:first]', {
              updated_at: next?.meta?.updated_at,
              server: next?.meta?.freshness_min_server,
              client: next?.meta?.freshness_min_client,
              used: next?.meta?.freshness_min_norm,
            });
            if (!alive) return;
            setBadges(next);
          } catch (e) {
            console.warn('[badges:first] failed', e);
          }

          // 2) Stations + Forecast (en parallèle)
          const st = await getStations();
          if (!alive) return;
          console.debug('[stations]', Array.isArray(st) ? st.length : -1);
          setStations(st ?? []);

          const codes = (st ?? []).map(s => String((s as any).stationcode));
          const [fc] = await Promise.all([
            getForecastBatch(codes, 15),
            // (optionnel) un 2e fetch badges pour rafraîchir si le temps a passé :
            // getBadges().then(bd => setBadges(normalizeBadges(bd))).catch(()=>{})
          ]);
          if (!alive) return;
          console.debug('[forecast]', Array.isArray(fc) ? fc.length : -1);
          setForecast(fc ?? []);
        } catch (e) {
          console.error(e);
        }
      };

    loadOnce();

    // refresh toutes les 5 min
    const FIVE_MIN = 300_000;
    const id = setInterval(loadOnce, FIVE_MIN);

    // refresh à la remise au premier plan
    const onVis = () => {
      if (document.visibilityState === 'visible') loadOnce();
    };
    document.addEventListener('visibilitychange', onVis);

    return () => {
      alive = false;
      clearInterval(id);
      document.removeEventListener('visibilitychange', onVis);
    };
  }, []);

  // ─────────────── Géoloc utilisateur ───────────────
  useEffect(() => {
    if (typeof navigator !== 'undefined' && navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        pos => setUserPos([pos.coords.latitude, pos.coords.longitude]),
        err => console.warn(err),
        { enableHighAccuracy: true }
      );
    }
  }, []);

  // ─────────────── Calculs dérivés ───────────────
  const stationsWithGeo = useMemo(
    () =>
      stations.filter(
        s =>
          typeof (s as any).lat === 'number' &&
          Number.isFinite((s as any).lat) &&
          typeof (s as any).lon === 'number' &&
          Number.isFinite((s as any).lon)
      ),
    [stations]
  );

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

  const nearby = useMemo(() => {
    if (!userPos || !stationsWithGeo.length || !Array.isArray(forecast)) return [];
    return stationsWithGeo
      .map((s: any) => {
        const pred = getPred(forecastByCode.get(String(s.stationcode)));
        const d = Math.hypot(Number(s.lat) - userPos[0], Number(s.lon) - userPos[1]);
        return { ...s, pred, distance: d };
      })
      .filter(s => s.pred >= 2)
      .sort((a, b) => a.distance - b.distance)
      .slice(0, NEARBY_LIMIT);
  }, [stationsWithGeo, forecast, forecastByCode, userPos]);

  // ─────────────── UI ───────────────
  return (
    <div className="container">
      <Head>
        <title>Vélib’ Paris — Disponibilités en temps réel et prévisions à 15 minutes</title>
        <meta
          name="description"
          content="Consultez en direct les vélos disponibles dans chaque station Vélib’ à Paris et découvrez les prévisions à 15 minutes."
        />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <header className="header">
        <h1>
          <span>Vélib’ Paris</span>
          <span className="subtitle">Disponibilités en temps réel · Prévisions à +15 min</span>
        </h1>
      </header>

      <main className="main">
        <div className="panel map-card">
          <div className="map-fill">
            <MapView
              stations={stationsWithGeo as Station[]}
              forecast={forecast}
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
            {badges ? (
              <BadgesBar data={badges} />
            ) : (
              <div className="small">Badges en cours de chargement…</div>
            )}
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

          <h3 style={{ margin: '20px 0 10px', fontSize: '1rem' }}>Stations proches</h3>
          <div className="list">
            {nearby.map(s => (
              <div
                key={(s as any).stationcode}
                className="row"
                style={{ background: 'rgba(255,255,255,0.03)', marginBottom: 6, padding: 8 }}
              >
                <div>
                  <div style={{ fontWeight: 700 }}>{(s as any).name ?? (s as any).stationcode}</div>
                  <div className="small">#{(s as any).stationcode}</div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div className="small">T+15</div>
                  <div style={{ fontWeight: 700, fontSize: '1.1rem' }}>
                    {getPred(forecastByCode.get(String((s as any).stationcode)))}
                  </div>
                </div>
              </div>
            ))}
            {!nearby.length && <div className="small">Aucune station proche avec ≥2 vélos prévus</div>}
          </div>
        </aside>
      </main>

      <footer className="footer footer--sticky">
        Fait avec ❤️ par{' '}
        <a href="https://www.linkedin.com/in/adrien-morel/" target="_blank" rel="noopener noreferrer">
          Adrien
        </a>
      </footer>
    </div>
  );
}
