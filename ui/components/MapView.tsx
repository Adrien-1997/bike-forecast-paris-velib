import dynamic from 'next/dynamic';
import { useMemo, useEffect, useState } from 'react';
import { useMap } from 'react-leaflet';
import type { Station, Forecast } from '@/lib/types';
import type * as LType from 'leaflet';

let L: typeof LType | null = null;
if (typeof window !== 'undefined') {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  L = require('leaflet');
}

const MapContainer = dynamic(() => import('react-leaflet').then(m => m.MapContainer), { ssr: false });
const TileLayer = dynamic(() => import('react-leaflet').then(m => m.TileLayer), { ssr: false });
const Marker = dynamic(() => import('react-leaflet').then(m => m.Marker), { ssr: false });
const Popup = dynamic(() => import('react-leaflet').then(m => m.Popup), { ssr: false });
const ZoomControl = dynamic(() => import('react-leaflet').then(m => m.ZoomControl), { ssr: false });
const MarkerClusterGroup = dynamic(() => import('react-leaflet-cluster').then(m => m.default), { ssr: false });

export type MapViewProps = {
  stations: Station[];
  forecast: Forecast[];
  mode: 'current' | 't15';
  center: [number, number];
  userPos?: [number, number] | null;
  setMapInstance?: (m: LType.Map) => void;
};

function MapInstanceBridge({ setMapInstance }: { setMapInstance?: (m: LType.Map) => void }) {
  const map = useMap();
  useEffect(() => {
    if (setMapInstance) setMapInstance(map);
  }, [map, setMapInstance]);
  return null;
}

function MapLocateControl({ userPos }: { userPos?: [number, number] | null }) {
  const map = useMap();
  if (!userPos) return null;
  return (
    <button
      className="map-fab"
      onClick={() => {
        const targetZoom = Math.max(map.getZoom(), 17);
        map.flyTo(userPos, targetZoom, { animate: true, duration: 1.2 });
      }}
      aria-label="Centrer sur ma position"
      style={{
        position: 'absolute',
        left: 12,
        bottom: 12,
        zIndex: 1000,
        background: '#1b1d27',
        color: '#fff',
        border: '1px solid rgba(255,255,255,.12)',
        borderRadius: 999,
        padding: '10px 14px',
        fontWeight: 600,
        boxShadow: '0 10px 24px rgba(0,0,0,.35)',
        cursor: 'pointer',
      }}
    >
      📍 Ma position
    </button>
  );
}

// ─────────────── Helpers
function isFiniteNum(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

function ringColor(ratio: number): string {
  if (!Number.isFinite(ratio)) return '#9aa0a6';
  if (ratio >= 0.75) return '#1a9641';
  if (ratio >= 0.50) return '#a6d96a';
  if (ratio >= 0.25) return '#fdae61';
  return '#d7191c';
}

// ─────────────── Carte principale
export default function MapView({ stations, forecast, mode, center, userPos, setMapInstance }: MapViewProps) {
  const [isMobile, setIsMobile] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia?.('(max-width: 768px)');
    const handler = () => setIsMobile(!!mq?.matches);
    handler();
    mq?.addEventListener?.('change', handler);
    return () => mq?.removeEventListener?.('change', handler);
  }, []);

  // 1) Index forecast par station (la plus récente)
  const forecastByCode = useMemo(() => {
    const m = new Map<string, Forecast>();
    for (const f of Array.isArray(forecast) ? forecast : []) {
      const code = String((f as any).stationcode ?? (f as any).station_id ?? (f as any).code ?? '');
      if (!code) continue;
      const prev = m.get(code);
      if (!prev) {
        m.set(code, f);
      } else {
        const tPrev = Date.parse((prev as any).ts_utc ?? (prev as any).ts ?? '') || 0;
        const tNew = Date.parse((f as any).ts_utc ?? (f as any).ts ?? '') || 0;
        if (tNew >= tPrev) m.set(code, f);
      }
    }
    return m;
  }, [forecast]);

  // 2) Stations valides
  const stationsWithGeo = useMemo(
    () => (Array.isArray(stations) ? stations.filter(s => isFiniteNum(s.lat) && isFiniteNum(s.lon)) : []),
    [stations]
  );

  // 3) Marqueurs Vélib’
  const markers = useMemo(() => {
    return stationsWithGeo.map(s => {
      const fc = forecastByCode.get(String(s.stationcode));
      const predRaw =
        (fc as any)?.bikes_pred_t15 ??
        (fc as any)?.pred_bikes_t15 ??
        (fc as any)?.bikes_t15 ??
        (fc as any)?.yhat_t15 ??
        0;
      const pred = Math.max(0, Math.round(Number(predRaw) || 0));
      const current = Math.max(0, Math.round(Number(s.num_bikes_available) || 0));
      const cap = Math.max(1, Math.round(Number(s.capacity) || 0));
      const value = mode === 'current' ? current : pred;
      const occ = value / cap;
      const colRing = ringColor(occ);
      const deg = Math.min(360, Math.max(0, occ * 360));
      const size = 44;

      const iconHtml = `
        <div style="position:relative; width:${size}px; height:${size}px; filter: drop-shadow(0 1px 2px rgba(0,0,0,.25));">
          <div style="width:100%; height:100%; border-radius:50%;
                      background: conic-gradient(${colRing} ${deg}deg, #e5e7eb 0deg);
                      display:flex; align-items:center; justify-content:center;
                      border:2px solid #263238; box-sizing:border-box;">
            <div style="width:${size - 12}px; height:${size - 12}px; border-radius:50%; background:#fff;
                        display:flex; align-items:center; justify-content:center;
                        font: 700 ${Math.floor(size * 0.42)}px/1 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif; color:#263238;">
              ${value}
            </div>
          </div>
          <div style="position:absolute; left:50%; bottom:-12px; transform:translateX(-50%);
                      width:0; height:0; border-left:8px solid transparent; border-right:8px solid transparent;
                      border-top:12px solid #263238; opacity:.9;"></div>
        </div>`;

      const divIcon = L
        ? L.divIcon({
            html: iconHtml,
            className: '',
            iconSize: [size, size + 12],
            iconAnchor: [size / 2, size + 12],
          })
        : undefined;

      // Δ variation
      const deltaPct = cap > 0 ? ((pred - current) / cap) * 100 : 0;
      const deltaBadge =
        Math.abs(deltaPct) < 0.5
          ? `<span style="padding:.2rem .5rem;border-radius:8px;background:#e5e7eb;color:#111827;border:1px solid rgba(0,0,0,.08)">▬ 0%</span>`
          : deltaPct > 0
          ? `<span style="padding:.2rem .5rem;border-radius:8px;background:#dcfce7;color:#166534;border:1px solid rgba(0,0,0,.08)">▲ ${deltaPct.toFixed(0)}%</span>`
          : `<span style="padding:.2rem .5rem;border-radius:8px;background:#fee2e2;color:#991b1b;border:1px solid rgba(0,0,0,.08)">▼ ${deltaPct.toFixed(0)}%</span>`;

      const popupHtml = `
        <div style="font-family:Inter,system-ui,Segoe UI,Roboto,sans-serif;">
          <div style="font-weight:700;margin-bottom:.15rem">${s.name ?? s.stationcode}</div>
          <div style="color:#6b7280;margin-bottom:.35rem">#${s.stationcode}</div>
          <div style="display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:.35rem">
            <span style="padding:.2rem .5rem;border-radius:8px;background:#f3f4f6">
              Actuel&nbsp;: <b>${current}</b> / ${cap}
            </span>
            <span style="padding:.2rem .5rem;border-radius:8px;background:#eef2ff">
              Prévision T+15&nbsp;min&nbsp;: <b>${pred}</b>
            </span>
            ${deltaBadge}
          </div>
        </div>`;

      return (
        <Marker key={String(s.stationcode)} position={[s.lat!, s.lon!]} icon={divIcon}>
          <Popup maxWidth={300}>
            <div dangerouslySetInnerHTML={{ __html: popupHtml }} />
          </Popup>
        </Marker>
      );
    });
  }, [stationsWithGeo, forecastByCode, mode]);

  // 4) Icône de cluster
  const clusterIcon = (cluster: any) => {
    const count = cluster.getChildCount();
    const bg =
      count >= 100 ? '#263238' :
      count >= 50  ? '#37474f' :
      count >= 20  ? '#455a64' : '#546e7a';
    const size = count >= 100 ? 46 : count >= 50 ? 42 : count >= 20 ? 38 : 34;
    return L!.divIcon({
      html: `
        <div style="
          width:${size}px;height:${size}px;border-radius:50%;
          background:${bg};color:#fff;display:flex;align-items:center;justify-content:center;
          border:2px solid rgba(255,255,255,.85); font:700 ${Math.floor(size*0.42)}px/1 Inter,system-ui,sans-serif;
          box-shadow:0 6px 16px rgba(0,0,0,.25);
        ">${count}</div>
      `,
      className: 'cluster-icon',
      iconSize: [size, size],
    });
  };

  // 5) Marqueur utilisateur (point bleu + halo)
  const userMarker = useMemo(() => {
    if (!userPos || !L) return null;
    const size = 22;
    const html = `
      <div style="position:relative;width:${size}px;height:${size}px;">
        <div style="position:absolute;inset:0;border-radius:50%;background:#3b82f6;border:2px solid #ffffff;"></div>
        <div style="position:absolute;inset:-6px;border-radius:50%;background:rgba(59,130,246,.15);animation:pulse 2s infinite;"></div>
      </div>
      <style>
        @keyframes pulse {
          0% { transform: scale(0.9); opacity: 0.7; }
          70% { transform: scale(1.4); opacity: 0; }
          100% { transform: scale(0.9); opacity: 0; }
        }
      </style>`;
    const icon = L.divIcon({
      html,
      className: '',
      iconSize: [size, size],
      iconAnchor: [size / 2, size / 2],
    });
    return (
      <Marker key="userpos" position={userPos} icon={icon}>
        <Popup>Vous êtes ici</Popup>
      </Marker>
    );
  }, [userPos]);

  return (
    <MapContainer
      center={center}
      zoom={12}
      zoomControl={false}
      scrollWheelZoom={!isMobile}
      dragging={true}
      preferCanvas
      attributionControl={true}
      style={{ width: '100%', height: '100%', borderRadius: 14, position: 'relative' }}
      className="leaflet-container"
    >
      <TileLayer
        attribution="&copy; OpenStreetMap contributors & CartoDB"
        url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
      />

      <ZoomControl position="bottomright" />

      {/* CLUSTERING fluide */}
      <MarkerClusterGroup
        maxClusterRadius={70}
        disableClusteringAtZoom={19}
        showCoverageOnHover={false}
        spiderfyOnMaxZoom
        zoomToBoundsOnClick
        removeOutsideVisibleBounds
        animateAddingMarkers
        chunkedLoading
        iconCreateFunction={clusterIcon}
        boundsOptions={{ padding: [20, 20] }}
      >
        {markers}
      </MarkerClusterGroup>

      {userMarker}
      <MapInstanceBridge setMapInstance={setMapInstance} />
      <MapLocateControl userPos={userPos} />
    </MapContainer>
  );
}
