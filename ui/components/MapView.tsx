import dynamic from 'next/dynamic'
import { useMemo, useEffect, useState } from 'react'
import { useMap } from 'react-leaflet'
import type { Station, Forecast } from '@/lib/types'
import type * as LType from 'leaflet'

/* Leaflet côté client uniquement */
let L: typeof import('leaflet') | null = null
if (typeof window !== 'undefined') {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  L = require('leaflet') as typeof import('leaflet')
}

/* Dynamic imports (SSR-safe) */
const MapContainer       = dynamic(() => import('react-leaflet').then(m => m.MapContainer), { ssr: false })
const TileLayer          = dynamic(() => import('react-leaflet').then(m => m.TileLayer), { ssr: false })
const Marker             = dynamic(() => import('react-leaflet').then(m => m.Marker), { ssr: false })
const Popup              = dynamic(() => import('react-leaflet').then(m => m.Popup), { ssr: false })
const ZoomControl        = dynamic(() => import('react-leaflet').then(m => m.ZoomControl), { ssr: false })
const MarkerClusterGroup = dynamic(() => import('react-leaflet-cluster').then(m => m.default), { ssr: false })

export type MapViewProps = {
  stations: Station[]
  forecast: Forecast[]
  mode: 'current' | 't15'
  center: [number, number]
  userPos?: [number, number] | null
  setMapInstance?: (m: LType.Map) => void
}

/* ───────────────── helpers ───────────────── */

const toNum = (x: unknown, def = 0) => {
  const n = Number(x)
  return Number.isFinite(n) ? n : def
}

const hasGeo = (s: Station): s is Station & { lat: number; lon: number } =>
  typeof s.lat === 'number' && Number.isFinite(s.lat) &&
  typeof s.lon === 'number' && Number.isFinite(s.lon)

const keyFor = (obj: any): string | null => {
  if (!obj) return null
  const id = obj.station_id ?? obj.stationId ?? obj.id ?? null
  return id != null && String(id).trim() !== '' ? String(id) : null
}

const ringColor = (ratio: number): string => {
  if (!Number.isFinite(ratio)) return '#9aa0a6'
  if (ratio >= 0.75) return '#1a9641'
  if (ratio >= 0.50) return '#a6d96a'
  if (ratio >= 0.25) return '#fdae61'
  return '#d7191c'
}

const makeDivIcon = (html: string, size: [number, number], anchor: [number, number]) =>
  L?.divIcon({ html, className: '', iconSize: size, iconAnchor: anchor })

const getPred = (f?: Forecast | any): number =>
  Math.max(0, Math.round(toNum(f?.bikes_pred_int ?? f?.bikes_pred, 0)))

/* ───────────────── sub-components ───────────────── */

function MapInstanceBridge({ setMapInstance }: { setMapInstance?: (m: LType.Map) => void }) {
  const map = useMap()
  useEffect(() => {
    if (setMapInstance) setMapInstance(map)
  }, [map, setMapInstance])
  return null
}

function MapLocateControl({ userPos }: { userPos?: [number, number] | null }) {
  const map = useMap()
  if (!userPos) return null
  return (
    <button
      className="map-fab"
      onClick={() => {
        const targetZoom = Math.max(map.getZoom(), 17)
        map.flyTo(userPos, targetZoom, { animate: true, duration: 1.2 })
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
  )
}

/* ───────────────── main component ───────────────── */

export default function MapView({ stations, forecast, mode, center, userPos, setMapInstance }: MapViewProps) {
  const [isMobile, setIsMobile] = useState(false)
  useEffect(() => {
    const mq = window.matchMedia?.('(max-width: 768px)')
    const handler = () => setIsMobile(!!mq?.matches)
    handler()
    mq?.addEventListener?.('change', handler)
    return () => mq?.removeEventListener?.('change', handler)
  }, [])

  // 1) Index des prévisions par station_id (string). Si doublons, on garde la plus récente via pred_ts_utc || tbin_latest.
  const forecastById = useMemo(() => {
    const m = new Map<string, Forecast>()
    for (const f of Array.isArray(forecast) ? forecast : []) {
      const id = keyFor(f as any)
      if (!id) continue
      const prev = m.get(id)
      if (!prev) {
        m.set(id, f)
      } else {
        const tPrev = Date.parse((prev as any).pred_ts_utc ?? prev.tbin_latest ?? '') || 0
        const tNew  = Date.parse((f as any).pred_ts_utc ?? f.tbin_latest ?? '') || 0
        if (tNew >= tPrev) m.set(id, f)
      }
    }
    return m
  }, [forecast])

  // 2) Filtrer les stations valides (avec géo)
  const stationsWithGeo = useMemo(
    () => (Array.isArray(stations) ? stations.filter(hasGeo) : []),
    [stations]
  )

  // 3) Marqueurs
  const markers = useMemo(() => {
    return stationsWithGeo.map(s => {
      const id = keyFor(s as any)
      const f  = id ? forecastById.get(id) : undefined

      const current = Math.max(0, Math.round(toNum((s as any).num_bikes_available, 0)))
      const pred    = getPred(f)
      const value   = mode === 'current' ? current : pred

      const cap = Math.max(1, Math.round(toNum((s as any).capacity, 0)))
      const occ = value / cap

      const colRing = ringColor(occ)
      const deg     = Math.min(360, Math.max(0, occ * 360))
      const size    = 44

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
        </div>`

      const divIcon =
        makeDivIcon(iconHtml, [size, size + 12], [size / 2, size + 12]) ??
        makeDivIcon(
          `<div style="font:700 12px/1 Inter;color:#111;background:#fff;border:1px solid #ddd;border-radius:8px;padding:2px 6px">${value}</div>`,
          [28, 18],
          [14, 9]
        )

      const delta = pred - current
      const deltaPct = cap > 0 ? ((pred - current) / cap) * 100 : 0
      const deltaBadge =
        Math.abs(deltaPct) < 0.5
          ? `<span style="padding:.2rem .5rem;border-radius:8px;background:#e5e7eb;color:#111827;border:1px solid rgba(0,0,0,.08)">▬ 0%</span>`
          : deltaPct > 0
          ? `<span style="padding:.2rem .5rem;border-radius:8px;background:#dcfce7;color:#166534;border:1px solid rgba(0,0,0,.08)">▲ ${deltaPct.toFixed(0)}%</span>`
          : `<span style="padding:.2rem .5rem;border-radius:8px;background:#fee2e2;color:#991b1b;border:1px solid rgba(0,0,0,.08)">▼ ${deltaPct.toFixed(0)}%</span>`

      const popupHtml = `
        <div style="font-family:Inter,system-ui,Segoe UI,Roboto,sans-serif;">
          <div style="font-weight:700;margin-bottom:.15rem">${(s as any).name ?? (s as any).station_id}</div>
          <div style="color:#6b7280;margin-bottom:.35rem">#${String((s as any).station_id ?? '')}</div>
          <div style="display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:.35rem">
            <span style="padding:.2rem .5rem;border-radius:8px;background:#f3f4f6">
              Actuel&nbsp;: <b>${current}</b> / ${cap}
            </span>
            <span style="padding:.2rem .5rem;border-radius:8px;background:#eef2ff">
              Prévision T+15&nbsp;: <b>${pred}</b>
            </span>
            ${deltaBadge}
          </div>
          ${Number.isFinite(delta) && delta !== 0 ? `<div style="color:${delta >= 0 ? '#166534' : '#991b1b'}">
            Δ vélos: <b>${delta >= 0 ? `+${delta}` : delta}</b>
          </div>` : ''}
        </div>`

      return (
        <Marker key={String((s as any).station_id)} position={[s.lat!, s.lon!]} icon={divIcon as any}>
          <Popup maxWidth={300}>
            <div dangerouslySetInnerHTML={{ __html: popupHtml }} />
          </Popup>
        </Marker>
      )
    })
  }, [stationsWithGeo, forecastById, mode])

  // 4) Icône de cluster
  const clusterIcon = (cluster: any) => {
    const count = cluster.getChildCount()
    const bg =
      count >= 100 ? '#263238' :
      count >= 50  ? '#37474f' :
      count >= 20  ? '#455a64' : '#546e7a'
    const s = count >= 100 ? 46 : count >= 50 ? 42 : count >= 20 ? 38 : 34

    return (L as typeof import('leaflet')).divIcon({
      html: `
        <div style="
          width:${s}px;height:${s}px;border-radius:50%;
          background:${bg};color:#fff;display:flex;align-items:center;justify-content:center;
          border:2px solid rgba(255,255,255,.85); font:700 ${Math.floor(s*0.42)}px/1 Inter,system-ui,sans-serif;
          box-shadow:0 6px 16px rgba(0,0,0,.25);
        ">${count}</div>
      `,
      className: 'cluster-icon',
      iconSize: [s, s],
    })
  }

  // 5) Marqueur utilisateur
  const userMarker = useMemo(() => {
    if (!userPos || !L) return null
    const size = 22
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
      </style>`
    const icon = L.divIcon({ html, className: '', iconSize: [size, size], iconAnchor: [size / 2, size / 2] })
    return (
      <Marker key="userpos" position={userPos} icon={icon as any}>
        <Popup>Vous êtes ici</Popup>
      </Marker>
    )
  }, [userPos])

  return (
    <MapContainer
      center={center}
      zoom={12}
      zoomControl={false}
      scrollWheelZoom={!isMobile}
      dragging
      preferCanvas
      attributionControl
      style={{ width: '100%', height: '100%', borderRadius: 14, position: 'relative' }}
      className="leaflet-container"
    >
      <TileLayer
        attribution="&copy; OpenStreetMap contributors & CartoDB"
        url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
      />

      <ZoomControl position="bottomright" />

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
  )
}
