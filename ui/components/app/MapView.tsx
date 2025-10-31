import dynamic from 'next/dynamic'
import { useMemo, useEffect, useState } from 'react'
import { useMap } from 'react-leaflet'
import type { Station, Forecast } from '@/lib/types/types'
import type * as LType from 'leaflet'

import 'leaflet.markercluster/dist/MarkerCluster.css'
import 'leaflet.markercluster/dist/MarkerCluster.Default.css'

/* Leaflet côté client uniquement */
let L: typeof import('leaflet') | null = null
if (typeof window !== 'undefined') {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  L = require('leaflet') as typeof import('leaflet')
}

/* Dynamic imports */
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

/* ───────── helpers ───────── */

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

/* ───────── Sub-components ───────── */

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
    >
      📍 Ma position
    </button>
  )
}

/* ───────── Main component ───────── */

export default function MapView({ stations, forecast, mode, center, userPos, setMapInstance }: MapViewProps) {
  const [isMobile, setIsMobile] = useState(false)
  useEffect(() => {
    const mq = window.matchMedia?.('(max-width: 768px)')
    const handler = () => setIsMobile(!!mq?.matches)
    handler()
    mq?.addEventListener?.('change', handler)
    return () => mq?.removeEventListener?.('change', handler)
  }, [])

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

  const stationsWithGeo = useMemo(
    () => (Array.isArray(stations) ? stations.filter(hasGeo) : []),
    [stations]
  )

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

      // ✅ ROND ORIGINAL INTACT
      const iconHtml = `
        <div style="position:relative;width:${size}px;height:${size}px;filter:drop-shadow(0 1px 2px rgba(0,0,0,.25));">
          <div style="width:100%;height:100%;border-radius:50%;
                      background:conic-gradient(${colRing} ${deg}deg, #e5e7eb 0deg);
                      display:flex;align-items:center;justify-content:center;
                      border:2px solid #263238;">
            <div style="width:${size - 12}px;height:${size - 12}px;border-radius:50%;background:#fff;
                        display:flex;align-items:center;justify-content:center;
                        font:700 ${Math.floor(size * 0.42)}px/1 'Inter',system-ui,sans-serif;color:#263238;">
              ${value}
            </div>
          </div>
          <div style="position:absolute;left:50%;bottom:-12px;transform:translateX(-50%);
                      width:0;height:0;border-left:8px solid transparent;border-right:8px solid transparent;
                      border-top:12px solid #263238;opacity:.9;"></div>
        </div>`

      const divIcon =
        makeDivIcon(iconHtml, [size, size + 12], [size / 2, size + 12]) ??
        makeDivIcon(`<div style="font:700 12px/1 Inter;color:#111;background:#fff;border:1px solid #ddd;border-radius:8px;padding:2px 6px">${value}</div>`, [28, 18], [14, 9])

      // ✅ Popup d’origine, juste sans Δ vélos
      const popupHtml = `
        <div class="popup-content">
          <div class="popup-title">${(s as any).name ?? (s as any).station_id}</div>
          <div class="popup-sub">#${String((s as any).station_id ?? '')}</div>
          <div class="popup-row">
            <span class="badge current">Actuel&nbsp;: <b>${current}</b> / ${cap}</span>
            <span class="badge pred">Prévision&nbsp;: <b>${pred}</b></span>
          </div>
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

  // ✅ Clusters sobres (gris-bleu neutre)
  const clusterIcon = (cluster: any) => {
    const count = cluster.getChildCount()
    const s =
      count >= 250 ? 50 :
      count >= 100 ? 46 :
      count >= 50  ? 42 :
      count >= 20  ? 38 : 34
    const fontPx = Math.floor(s * 0.42)

    const html = `
      <div style="
        width:${s}px;height:${s}px;border-radius:999px;
        display:flex;align-items:center;justify-content:center;
        background:#1c2535;
        border:1px solid rgba(255,255,255,.15);
        box-shadow:0 3px 10px rgba(0,0,0,.3);
        color:#e8ecf2;
        font:800 ${fontPx}px/1 Inter, system-ui, sans-serif;
      ">
        ${count}
      </div>`

    return (L as typeof import('leaflet')).divIcon({
      html,
      className: 'cluster-icon cluster-icon--neutral',
      iconSize: [s, s],
    })
  }

  const userMarker = useMemo(() => {
    if (!userPos || !L) return null
    const size = 22
    const html = `
      <div class="user-marker" style="width:${size}px;height:${size}px;">
        <div class="dot"></div><div class="pulse"></div>
      </div>`
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
      className="map-fill"
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
