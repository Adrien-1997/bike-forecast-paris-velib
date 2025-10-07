// ui/components/MapView.tsx
import dynamic from 'next/dynamic'
import { useMemo } from 'react'
import type { Station, Forecast } from '@/lib/types'

// Leaflet côté client uniquement
let L: typeof import('leaflet') | null = null
if (typeof window !== 'undefined') {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  L = require('leaflet')
}

// ───────────────────────────────────────────────────────────────
// Typage exporté → utilisé par dynamic() (ex: pages)
// ───────────────────────────────────────────────────────────────
export type MapViewProps = {
  stations: Station[]
  forecast: Forecast[] | any[] // tolérant côté UI
  mode: 'current' | 't15'
  center: [number, number]
}

// SSR-safe dynamic imports
const MapContainer = dynamic(() => import('react-leaflet').then(m => m.MapContainer), { ssr: false })
const TileLayer     = dynamic(() => import('react-leaflet').then(m => m.TileLayer),     { ssr: false })
const Marker        = dynamic(() => import('react-leaflet').then(m => m.Marker),        { ssr: false })
const Popup         = dynamic(() => import('react-leaflet').then(m => m.Popup),         { ssr: false })
const MarkerClusterGroup = dynamic(
  () => import('react-leaflet-cluster').then(m => m.default),
  { ssr: false },
)

// Helpers
function toNumber(x: unknown, fallback = 0): number {
  const n = Number(x)
  return Number.isFinite(n) ? n : fallback
}

function ringColor(ratio: number): string {
  if (Number.isNaN(ratio)) return '#9aa0a6'
  if (ratio >= 0.75) return '#1a9641'
  if (ratio >= 0.50) return '#a6d96a'
  if (ratio >= 0.25) return '#fdae61'
  return '#d7191c'
}

// Type guard station avec géo
function hasGeo(s: Station): s is Station & { lat: number; lon: number } {
  return typeof s.lat === 'number' && Number.isFinite(s.lat)
      && typeof s.lon === 'number' && Number.isFinite(s.lon)
}

// clé station (id prioritaire puis code)
const keyFor = (obj: any): string | null => {
  if (!obj) return null
  const id = obj.station_id ?? obj.stationId ?? null
  const code = obj.stationcode ?? obj.code ?? null
  if (id != null && String(id).trim() !== '') return String(id)
  if (code != null && String(code).trim() !== '') return String(code)
  return null
}

// lecture robuste du champ de prédiction
const getPred = (f: any): number =>
  toNumber(f?.bikes_pred_int ?? f?.bikes_pred ?? f?.bikes_pred_t15, 0)

export default function MapView({ stations, forecast, mode, center }: MapViewProps) {
  // index forecast par station_id ET stationcode
  const forecastByKey = useMemo(() => {
    const m = new Map<string, any>()
    for (const f of (forecast ?? []) as any[]) {
      const id = f?.station_id
      const code = f?.stationcode
      if (id != null)   m.set(String(id), f)
      if (code != null) m.set(String(code), f)
    }
    return m
  }, [forecast])

  const stationsWithGeo = useMemo(() => (stations ?? []).filter(hasGeo), [stations])

  const markers = useMemo(() =>
    stationsWithGeo.map(s => {
      const k = keyFor(s as any) ?? String(s.station_id)
      const f = k ? forecastByKey.get(k) : undefined

      const current = toNumber(s.num_bikes_available, 0)
      const pred    = getPred(f)
      const value   = mode === 'current' ? current : pred

      const cap = Math.max(1, toNumber(s.capacity, 0))
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

      const divIcon = L
        ? L.divIcon({
            html: iconHtml,
            className: '',
            iconSize: [size, size + 12],
            iconAnchor: [size / 2, size + 12],
          })
        : undefined

      const delta = pred - current

      return (
        <Marker key={String(s.station_id)} position={[s.lat, s.lon]} icon={divIcon as any}>
          <Popup>
            <div style={{ minWidth: 220 }}>
              <div style={{ fontWeight: 700 }}>{s.name ?? s.station_id}</div>
              <div className="small">#{s.station_id} • Cap: {cap}</div>
              <div style={{ marginTop: 6 }}>
                <div>Actuel: <b>{current}</b></div>
                <div>
                  Prévision T+15: <b>{pred}</b>
                  {delta !== 0 && (
                    <span style={{ marginLeft: 8, color: delta >= 0 ? 'var(--good)' : 'var(--bad)' }}>
                      {delta >= 0 ? `+${delta}` : `${delta}`}
                    </span>
                  )}
                </div>
              </div>
            </div>
          </Popup>
        </Marker>
      )
    })
  , [stationsWithGeo, forecastByKey, mode])

  return (
    <MapContainer
      center={center}
      zoom={13}
      scrollWheelZoom
      style={{ width: '100%', height: '100%', borderRadius: 14 }}
    >
      <TileLayer
        attribution="&copy; OpenStreetMap contributors"
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <MarkerClusterGroup chunkedLoading>
        {markers}
      </MarkerClusterGroup>
    </MapContainer>
  )
}
