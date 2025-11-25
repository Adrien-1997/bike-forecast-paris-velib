// components/app/MapView.tsx
//
// =============================================================================
// Carte principale de l’application (Vue "App").
// Affiche :
// - l’ensemble des stations Vélib’ en cercles proportionnels (occupancy),
// - les prédictions (T+15 / T+60) ou la situation actuelle selon `mode`,
// - un clustering performant des marqueurs (react-leaflet-cluster),
// - un marqueur et un bouton "📍 Ma position" si la position utilisateur est connue,
// - un bridge pour exposer l’instance Leaflet au parent (`setMapInstance`).
//
// Points clés :
// - rendu uniquement côté client (SSR désactivé pour les composants Leaflet),
// - compatibilité avec plusieurs formats d’objets Forecast (bikes_pred / bikes_pred_int),
// - icônes HTML entièrement custom (conic-gradient + drop-shadow),
// - clusters sobres et lisibles, adaptés à des centaines de stations.
// =============================================================================

import dynamic from 'next/dynamic'
import { useMemo, useEffect, useState } from 'react'
import { useMap } from 'react-leaflet'
import type { Station, Forecast } from '@/lib/types/types'
import type * as LType from 'leaflet'

import 'leaflet.markercluster/dist/MarkerCluster.css'
import 'leaflet.markercluster/dist/MarkerCluster.Default.css'

/* Leaflet côté client uniquement (évite les erreurs `window` côté serveur) */
let L: typeof import('leaflet') | null = null
if (typeof window !== 'undefined') {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  L = require('leaflet') as typeof import('leaflet')
}

/* Dynamic imports des composants react-leaflet (SSR désactivé) */
const MapContainer       = dynamic(() => import('react-leaflet').then(m => m.MapContainer), { ssr: false })
const TileLayer          = dynamic(() => import('react-leaflet').then(m => m.TileLayer), { ssr: false })
const Marker             = dynamic(() => import('react-leaflet').then(m => m.Marker), { ssr: false })
const Popup              = dynamic(() => import('react-leaflet').then(m => m.Popup), { ssr: false })
const ZoomControl        = dynamic(() => import('react-leaflet').then(m => m.ZoomControl), { ssr: false })
const MarkerClusterGroup = dynamic(() => import('react-leaflet-cluster').then(m => m.default), { ssr: false })

export type MapViewProps = {
  stations: Station[]
  forecast: Forecast[]
  /**
   * Mode d’affichage :
   * - "current" : nombre de vélos actuels (num_bikes_available),
   * - "t15"     : prévision à 15 minutes,
   * - "t60"     : prévision à 60 minutes.
   */
  mode: 'current' | 't15' | 't60'
  /** Centre initial de la carte [lat, lon]. */
  center: [number, number]
  /** Position utilisateur (optionnelle), si connue. */
  userPos?: [number, number] | null
  /** Callback pour exposer l’instance Leaflet Map au parent. */
  setMapInstance?: (m: LType.Map) => void
}

/* ───────── helpers ───────── */

/**
 * Conversion robuste en nombre :
 * - `Number(x)` puis vérification de finitude,
 * - renvoie `def` si NaN / ±Inf.
 */
const toNum = (x: unknown, def = 0) => {
  const n = Number(x)
  return Number.isFinite(n) ? n : def
}

/**
 * Type guard : station possédant bien des coordonnées (lat, lon numériques).
 */
const hasGeo = (s: Station): s is Station & { lat: number; lon: number } =>
  typeof s.lat === 'number' && Number.isFinite(s.lat) &&
  typeof s.lon === 'number' && Number.isFinite(s.lon)

/**
 * Extrait une clé station "robuste" :
 * - supporte station_id, stationId, id,
 * - renvoie null si aucune clé propre n’est trouvée.
 */
const keyFor = (obj: any): string | null => {
  if (!obj) return null
  const id = obj.station_id ?? obj.stationId ?? obj.id ?? null
  return id != null && String(id).trim() !== '' ? String(id) : null
}

/**
 * Couleur du ring d’occupation en fonction du ratio bikes/capacity.
 * Palette inspirée des rampes heatmap (vert → rouge).
 */
const ringColor = (ratio: number): string => {
  if (!Number.isFinite(ratio)) return '#9aa0a6'   // gris neutre si inconnue
  if (ratio >= 0.75) return '#1a9641'            // très plein (vert foncé)
  if (ratio >= 0.50) return '#a6d96a'            // plein (vert clair)
  if (ratio >= 0.25) return '#fdae61'            // moyen (orange)
  return '#d7191c'                               // très vide (rouge)
}

/**
 * Fabrique un divIcon Leaflet à partir d’un snippet HTML et de sa géométrie.
 */
const makeDivIcon = (html: string, size: [number, number], anchor: [number, number]) =>
  L?.divIcon({ html, className: '', iconSize: size, iconAnchor: anchor })

/**
 * Retourne la prédiction entière (>= 0) à partir d’un objet Forecast :
 * - priorité à `bikes_pred_int`,
 * - fallback sur `bikes_pred`,
 * - clamp à 0 et arrondi.
 */
const getPred = (f?: Forecast | any): number =>
  Math.max(0, Math.round(toNum(f?.bikes_pred_int ?? f?.bikes_pred, 0)))

/* ───────── Sub-components ───────── */

/**
 * Bridge react-leaflet → parent :
 * - récupère l’instance Map via `useMap`,
 * - la pousse dans `setMapInstance` une fois disponible.
 */
function MapInstanceBridge({ setMapInstance }: { setMapInstance?: (m: LType.Map) => void }) {
  const map = useMap()
  useEffect(() => {
    if (setMapInstance) setMapInstance(map)
  }, [map, setMapInstance])
  return null
}

/**
 * Bouton flottant "📍 Ma position" :
 * - visible uniquement si `userPos` est fourni,
 * - recentre la carte sur l’utilisateur avec un zoom mini (17).
 */
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

  // Détection simple du mobile via media query (largeur ≤ 768px)
  // → utilisée pour désactiver le scrollWheelZoom sur mobile.
  useEffect(() => {
    const mq = window.matchMedia?.('(max-width: 768px)')
    const handler = () => setIsMobile(!!mq?.matches)
    handler()
    mq?.addEventListener?.('change', handler)
    return () => mq?.removeEventListener?.('change', handler)
  }, [])

  /**
   * Indexation des prévisions par station_id :
   * - construit une Map<string, Forecast>,
   * - en cas de doublon, garde le forecast le plus récent (pred_ts_utc / tbin_latest).
   */
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

  /** Filtrage des stations disposant de coordonnées valides. */
  const stationsWithGeo = useMemo(
    () => (Array.isArray(stations) ? stations.filter(hasGeo) : []),
    [stations]
  )

  /**
   * Construction des marqueurs :
   * - un divIcon circulaire par station,
   * - couleur de ring dépendant de l’occupation,
   * - valeur affichée = bikes actuels ou prédits selon le `mode`,
   * - popup HTML avec nom, id, actuel et prévision (horizon explicite).
   */
  const markers = useMemo(() => {
    const predLabel =
      mode === 't60' ? 'Prévision (60 min)' :
      mode === 't15' ? 'Prévision (15 min)' :
      'Prévision'

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

      // ✅ ROND ORIGINAL INTACT (icône circulaire avec ring conique + valeur centrale)
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

      // Fallback minimal en cas de problème de divIcon principal
      const divIcon =
        makeDivIcon(iconHtml, [size, size + 12], [size / 2, size + 12]) ??
        makeDivIcon(
          `<div style="font:700 12px/1 Inter;color:#111;background:#fff;border:1px solid #ddd;border-radius:8px;padding:2px 6px">${value}</div>`,
          [28, 18],
          [14, 9]
        )

      // ✅ Popup d’origine, avec libellé horizon explicite
      const popupHtml = `
        <div class="popup-content">
          <div class="popup-title">${(s as any).name ?? (s as any).station_id}</div>
          <div class="popup-sub">#${String((s as any).station_id ?? '')}</div>
          <div class="popup-row">
            <span class="badge current">Actuel&nbsp;: <b>${current}</b> / ${cap}</span>
            <span class="badge pred">${predLabel}&nbsp;: <b>${pred}</b></span>
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

  /**
   * Icône de cluster (react-leaflet-cluster) :
   * - cercle plein gris-bleu,
   * - taille légèrement croissante avec le nombre de stations,
   * - valeur au centre = nombre de marqueurs enfants.
   */
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

  /**
   * Marqueur "vous êtes ici" :
   * - pulsation CSS (voir .user-marker/.dot/.pulse),
   * - rendu uniquement si `userPos` et Leaflet sont disponibles.
   */
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
      {/* Fond de carte (CartoDB light) */}
      <TileLayer
        attribution="&copy; OpenStreetMap contributors & CartoDB"
        url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
      />

      {/* Zoom +/- en bas à droite (plus lisible avec le FAB position) */}
      <ZoomControl position="bottomright" />

      {/* Clusterisation des stations (avec icône custom) */}
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

      {/* Marqueur utilisateur, si disponible */}
      {userMarker}

      {/* Bridge & FAB "Ma position" */}
      <MapInstanceBridge setMapInstance={setMapInstance} />
      <MapLocateControl userPos={userPos} />
    </MapContainer>
  )
}
