// ui/lib/types.ts

// ─────────────────── Stations ───────────────────
export type Station = {
  station_id: string
  name?: string
  lat?: number
  lon?: number
  capacity?: number
  num_bikes_available?: number
  num_docks_available?: number
}

export type Forecast = {
  station_id?: string
  tbin_latest?: string | null
  tbin_utc?: string | null
  pred_ts_utc?: string | null     // moment où le modèle a tourné
  target_ts_utc?: string | null   // horodatage de la cible prévisionnelle (tbin_latest + horizon)
  horizon_min: number
  bikes_pred?: number             // prédiction brute (float)
  bikes_pred_int?: number         // prédiction arrondie (entier)
  capacity_bin?: number           // capacité de la station
  model_version?: string | null   // version ou nom du modèle
}

// ─────────────────── Weather ───────────────────
// /weather/live (données Open-Meteo ou null)
export type LiveWeather = {
  ts_utc?: string | null
  temp_C?: number | null
  precip_mm?: number | null
  wind_mps?: number | null
} | null

// ─────────────────── Badges (UI) ───────────────────
// Construit côté client à partir de météo + fraîcheur
export type Badges = {
  weather?: {
    ts_utc?: string | null
    temp_C?: number | null
    precip_mm?: number | null
    wind_mps?: number | null
  } | null
  freshness?: {
    age_minutes?: number | null
  } | null
  meta?: {
    pred_ts_utc?: string | null    // timestamp du modèle
    forecast_hour?: string | null  // heure locale affichée
    freshness_min?: number | null  // âge des données
  } | null
}
