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

// ─────────────────── Monitoring — Manifest ───────────────────
export interface MonitoringManifestItem {
  path: string;
  bucket: string;
  key: string;
  name: string;
  size: number;
  category: "docs" | "drift" | "model_perf" | "network" | "other";
  updated: number;        // epoch seconds (float)
  updated_iso: string;    // ISO
}

export interface MonitoringManifestSummaryEntry {
  base: string;
  count: number;
  latest: string;         // gs://...
  latest_updated: string; // ISO
}

export interface MonitoringManifestSummary {
  docs?: { resources: MonitoringManifestSummaryEntry[] };
  drift?: { resources: MonitoringManifestSummaryEntry[] };
  model_perf?: { resources: MonitoringManifestSummaryEntry[] };
  network?: { resources: MonitoringManifestSummaryEntry[] };
  other?: { resources: MonitoringManifestSummaryEntry[] };
}

export interface MonitoringManifest {
  schema_version: string;   // "1.0"
  generated_at: string;     // ISO
  root: string;             // gs://...
  total_items: number;
  items: MonitoringManifestItem[];
  summary: MonitoringManifestSummary;
}

// ─────────────────── Monitoring — Model perf (daily) ─────────
export interface PerfDailyMetric {
  date: string;                 // "YYYY-MM-DD"
  mae_model: number | null;
  mae_baseline: number | null;
  lift_vs_baseline: number | null;
  rmse_model: number | null;
  coverage_pred_pct: number;    // 0..1
  n: number;
}

export interface PerfDailyResponse {
  schema_version: string;       // "1.0"
  generated_at: string;         // ISO
  horizon_min: number;          // 15 | 60 | ...
  metrics: PerfDailyMetric[];
}


// ─────────────────── Monitoring — Model perf (segments) ──────────────────────
// Format attendu (observé côté UI et manifest) : tableau plat d'objets
export interface PerfSegment {
  segment: string;                 // ex: "hour=8", "dow=2", "capacity=20-30"
  rmse: number | null;
  mae: number | null;
  n: number;                       // effectif
}
// L’API renvoie généralement un tableau de PerfSegment
export type PerfSegmentsResponse = PerfSegment[];


// ─────────────────── Monitoring — Drift (summary) ───────────────────
// Schéma observé sur /monitoring/drift/summary

export interface DriftFeatureRow {
  feature: string;
  psi?: number | null;       // Population Stability Index
  ks?: number | null;        // Kolmogorov–Smirnov
  p_value?: number | null;   // éventuel p-value (KS/AD)
  drifted?: boolean | null;  // indicateur de drift si dispo
}

export interface DriftSummary {
  schema_version?: string;         // "1.0"
  generated_at?: string;           // ISO
  reference_window?: string | null; // ex: "2025-09-25..2025-10-01"
  current_window?: string | null;   // ex: "2025-10-02..2025-10-08"
  features?: DriftFeatureRow[];      // liste des features avec métriques
}

// Alias pratique pour services/pages
export type DriftSummaryResponse = DriftSummary | null;

