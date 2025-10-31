// ui/lib/services/monitoring/model_performance.ts
// Service pour /monitoring/model/performance (ETag + base/token gérés par lib/http)

import { fetchJsonWithEtag } from "@/lib/http";

/* ── Types ─────────────────────────────────────────────── */
export type Manifest = {
  schema_version: string;
  generated_at: string;
  latest_prefix: string;
  window_days: number;
  horizons: number[];
};

export type KPIs = {
  schema_version: string;
  generated_at: string;
  window_days: number;
  horizon_min: number | null;
  coverage_pred_pct: number | null;
  mae_model: number | null;
  rmse_model: number | null;
  me_model: number | null;
  mae_baseline: number | null;
  rmse_baseline: number | null;
  me_baseline: number | null;
  lift_vs_baseline: number | null; // 0..1
  n_rows: number;
  n_stations: number;
  ts_min_utc: string | null;
  ts_max_utc: string | null;
};

export type DailyRow = {
  date: string;
  mae_model: number | null;
  mae_baseline: number | null;
  rmse_model: number | null;
  rmse_baseline: number | null;
  coverage_pred_pct: number | null;
  lift_vs_baseline: number | null;
  n: number;
};
export type DailyMetrics = { schema_version: string; horizon_min: number; rows: DailyRow[] };

export type HourRow = {
  hour: number;
  mae_model: number | null;
  mae_baseline: number | null;
  coverage_pred_pct: number | null;
  n: number;
};
export type ByHour = { schema_version: string; horizon_min: number; rows: HourRow[] };

export type DOWRow = {
  dow: number;
  mae_model: number | null;
  mae_baseline: number | null;
  coverage_pred_pct: number | null;
  n: number;
};
export type ByDow = { schema_version: string; horizon_min: number; rows: DOWRow[] };

export type StationRow = {
  station_id: string;
  mae_model: number | null;
  mae_baseline: number | null;
  coverage_pred_pct: number | null;
  n: number;
  lift_vs_baseline: number | null;
};
export type ByStation = { schema_version: string; horizon_min: number; rows: StationRow[] };

export type LiftCurve = {
  schema_version: string;
  horizon_min?: number;
  points: Array<{ date: string; lift_vs_baseline: number | null }>;
};
export type HistResiduals = {
  schema_version: string;
  horizon_min?: number;
  bins: number[];
  counts: number[];
  n: number;
};

/* ⬇️ NEW — Série 24 h Observé / Modèle / Baseline pour une station */
export type StationTimeseries = {
  schema_version: string;
  generated_at: string;
  h: number;
  station_id: string;
  name?: string | null;
  tz?: string;
  ts: string[];       // ISO8601 UTC
  y_true: number[];   // Observé
  y_pred: number[];   // Modèle
  y_base: number[];   // Baseline
};

/* ── Helpers ───────────────────────────────────────────── */
const path = (s: string) => `/monitoring/model/performance${s}`;

/* ── API (ETag) ────────────────────────────────────────── */
export const getPerformanceManifest = () =>
  fetchJsonWithEtag<Manifest>(path("/manifest"));

export const getPerformanceKpis = (h: number) =>
  fetchJsonWithEtag<KPIs>(path(`/kpis?h=${encodeURIComponent(h)}`));

export const getPerformanceDailyMetrics = (h: number) =>
  fetchJsonWithEtag<DailyMetrics>(path(`/daily_metrics?h=${encodeURIComponent(h)}`));

export const getPerformanceByHour = (h: number) =>
  fetchJsonWithEtag<ByHour>(path(`/by_hour?h=${encodeURIComponent(h)}`));

export const getPerformanceByDow = (h: number) =>
  fetchJsonWithEtag<ByDow>(path(`/by_dow?h=${encodeURIComponent(h)}`));

export const getPerformanceByStation = (h: number) =>
  fetchJsonWithEtag<ByStation>(path(`/by_station?h=${encodeURIComponent(h)}`));

export const getPerformanceLiftCurve = (h: number) =>
  fetchJsonWithEtag<LiftCurve>(path(`/lift_curve?h=${encodeURIComponent(h)}`));

export const getPerformanceHistResiduals = (h: number) =>
  fetchJsonWithEtag<HistResiduals>(path(`/hist_residuals?h=${encodeURIComponent(h)}`));

/* ⬇️ NEW */
export const getPerformanceStationTimeseries = (h: number, at?: string | null) => {
  const q = new URLSearchParams({ h: String(h) });
  if (at) q.set("at", at);
  return fetchJsonWithEtag<StationTimeseries>(path(`/station_timeseries?${q.toString()}`));
};
