// ui/lib/services/monitoring/model_explainability.ts
// Service pour /monitoring/model/explainability (ETag + base/token gérés par lib/http)

import { fetchJsonWithEtag } from "@/lib/http";

/* ── Types ─────────────────────────────────────────────── */
export type Overview = {
  schema_version: string;
  generated_at: string;
  tz: string;
  anchor_day_perf: string | null;
  perf_rows: number;
  perf_stations: number;
  ts_min_perf: string | null;
  ts_max_perf: string | null;
  has_y_pred: boolean;
  has_uncertainty: boolean;
};

export type ResidHistBin = { bin_left: number; bin_right: number; count: number };

export type ResidualsDoc = {
  schema_version: string;
  generated_at: string;
  hist: ResidHistBin[];
  qq: { th: number[]; emp: number[] };
  acf: number[];
  hetero: Array<{ quantile: string; mae: number; n: number }>;
  episodes: Array<{ station_id: string; max_run: number; n: number }>;
};

export type CalibrationDoc = {
  schema_version: string;
  generated_at: string;
  fit: { alpha: number | null; beta: number | null; n: number };
  binned: Array<{ quantile: string; y_pred_mean: number; y_true_mean: number; n: number }>;
  by_hour: Array<{ hour: number; alpha: number | null; beta: number | null; n: number }>;
  rel_error_levels: Array<{ level: string; mape_like: number; n: number }>;
  bias_by_station: Array<{
    station_id: string;
    name: string | null;
    bias: number | null;
    lat: number | null;
    lon: number | null;
    n: number;
  }>;
};

export type UncertaintyDoc = {
  schema_version: string;
  generated_at: string;
  coverage: { empirical: number; n: number } | null;
  method?: string;
  nominal?: number;
};

/* ── Helpers ───────────────────────────────────────────── */
const path = (s: string) => `/monitoring/model/explainability${s}`;

/* ── API (ETag) ────────────────────────────────────────── */
export const getExplainOverview = (h: number) =>
  fetchJsonWithEtag<Overview>(path(`/overview?h=${encodeURIComponent(h)}`));

export const getExplainResiduals = (h: number) =>
  fetchJsonWithEtag<ResidualsDoc>(path(`/residuals?h=${encodeURIComponent(h)}`));

export const getExplainCalibration = (h: number) =>
  fetchJsonWithEtag<CalibrationDoc>(path(`/calibration?h=${encodeURIComponent(h)}`));

export const getExplainUncertainty = (h: number) =>
  fetchJsonWithEtag<UncertaintyDoc>(path(`/uncertainty?h=${encodeURIComponent(h)}`));
