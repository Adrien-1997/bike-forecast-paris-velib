// ui/lib/services/monitoring/model_explainability.ts
// Service pour la page /monitoring/model/explainability
// → utilise lib/http.ts (token & base déjà gérés)

import { fetchJsonWithEtag } from "@/lib/http";

/* ───────────────────────── Types ───────────────────────── */
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

/* ───────────────────────── Helpers ───────────────────────── */
const path = (suffix: string) => `/monitoring/model/explainability${suffix}`;

/* ───────────────────────── API calls (ETag) ───────────────────────── */
export const getExplainOverview = () =>
  fetchJsonWithEtag<Overview>(path("/overview"));

export const getExplainResiduals = () =>
  fetchJsonWithEtag<ResidualsDoc>(path("/residuals"));

export const getExplainCalibration = () =>
  fetchJsonWithEtag<CalibrationDoc>(path("/calibration"));

export const getExplainUncertainty = () =>
  fetchJsonWithEtag<UncertaintyDoc>(path("/uncertainty"));
