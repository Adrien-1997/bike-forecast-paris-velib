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
  // Backend ≥ 1.3
  horizon_min?: number;
  window_days?: number | null;
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
  // Backend ≥ 1.3
  horizon_min?: number;
  window_days?: number | null;
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
  // Backend ≥ 1.3
  horizon_min?: number;
  window_days?: number | null;
};

export type UncertaintyDoc = {
  schema_version: string;
  generated_at: string;
  coverage: { empirical: number; n: number } | null;
  method?: string;
  nominal?: number | null;
  // Backend ≥ 1.3
  horizon_min?: number;
  window_days?: number | null;
};

/* ── Feature Importance ───────────────────────────────────
   Compat:
   - Ancien (surrogate RF): rows = { feature, importance, std }
   - Nouveau (XGBoost native): rows = { feature, gain, weight, cover }
*/
export type FeatureImportanceRow = {
  feature: string;

  // Surrogate RF (ancien)
  importance?: number; // delta MAE (positif)
  std?: number;        // écart-type permutations

  // XGBoost natif (nouveau)
  gain?: number;       // moyenne du gain
  weight?: number;     // nombre de splits
  cover?: number;      // couverture moyenne
};

export type FeatureImportanceDoc = {
  schema_version: string;
  generated_at: string;
  horizon_min: number;
  // Étend l’union pour couvrir les deux modes
  method:
    | "xgboost_native"
    | "surrogate_rf_permutation"
    | "disabled"
    | "unavailable_sklearn"
    | "fit_error"
    | "permutation_error"
    | "no_features"
    | "no_data";
  rows: FeatureImportanceRow[];
  n_features: number;
  n_rows: number;
  notes: string[];
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

export const getExplainFeatureImportance = (h: number) =>
  fetchJsonWithEtag<FeatureImportanceDoc>(path(`/feature_importance?h=${encodeURIComponent(h)}`));
