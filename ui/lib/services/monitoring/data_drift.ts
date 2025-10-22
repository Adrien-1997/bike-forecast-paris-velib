// ui/lib/services/monitoring/data_drift.ts
// Service pour /monitoring/data/drift (ETag + base/token gérés par lib/http)

import { fetchJsonWithEtag } from "@/lib/http";

/* ───────────────────────── Types ───────────────────────── */
export type DriftSummary = {
  schema_version: string;
  generated_at: string;
  psi_global: number | null;
  top_feature?: string | null;
  top_feature_psi?: number | null;
};

export type RowPSI = { feature: string; psi: number | null };
export type RowKS = { feature: string; ks: number | null };
export type RowDelta = { feature: string; delta_mean: number | null; delta_var: number | null };
export type EmaPoint = { date_local: string; psi_ema: number | null };

export type ZonesDoc = {
  rows: Array<{ zone: string | null; psi: number | null; lat?: number | null; lon?: number | null }>;
};

/* ───────────────────────── Helpers ───────────────────────── */
const path = (suffix: string) => `/monitoring/data/drift${suffix}`;

/* ───────────────────────── API (ETag) ───────────────────────── */
export const getDataDriftSummary = () =>
  fetchJsonWithEtag<DriftSummary>(path(""));

export const getDataDriftPsiByFeature = () =>
  fetchJsonWithEtag<RowPSI[]>(path("/psi_by_feature"));

export const getDataDriftKsByFeature = () =>
  fetchJsonWithEtag<RowKS[]>(path("/ks_by_feature"));

export const getDataDriftDeltasByFeature = () =>
  fetchJsonWithEtag<RowDelta[]>(path("/deltas_by_feature"));

export const getDataDriftPsiGlobalDailyEma = () =>
  fetchJsonWithEtag<EmaPoint[]>(path("/psi_global_daily_ema"));

export const getDataDriftZones = () =>
  fetchJsonWithEtag<ZonesDoc>(path("/zones"));
