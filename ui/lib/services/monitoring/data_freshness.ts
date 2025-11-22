// ui/lib/services/monitoring/data_freshness.ts
// Service pour /monitoring/data/freshness (ETag + base/token gérés par lib/http)

import { fetchJsonWithEtag } from "@/lib/http";

/* ───────────────────────── Types ───────────────────────── */
export type FreshnessDoc = {
  now_utc: string;
  stations: {
    count: number;
    freshness: {
      p50_min: number | null;
      p95_min: number | null;
      max_min: number | null;
    };
    top_oldest?: Array<{ station_id: number | string; freshness_min: number }>;
  };
  weather?: { freshness_min: number | null };
  meta?: { bin_t_utc?: string; schema?: string };
};

/* ───────────────────────── Helpers ───────────────────────── */
const path = (suffix = "") => `/monitoring/data/freshness${suffix}`;

/* ───────────────────────── API (ETag) ───────────────────────── */
export const getDataFreshnessLatest = () =>
  fetchJsonWithEtag<FreshnessDoc>(path());

/* ───────────────────────── Selectors utiles ───────────────────────── */
export const selectFreshnessP95 = (d?: FreshnessDoc | null) =>
  (d?.stations?.freshness?.p95_min ?? null);

export const selectFreshnessP50 = (d?: FreshnessDoc | null) =>
  (d?.stations?.freshness?.p50_min ?? null);

export const selectFreshnessMetaBin = (d?: FreshnessDoc | null) =>
  d?.meta?.bin_t_utc ?? null;