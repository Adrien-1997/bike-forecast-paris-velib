// ui/lib/services/monitoring/network_overview.ts
// Service pour la page /monitoring/overview
// → utilise lib/http.ts (token & base déjà gérés)

import { fetchJsonWithEtag, getJSON } from "@/lib/http";

/* ───────────────────────── Types ───────────────────────── */
export type OverviewKpis = {
  schema_version: string;
  generated_at: string;
  snapshot_ts_utc: string;
  snapshot_ts_local: string;
  stations_universe: number;
  stations_active: number;
  stations_offline: number;
  availability_bike_pct: number | null;
  availability_dock_pct: number | null;
  penury_pct: number | null;
  saturation_pct: number | null;
  coverage_pct: number | null;
  volatility_today: number | null;
  last_days: number;
  ref_days: number;
};

export type OverviewSnapshotDistributionItem = {
  metric: "bike_avail" | "dock_avail" | "penury" | "saturation";
  count: number;
  total_active: number;
  pct: number; // 0..100
};
export type OverviewSnapshotDistribution = OverviewSnapshotDistributionItem[];

export type OverviewTodayCurve = {
  schema_version: string;
  generated_at: string;
  points: Array<{ hhmm: string; pct: number | null }>;
};

export type OverviewRefMedianCurve = {
  schema_version: string;
  generated_at: string;
  median: Array<{ hhmm: string; pct_median: number | null }>;
};

export type OverviewKpisTodayVsLags = {
  schema_version: string;
  generated_at: string;
  today: { avail_bike: number | null; avail_dock: number | null; pen: number | null; sat: number | null };
  lags: {
    "J-7": OverviewKpisTodayVsLags["today"];
    "J-14": OverviewKpisTodayVsLags["today"];
    "J-21": OverviewKpisTodayVsLags["today"];
  };
};

export type OverviewSnapshotMap = {
  schema_version: string;
  generated_at: string;
  rows: Array<{
    station_id: string;
    name: string;
    lat: number | null;
    lon: number | null;
    bikes: number | null;
    docks_avail: number | null;
    is_penury: 0 | 1;
    is_saturation: 0 | 1;
  }>;
};

export type OverviewStationsTension = {
  schema_version: string;
  generated_at: string;
  rows: Array<{ station_id: string; penury_rate: number | null; saturation_rate: number | null }>;
};

export type StationMeta = {
  station_id: string;
  name?: string | null;
  lat?: number | null;
  lon?: number | null;
};

/* ───────────────────────── Helpers ───────────────────────── */

const path = (suffix: string) => `/monitoring/network/overview${suffix}`;

/** Index minimal des stations (id → meta) pour libellés/carte */
export async function fetchStationsIndex(): Promise<Record<string, StationMeta>> {
  const arr = await getJSON<StationMeta[]>("/stations").catch(() => []);
  const idx: Record<string, StationMeta> = {};
  for (const s of arr) {
    const sid = String((s as any).station_id);
    idx[sid] = {
      station_id: sid,
      name: (s as any).name ?? null,
      lat: Number.isFinite(Number((s as any).lat)) ? Number((s as any).lat) : null,
      lon: Number.isFinite(Number((s as any).lon)) ? Number((s as any).lon) : null,
    };
  }
  return idx;
}

/* ───────────────────────── API calls (ETag) ───────────────────────── */

export const getOverviewKpis = () =>
  fetchJsonWithEtag<OverviewKpis>(path("/kpis"));

export const getOverviewSnapshotDistribution = () =>
  fetchJsonWithEtag<OverviewSnapshotDistribution>(path("/snapshot_distribution"));

export const getOverviewTodayCurve = () =>
  fetchJsonWithEtag<OverviewTodayCurve>(path("/today_curve"));

export const getOverviewRefMedianCurve = () =>
  fetchJsonWithEtag<OverviewRefMedianCurve>(path("/ref_median_curve"));

export const getOverviewKpisTodayVsLags = () =>
  fetchJsonWithEtag<OverviewKpisTodayVsLags>(path("/kpis_today_vs_lags"));

export const getOverviewSnapshotMap = () =>
  fetchJsonWithEtag<OverviewSnapshotMap>(path("/snapshot_map"));

export const getOverviewStationsTension = () =>
  fetchJsonWithEtag<OverviewStationsTension>(path("/stations_tension"));
