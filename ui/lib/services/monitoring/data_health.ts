// ui/lib/services/monitoring/data_health.ts
// Service pour la page /monitoring/data/health
// → utilise lib/http.ts (token & base déjà gérés)

import { fetchJsonWithEtag } from "@/lib/http";

/* ───────────────────────── Types ───────────────────────── */
export type DataKpis = {
  schema_version: string;
  generated_at: string;
  rows: number;
  stations: number;
  span: [string | null, string | null];
  bin_min: number;
  current_days: number;
  tz: string;
  now_utc?: string | null;
  ts_global_max?: string | null;
  freshness_age_p50_min?: number | null;
  freshness_age_p95_min?: number | null;
  freshness_slo_min?: number | null;
  freshness_p95_ok?: boolean | null;
  coverage_global_pct?: number | null; // 0..100
  latency_p50_min?: number | null;
  latency_p95_min?: number | null;
  dups_pct?: number | null; // 0..100
  missing_stations?: number | null;
  thresholds?: {
    fresh_slo_min?: number;
    compl_alert_pct?: number;
    dup_alert_pct?: number;
    flat_steps?: number;
  };
  alerts?: {
    freshness_p95_ok?: boolean | null;
    coverage_ok?: boolean;
    duplication_alert?: boolean;
    flat_sequences_found?: boolean;
  };
};

export type StationHealthRow = {
  station_id: string;
  name?: string;
  obs?: number;
  expected?: number;
  coverage_pct?: number; // 0..100
  missing?: number;
};

export type CoverageByHourRow = {
  hour: number; // 0..23 (local)
  coverage_pct: number; // 0..100
};

export type Anomaly =
  | {
      type: "flat_sequence";
      station_id: string;
      name?: string;
      start: string;
      end: string;
      steps: number;
      duration_min: number;
    }
  | {
      type: "duplicates";
      station_id: string;
      name?: string;
      dups: number;
    }
  | {
      type: "missing_bins";
      station_id: string;
      name?: string;
      missing: number;
      expected: number;
    };

export type AlertsDoc = {
  freshness_p95_ok?: boolean | null;
  coverage_ok?: boolean;
  duplication_alert?: boolean;
  flat_sequences_found?: boolean;
};

/* ───────────────────────── Helpers ───────────────────────── */

const path = (suffix: string) => `/monitoring/data/health${suffix}`;

/* Certains endpoints peuvent renvoyer un objet enveloppé (ex: {kpis: {...}}) */
function unwrap<T = unknown>(obj: any, key: string): T {
  return (obj?.[key] ?? obj) as T;
}

/* ───────────────────────── API calls (ETag) ───────────────────────── */

export async function getDataHealthKpis(): Promise<DataKpis> {
  const raw = await fetchJsonWithEtag<any>(path("/kpis"));
  return unwrap<DataKpis>(raw, "kpis");
}

export async function getDataHealthStationHealth(): Promise<StationHealthRow[]> {
  const raw = await fetchJsonWithEtag<any>(path("/station_health"));
  return unwrap<StationHealthRow[]>(raw, "station_health");
}

export async function getDataHealthCoverageByHour(): Promise<CoverageByHourRow[]> {
  const raw = await fetchJsonWithEtag<any>(path("/coverage_by_hour"));
  return unwrap<CoverageByHourRow[]>(raw, "coverage_by_hour");
}

export async function getDataHealthAnomalies(): Promise<Anomaly[]> {
  const raw = await fetchJsonWithEtag<any>(path("/anomalies"));
  return unwrap<Anomaly[]>(raw, "anomalies");
}

export async function getDataHealthAlerts(): Promise<AlertsDoc> {
  const raw = await fetchJsonWithEtag<any>(path("/alerts"));
  return unwrap<AlertsDoc>(raw, "alerts");
}
