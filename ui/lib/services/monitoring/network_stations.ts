// ui/lib/services/monitoring/network_stations.ts
// Service pour la page /monitoring/network/stations
// → utilise lib/http.ts (token & base déjà gérés)

import { fetchJsonWithEtag, getJSON } from "@/lib/http";

/* ───────────────────────── Types ───────────────────────── */
export type KpisDoc = {
  schema_version: string;
  generated_at: string;
  n_stations: number | null;
  k_effective: number | null;
  silhouette: number | null;
  davies_bouldin: number | null;
  window_days: number;
};

export type CentroidsDoc = {
  schema_version: string;
  generated_at: string;
  x_labels: string[];
  centroids: Array<{ cluster: number; y: (number | null)[] }>;
};

export type Stats7Doc = {
  schema_version: string;
  generated_at: string;
  rows: Array<{
    station_id: string;
    name?: string | null;
    lat?: number | null;
    lon?: number | null;
    capacity_est?: number | null;
    volatility?: number | null;
    penury_rate?: number | null;
    saturation_rate?: number | null;
    coverage_pct?: number | null;
    cluster?: number | null;
  }>;
};

export type StationMeta = {
  station_id: string;
  name?: string | null;
  lat?: number | null;
  lon?: number | null;
  capacity?: number | null;
};

/* ───────────────────────── Helpers ───────────────────────── */

const path = (suffix: string) => `/monitoring/network/stations${suffix}`;

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
      capacity: Number.isFinite(Number((s as any).capacity)) ? Number((s as any).capacity) : null,
    };
  }
  return idx;
}

/* ───────────────────────── API calls (ETag) ───────────────────────── */

export const getStationsKpis = () =>
  fetchJsonWithEtag<KpisDoc>(path("/kpis"));

export const getStationsCentroids = () =>
  fetchJsonWithEtag<CentroidsDoc>(path("/centroids"));

export const getStationsStats7 = () =>
  fetchJsonWithEtag<Stats7Doc>(path("/stats7"));
