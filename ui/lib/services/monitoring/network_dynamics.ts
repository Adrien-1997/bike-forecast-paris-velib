// ui/lib/services/monitoring/network_dynamics.ts
// Service pour la page /monitoring/network/dynamics
// → utilise lib/http.ts (token & base déjà gérés)

import { fetchJsonWithEtag, getJSON } from "@/lib/http";

/* ───────────────────────── Types ───────────────────────── */
export type HeatmapsProfilesDoc = {
  schema_version: string;
  generated_at: string;
  heatmap: {
    occ_mean: (number | null)[][];
    penury_rate: (number | null)[][];
    saturation_rate: (number | null)[][];
  };
  profiles_occ_by_dow: Record<string, (number | null)[]>;
};

export type HourlyDoc = {
  schema_version: string;
  generated_at: string;
  rows: Array<{ hour: number; penury_rate: number | null; saturation_rate: number | null }>;
};

export type EpisodesDoc = {
  schema_version: string;
  generated_at: string;
  last_days: number;
  rows: Array<{
    station_id: string;
    type: "penury" | "saturation";
    start_utc: string;
    end_utc: string;
    steps: number;
    duration_min: number | null;
  }>;
};

export type TensionByStationDoc = {
  schema_version: string;
  generated_at: string;
  last_days: number;
  rows: Array<{
    station_id: string;
    name?: string | null;
    lat?: number | null;
    lon?: number | null;
    penury_rate: number | null;
    saturation_rate: number | null;
    occ_mean: number | null;
    tension_index: number | null; // 0..2
    n_obs: number;
  }>;
};

export type StationMeta = {
  station_id: string;
  name?: string | null;
  lat?: number | null;
  lon?: number | null;
};

/* ───────────────────────── Helpers ───────────────────────── */

const path = (suffix: string) => `/monitoring/network/dynamics${suffix}`;

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

/* Certains endpoints peuvent renvoyer un objet enveloppé (ex: {heatmaps_profiles: {...}}) */
function unwrapHeatmapsProfiles(p: any): HeatmapsProfilesDoc {
  return (p?.heatmaps_profiles ?? p) as HeatmapsProfilesDoc;
}
function unwrapHourly(p: any): HourlyDoc {
  return (p?.hourly ?? p) as HourlyDoc;
}
function unwrapEpisodes(p: any): EpisodesDoc {
  return (p?.episodes ?? p) as EpisodesDoc;
}
function unwrapTension(p: any): TensionByStationDoc {
  return (p?.tension_by_station ?? p) as TensionByStationDoc;
}

/* ───────────────────────── API calls (ETag) ───────────────────────── */

export const getDynamicsHeatmapsProfiles = async () => {
  const raw = await fetchJsonWithEtag<any>(path("/heatmaps_profiles"));
  return unwrapHeatmapsProfiles(raw);
};

export const getDynamicsHourlyPenSat = async () => {
  const raw = await fetchJsonWithEtag<any>(path("/hourly_pen_sat"));
  return unwrapHourly(raw);
};

export const getDynamicsEpisodes = async () => {
  const raw = await fetchJsonWithEtag<any>(path("/episodes"));
  return unwrapEpisodes(raw);
};

export const getDynamicsTensionByStation = async () => {
  const raw = await fetchJsonWithEtag<any>(path("/tension_by_station"));
  return unwrapTension(raw);
};
