// ui/lib/types.ts
// Types EXACTEMENT alignés à l'API backend actuelle

export type Station = {
  stationcode: string;              // toujours string côté API
  name?: string;
  lat?: number;                     // peut manquer selon le parquet
  lon?: number;                     // peut manquer selon le parquet
  capacity?: number;
  num_bikes_available?: number;
  num_docks_available?: number;
};

export type Forecast = {
  stationcode: string;
  bikes_pred_t15: number;           // champ officiel de l’API
  capacity: number;
  ts_utc: string | null;
};

export type Badges = {
  // nouveau format
  weather?: { temp_C?: number; precip_mm?: number; wind_mps?: number } | null;
  freshness?: { parquet_ts_utc?: string; age_minutes?: number } | null;
  // ancien format (plat)
  temp_C?: number | null;
  precip_mm?: number | null;
  wind_mps?: number | null;
  parquet_age_min?: number | null;
};
