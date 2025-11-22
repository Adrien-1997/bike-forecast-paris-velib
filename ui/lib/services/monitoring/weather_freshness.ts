// ui/lib/services/monitoring/weather_freshness.ts
import { getJSON } from "@/lib/http";

export type WeatherFreshnessDoc = {
  schema_version?: string;
  generated_at?: string;
  meta?: { bin_t_utc?: string } | null;
  provider?: {
    freshness?: {
      p95_min?: number | null;
    } | null;
  } | null;
};

export async function getWeatherFreshnessLatest(): Promise<WeatherFreshnessDoc> {
  return getJSON<WeatherFreshnessDoc>("/monitoring/weather/freshness/latest");
}
