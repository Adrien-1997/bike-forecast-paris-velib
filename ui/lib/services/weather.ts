// ui/lib/services/weather.ts
import { json } from '@/lib/http';

export type LiveWeather = {
  ts_utc?: string | null;
  temp_C?: number | null;
  precip_mm?: number | null;
  wind_mps?: number | null;
} | null;

export async function getWeather(): Promise<LiveWeather> {
  try {
    return await json<LiveWeather>('/weather/live');
  } catch (e) {
    console.warn('[getWeather] failed', e);
    return null;
  }
}
