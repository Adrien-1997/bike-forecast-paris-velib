// ui/lib/services/snapshot.ts
import { json } from '@/lib/http';

/**
 * Fetch the live snapshot directly from the API.
 * It contains weather info (temp_C, precip_mm, wind_mps, ts_utc, etc.)
 */
export async function getSnapshot(): Promise<any> {
  try {
    return await json<any>('/snapshot?mode=latest');
  } catch (err) {
    console.warn('[getSnapshot] failed', err);
    return null;
  }
}
