import { json } from '@/lib/http';
import type { Forecast } from '@/lib/types';

// Split a large array of station codes into chunks (max 600 each)
const chunk = <T,>(arr: T[], n = 600) =>
  Array.from({ length: Math.ceil(arr.length / n) }, (_, i) => arr.slice(i * n, i * n + n));

export async function getForecastBatch(stationcodes: string[], h = 15): Promise<Forecast[]> {
  const codes = Array.from(new Set(stationcodes.map(String)));
  if (!codes.length) return [];

  const parts = chunk(codes, 600);
  const out: Forecast[] = [];

  for (const part of parts) {
    try {
      const res = await json<Forecast[]>('/forecast/batch', {
        method: 'POST',
        body: JSON.stringify({ stationcodes: part, h }),
      });
      if (Array.isArray(res)) out.push(...res);
    } catch (e: any) {
      // fallback GET permissif (si ton API l’accepte)
      try {
        const res = await json<Forecast[]>(`/forecast?h=${h}`);
        if (Array.isArray(res)) out.push(...res);
      } catch {
        throw e;
      }
    }
  }

  return out;
}
