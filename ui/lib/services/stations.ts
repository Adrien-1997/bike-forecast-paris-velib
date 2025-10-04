// ui/lib/services/stations.ts
import { json } from '@/lib/http';
import type { Station } from '@/lib/types';

export async function getStations(): Promise<Station[]> {
  return json<Station[]>('/stations', { dedupeKey: 'stations' });
}
