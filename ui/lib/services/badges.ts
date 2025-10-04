// ui/lib/services/badges.ts
import { json } from '../http';

export async function getBadges(): Promise<any> {
  const raw = await json<any>('/badges?mode=latest', { dedupeKey: 'badges' });

  // Timestamps renvoyés par l’API actuelle
  const parquetTs =
    raw?.freshness?.parquet_ts_utc ?? null;        // ex: "2025-10-04T10:55:00Z"
  const weatherTs =
    raw?.weather?.ts_utc ?? null;                  // ex: "2025-10-04T10:55:00Z"

  // Fraîcheur servie par l’API
  const ageMin =
    raw?.freshness?.age_minutes ??                 // ex: 58.4
    raw?.freshness_min ??
    raw?.meta?.freshness_min ??
    null;

  return {
    ...raw,
    meta: {
      ...(raw?.meta || {}),
      // Priorité au timestamp parquet (plus pertinent pour la fraicheur des données)
      updated_at: parquetTs ?? weatherTs ?? null,
      freshness_min: typeof ageMin === 'number' ? ageMin : (ageMin != null ? Number(ageMin) : null),
    },
  };
}
