// pages/api/debug/gauge.ts
import type { NextApiRequest, NextApiResponse } from 'next';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://127.0.0.1:8081';

async function tryJson(path: string) {
  const url = `${API_BASE}${path}${path.includes('?') ? '&' : '?'}_ts=${Date.now()}`;
  try {
    const res = await fetch(url, { cache: 'no-store', headers: { accept: 'application/json' } });
    const text = await res.text();
    let data: any = null;
    try { data = JSON.parse(text); } catch { /* not json */ }
    return { ok: res.ok, status: res.status, path, data, raw: data ?? text };
  } catch (e: any) {
    return { ok: false, status: 0, path, error: String(e) };
  }
}

function toMs(x: any): number | null {
  if (!x) return null;
  const t = Date.parse(String(x));
  return Number.isFinite(t) ? t : null;
}
function diffMin(a: number | null, b: number | null): number | null {
  if (a == null || b == null) return null;
  return Math.floor((a - b) / 60000);
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const probes = await Promise.all([
    // badges: freshness + updated_at
    tryJson('/badges?mode=latest'),
    // candidates pour “serving/parquet latest”
    tryJson('/serving/latest'),              // ex: meta.latest_ts or window_end
    tryJson('/monitoring/data-health'),      // ex: parquet_latest_at / serving_latest_at
    // dernier run features_4h (si exposé)
    tryJson('/jobs/features-4h/last-run'),   // ex: { last_success_at: ... }
  ]);

  const byPath: Record<string, any> = {};
  for (const p of probes) byPath[p.path] = p;

  // extraction tolérante aux schémas
  const badges = byPath['/badges?mode=latest']?.data ?? {};
  const badgesUpdatedAt =
    badges?.meta?.updated_at ?? badges?.updated_at ?? badges?.ts ?? badges?.weather?.updated_at ?? null;
  const badgesUpdatedMs = toMs(badgesUpdatedAt);

  // Serving/parquet latest candidates
  const serving1 = byPath['/serving/latest']?.data ?? {};
  const serving2 = byPath['/monitoring/data-health']?.data ?? {};
  const servingTs =
    serving1?.meta?.window_end ??
    serving1?.window_end ??
    serving2?.serving_latest_at ??
    serving2?.parquet_latest_at ??
    serving2?.latest_ts ??
    null;
  const servingMs = toMs(servingTs);

  // Dernier run features_4h
  const job = byPath['/jobs/features-4h/last-run']?.data ?? {};
  const lastRunAt = job?.last_success_at ?? job?.last_run_at ?? job?.updated_at ?? null;
  const lastRunMs = toMs(lastRunAt);

  const nowMs = Date.now();
  const result = {
    now_iso: new Date(nowMs).toISOString(),
    badges: {
      updated_at: badgesUpdatedAt,
      updated_age_min: badgesUpdatedMs ? Math.floor((nowMs - badgesUpdatedMs) / 60000) : null,
      freshness_min_server: badges?.meta?.freshness_min ?? badges?.freshness?.age_minutes ?? null,
    },
    serving: {
      latest_ts: servingTs,
      latest_age_min: servingMs ? Math.floor((nowMs - servingMs) / 60000) : null,
    },
    features_4h: {
      last_success_at: lastRunAt,
      age_min: lastRunMs ? Math.floor((nowMs - lastRunMs) / 60000) : null,
    },
    // Diff utiles
    deltas: {
      serving_vs_badges_min: diffMin(servingMs, badgesUpdatedMs),
      job_vs_serving_min: diffMin(lastRunMs, servingMs),
      job_vs_badges_min: diffMin(lastRunMs, badgesUpdatedMs),
    },
    // traces brutes pour voir ce qui répond vraiment
    raw: {
      badges: byPath['/badges?mode=latest'],
      serving_latest: byPath['/serving/latest'],
      data_health: byPath['/monitoring/data-health'],
      features_job: byPath['/jobs/features-4h/last-run'],
    },
  };

  res.status(200).json(result);
}
