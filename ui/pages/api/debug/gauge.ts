// pages/api/debug/gauge.ts
import type { NextApiRequest, NextApiResponse } from "next";

// Private upstream (never NEXT_PUBLIC_*)
const BASE = (process.env.CLOUD_RUN_BASE || "").replace(/\/+$/, "");
const API_TOKEN = process.env.API_TOKEN || "";

async function tryJson(path: string) {
  if (!BASE) return { ok: false, status: 500, path, error: "Upstream not configured" };

  // No URL echo in responses; keep it local only
  const url = `${BASE}${path}${path.includes("?") ? "&" : "?"}_ts=${Date.now()}`;

  try {
    const res = await fetch(url, {
      method: "GET",
      headers: {
        accept: "application/json",
        ...(API_TOKEN ? { authorization: `Bearer ${API_TOKEN}` } : {}),
      },
      cache: "no-store",
    });

    const text = await res.text();
    // Parse JSON if possible; otherwise return a minimal marker
    try {
      const data = JSON.parse(text);
      return { ok: res.ok, status: res.status, path, data };
    } catch {
      return { ok: res.ok, status: res.status, path, data: null };
    }
  } catch (e: any) {
    return { ok: false, status: 0, path, error: "fetch-failed" };
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
  if (req.method !== "GET") return res.status(405).json({ error: "Method not allowed" });
  if (!BASE) return res.status(500).json({ error: "Upstream not configured" });

  const [badgesP, servingP, healthP, jobP] = await Promise.all([
    tryJson("/badges?mode=latest"),
    tryJson("/serving/latest"),
    tryJson("/monitoring/data-health"),
    tryJson("/jobs/features-4h/last-run"),
  ]);

  // Extract with schema tolerance, without exposing raw payloads
  const badges = (badgesP as any)?.data ?? {};
  const badgesUpdatedAt =
    badges?.meta?.updated_at ??
    badges?.updated_at ??
    badges?.ts ??
    badges?.weather?.updated_at ??
    null;

  const serving = (servingP as any)?.data ?? {};
  const health = (healthP as any)?.data ?? {};

  const servingTs =
    serving?.meta?.window_end ??
    serving?.window_end ??
    health?.serving_latest_at ??
    health?.parquet_latest_at ??
    health?.latest_ts ??
    null;

  const job = (jobP as any)?.data ?? {};
  const lastRunAt = job?.last_success_at ?? job?.last_run_at ?? job?.updated_at ?? null;

  const nowMs = Date.now();
  const badgesMs = toMs(badgesUpdatedAt);
  const servingMs = toMs(servingTs);
  const jobMs = toMs(lastRunAt);

  const result = {
    now_iso: new Date(nowMs).toISOString(),
    badges: {
      updated_at: badgesUpdatedAt,
      updated_age_min: badgesMs ? Math.floor((nowMs - badgesMs) / 60000) : null,
      freshness_min_server:
        badges?.meta?.freshness_min ?? badges?.freshness?.age_minutes ?? null,
    },
    serving: {
      latest_ts: servingTs,
      latest_age_min: servingMs ? Math.floor((nowMs - servingMs) / 60000) : null,
    },
    features_4h: {
      last_success_at: lastRunAt,
      age_min: jobMs ? Math.floor((nowMs - jobMs) / 60000) : null,
    },
    deltas: {
      serving_vs_badges_min: diffMin(servingMs, badgesMs),
      job_vs_serving_min: diffMin(jobMs, servingMs),
      job_vs_badges_min: diffMin(jobMs, badgesMs),
    },
  };

  // No raw, no headers, no URLs
  return res.status(200).json(result);
}
