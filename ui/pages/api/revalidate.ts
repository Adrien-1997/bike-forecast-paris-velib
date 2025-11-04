// ui/pages/api/revalidate.ts
import type { NextApiRequest, NextApiResponse } from "next";

const SECRET =
  process.env.REVALIDATE_SECRET ||
  process.env.NEXT_REVALIDATION_SECRET ||
  "";

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  res.setHeader("Cache-Control", "no-store, private, max-age=0");

  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ ok: false, error: "Method not allowed" });
  }

  // Secret ONLY via header to avoid appearing in URLs/logs
  const token = (req.headers["x-revalidate-secret"] as string | undefined) || "";
  if (!SECRET || token !== SECRET) {
    // Generic message; never reveal details
    return res.status(401).json({ ok: false, error: "Unauthorized" });
  }

  // Accept either a single path string or an array in JSON body
  // Body example: { "paths": ["/monitoring", "/monitoring/perf"] }
  let body: any = undefined;
  try {
    body = req.body && typeof req.body === "object" ? req.body : JSON.parse(String(req.body || "{}"));
  } catch {
    body = {};
  }

  const requested = Array.isArray(body?.paths)
    ? (body.paths as string[])
    : (typeof body?.path === "string" && body.path) ? [body.path] : [];

  // Default targets if none provided
  const targets = requested.length
    ? new Set(requested)
    : new Set<string>([
        "/monitoring",
        "/monitoring/perf",
        "/monitoring/data-health",
        "/monitoring/drift",
        "/monitoring/network/overview",
        "/monitoring/network/stations",
        "/monitoring/network/dynamics",
        "/monitoring/model-health",
        "/monitoring/pipeline",
        "/monitoring/explain",
        "/docs/dictionary",
        "/docs/methodology",
        "/docs/exports",
      ]);

  const result: Record<string, "revalidated" | "skipped"> = {};
  for (const p of targets) {
    try {
      await res.revalidate(p);
      result[p] = "revalidated";
    } catch {
      // Page may not exist or ISR not enabled yet — ignore
      result[p] = "skipped";
    }
  }

  return res.status(200).json({ ok: true, results: result });
}
