// ui/pages/api/revalidate.ts
import type { NextApiRequest, NextApiResponse } from "next";

// Utilise soit REVALIDATE_SECRET soit NEXT_REVALIDATION_SECRET
const SECRET = process.env.REVALIDATE_SECRET || process.env.NEXT_REVALIDATION_SECRET;

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  try {
    // Auth simple via query ?secret=... ou header X-Revalidate-Secret
    const token =
      (req.query.secret as string | undefined) ||
      (req.headers["x-revalidate-secret"] as string | undefined);

    if (!SECRET || token !== SECRET) {
      return res.status(401).json({ ok: false, error: "Unauthorized" });
    }

    // Cibles par défaut à revalider (ajoute/enlève à ta convenance)
    const targets = new Set<string>([
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

    // Permet de cibler une route précise: /api/revalidate?path=/monitoring/perf
    const onePath = (req.query.path as string | undefined)?.trim();
    if (onePath) targets.add(onePath);

    const results: Record<string, "revalidated" | "skipped"> = {};
    for (const p of targets) {
      try {
        await res.revalidate(p);
        results[p] = "revalidated";
      } catch {
        // Next renvoie parfois des erreurs si la page n'existe pas encore : on ignore
        results[p] = "skipped";
      }
    }

    return res.json({ ok: true, results });
  } catch (err: any) {
    return res.status(500).json({ ok: false, error: err?.message || "Server error" });
  }
}
