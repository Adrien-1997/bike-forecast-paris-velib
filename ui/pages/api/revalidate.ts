// ui/pages/api/revalidate.ts
//
// =============================================================================
// Endpoint Next.js pour la revalidation on-demand (ISR)
// -----------------------------------------------------------------------------
// Cette route API permet de déclencher côté serveur la revalidation de pages
// statiques générées via getStaticProps (Incremental Static Regeneration).
//
// Rôles principaux :
//   - sécuriser l’accès à la revalidation via un secret partagé (header HTTP) ;
//   - accepter un corps JSON décrivant les chemins à revalider (path ou paths) ;
//   - définir, en l’absence de payload, une liste de chemins par défaut à
//     revalider (pages de monitoring / docs) ;
//   - appeler res.revalidate() pour chaque chemin cible, en tolérant les
//     erreurs individuelles (chemins non ISR, pages manquantes, etc.) ;
//   - renvoyer un récapitulatif JSON des chemins revalidés / ignorés.
//
// Sécurité :
//   - le secret n’est accepté que via un header dédié x-revalidate-secret
//     (jamais via la query string) pour éviter les fuites dans les logs/URLs ;
//   - le message d’erreur reste générique en cas de secret invalide ou manquant.
// =============================================================================

import type { NextApiRequest, NextApiResponse } from "next";

// Secret partagé pour authentifier les appels à /api/revalidate.
// - On supporte deux noms de variables d’environnement possibles :
//     • REVALIDATE_SECRET
//     • NEXT_REVALIDATION_SECRET
// - Si aucune n’est définie, SECRET = "" et toute tentative échouera (401),
//   ce qui évite d’ouvrir la revalidation par inadvertance.
const SECRET =
  process.env.REVALIDATE_SECRET ||
  process.env.NEXT_REVALIDATION_SECRET ||
  "";

/**
 * Handler API Next.js pour /api/revalidate.
 *
 * Pipeline :
 *   1. Désactive toute mise en cache HTTP pour s’assurer que chaque appel
 *      touche réellement le serveur (no-store).
 *   2. N’accepte que la méthode POST (sinon 405 + Allow: POST).
 *   3. Lit le secret d’authentification dans le header x-revalidate-secret
 *      et le compare à SECRET (401 en cas d’échec).
 *   4. Tente de parser le body (JSON) pour déterminer une liste explicite
 *      de chemins à revalider (path ou paths).
 *   5. Si aucun chemin n’est fourni, utilise une liste par défaut orientée
 *      monitoring et documentation.
 *   6. Pour chaque chemin, appelle res.revalidate(p) et enregistre le statut
 *      "revalidated" ou "skipped" dans un objet résultat.
 *   7. Retourne un JSON { ok: true, results }.
 *
 * Notes :
 *   - Les erreurs de revalidation sur un chemin donné n’empêchent pas le
 *     traitement des autres chemins (best effort).
 *   - Aucun détail sur le secret ou la configuration n’est exposé au client.
 */
export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  // Empêche l’utilisation d’un cache HTTP (client ou proxy) :
  // chaque appel de revalidation doit être traité côté serveur.
  res.setHeader("Cache-Control", "no-store, private, max-age=0");

  // Seule la méthode POST est autorisée : la revalidation est une action
  // "mutative" sur le cache ISR, on la limite donc explicitement.
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
    // Si Next a déjà parsé le body (req.body objet), on le garde tel quel.
    // Sinon, on tente un parse JSON à partir du corps brut.
    body = req.body && typeof req.body === "object" ? req.body : JSON.parse(String(req.body || "{}"));
  } catch {
    // En cas d’échec de parsing, on retombe sur un objet vide.
    body = {};
  }

  // Normalisation des chemins demandés :
  //   - prioritaire : body.paths si c’est un tableau de chaînes ;
  //   - fallback   : body.path si c’est une chaîne unique.
  const requested = Array.isArray(body?.paths)
    ? (body.paths as string[])
    : (typeof body?.path === "string" && body.path) ? [body.path] : [];

  // Default targets if none provided
  // ---------------------------------------------------------------------------
  // Si aucun chemin n’est explicitement demandé, on revalide un set de pages
  // "noyau dur" : monitoring et documentation. On utilise un Set pour éviter
  // les doublons potentiels.
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

  // Pour chaque chemin, on tente la revalidation ISR. En cas d’exception,
  // on marque simplement le chemin comme "skipped" sans faire échouer la
  // requête entière (best effort).
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

  // Réponse finale : statut global + carte { chemin → statut }.
  return res.status(200).json({ ok: true, results: result });
}
