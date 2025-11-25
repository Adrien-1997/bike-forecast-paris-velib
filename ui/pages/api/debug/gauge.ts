// pages/api/debug/gauge.ts
//
// =============================================================================
// Endpoint de DEBUG pour alimenter une "gauge" de fraîcheur / latence globale
// -----------------------------------------------------------------------------
// Cette route Next.js agrège plusieurs signaux backend pour produire un petit
// résumé homogène, destiné à une jauge de monitoring "temps réel" côté UI.
//
// Rôles principaux :
//   - interroger plusieurs endpoints internes du backend Cloud Run :
//       • /badges?mode=latest
//       • /serving/latest
//       • /monitoring/data-health
//       • /jobs/features-4h/last-run
//   - tolérer des schémas de réponse légèrement différents (meta, root, etc.) ;
//   - convertir les timestamps en âges (minutes) par rapport à "maintenant" ;
//   - calculer quelques deltas (écarts en minutes entre badges / serving / job) ;
//   - NE JAMAIS exposer les payloads bruts, ni les URLs, ni les headers d’upstream.
//
// Variables d’environnement (privées, côté serveur uniquement) :
//   - CLOUD_RUN_BASE : base URL du service Cloud Run
//       ex. "https://velib-api-xyz-ue.a.run.app"
//       → nettoyée pour retirer les "/" finaux.
//   - API_TOKEN      : token d’API (Bearer) utilisé pour sécuriser les appels
//       vers le backend ; n’est jamais renvoyé au client.
//
// Sécurité & confidentialité :
//   - en cas de mauvaise config upstream, on renvoie des erreurs génériques ;
//   - aucun echo d’URL dans la réponse JSON, ni d’headers upstream ;
//   - les données brutes sont réduites à quelques champs agrégés (timestamps,
//     âges en minutes, deltas) adaptés à un widget de monitoring simple.
// =============================================================================

import type { NextApiRequest, NextApiResponse } from "next";

// Upstream privé (jamais exposé en NEXT_PUBLIC_*)
const BASE = (process.env.CLOUD_RUN_BASE || "").replace(/\/+$/, "");
const API_TOKEN = process.env.API_TOKEN || "";

/**
 * Appelle un endpoint JSON côté backend en GET et renvoie un petit wrapper.
 *
 * Comportement :
 *   - renvoie toujours un objet du type :
 *       { ok, status, path, data | error }
 *   - n’expose jamais l’URL complète dans la réponse (uniquement le path) ;
 *   - ajoute un paramètre `_ts` pour contourner d’éventuels caches agressifs ;
 *   - tente de parser la réponse en JSON ; si ce n’est pas du JSON valide,
 *     renvoie simplement `data: null` (sans remonter le body).
 *
 * Erreurs :
 *   - si BASE est absent          → { ok: false, status: 500, error: "Upstream not configured" }
 *   - si fetch échoue (réseau...) → { ok: false, status: 0,   error: "fetch-failed" }
 *
 * @param path Chemin relatif de l’endpoint backend (ex. "/badges?mode=latest").
 */
async function tryJson(path: string) {
  if (!BASE) return { ok: false, status: 500, path, error: "Upstream not configured" };

  // On ne renvoie jamais l’URL complète au client : elle reste locale au serveur
  const url = `${BASE}${path}${path.includes("?") ? "&" : "?"}_ts=${Date.now()}`;

  try {
    const res = await fetch(url, {
      method: "GET",
      headers: {
        accept: "application/json",
        ...(API_TOKEN ? { authorization: `Bearer ${API_TOKEN}` } : {}),
      },
      // On force un fetch "frais" (endpoint de debug / monitoring)
      cache: "no-store",
    });

    const text = await res.text();
    // On essaie de parser en JSON ; si ça échoue, on renvoie data: null
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

/**
 * Convertit un timestamp quelconque en millisecondes depuis l’époque.
 *
 * Accepté :
 *   - string ou valeur convertible en string parsable par Date.parse().
 *
 * @param x Valeur représentant un instant (timestamp, date ISO, etc.).
 * @returns Nombre de millisecondes depuis l’époque, ou null si invalide.
 */
function toMs(x: any): number | null {
  if (!x) return null;
  const t = Date.parse(String(x));
  return Number.isFinite(t) ? t : null;
}

/**
 * Calcule un écart en minutes entre deux instants (a - b).
 *
 * @param a Instant de référence en millisecondes (ou null).
 * @param b Instant de comparaison en millisecondes (ou null).
 * @returns Écart entier en minutes, ou null si l’un des deux est null.
 */
function diffMin(a: number | null, b: number | null): number | null {
  if (a == null || b == null) return null;
  return Math.floor((a - b) / 60000);
}

/**
 * Handler API Next.js pour /api/debug/gauge.
 *
 * Objectif :
 *   - Agréger les signaux de plusieurs endpoints backend (badges, serving,
 *     health data, job features-4h) pour produire un JSON compact :
 *
 *     {
 *       now_iso: "...",
 *       badges: {
 *         updated_at,
 *         updated_age_min,
 *         freshness_min_server
 *       },
 *       serving: {
 *         latest_ts,
 *         latest_age_min
 *       },
 *       features_4h: {
 *         last_success_at,
 *         age_min
 *       },
 *       deltas: {
 *         serving_vs_badges_min,
 *         job_vs_serving_min,
 *         job_vs_badges_min
 *       }
 *     }
 *
 * Points de vigilance :
 *   - méthode limitée à GET (405 sinon) ;
 *   - si CLOUD_RUN_BASE est absent → 500 avec message générique ;
 *   - aucune donnée brute d’upstream n’est renvoyée (pas de payload complet,
 *     pas d’URL, pas de headers).
 */
export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== "GET") return res.status(405).json({ error: "Method not allowed" });
  if (!BASE) return res.status(500).json({ error: "Upstream not configured" });

  const [badgesP, servingP, healthP, jobP] = await Promise.all([
    tryJson("/badges?mode=latest"),
    tryJson("/serving/latest"),
    tryJson("/monitoring/data-health"),
    tryJson("/jobs/features-4h/last-run"),
  ]);

  // Extraction tolérante aux variations de schéma, sans exposer les payloads complets
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
    // Timestamp de référence (côté serveur) pour tous les calculs d’âge
    now_iso: new Date(nowMs).toISOString(),
    badges: {
      updated_at: badgesUpdatedAt,
      // Âge du dernier refresh "badges" en minutes
      updated_age_min: badgesMs ? Math.floor((nowMs - badgesMs) / 60000) : null,
      // Indicateur de fraîcheur côté serveur, si fourni par le backend
      freshness_min_server:
        badges?.meta?.freshness_min ?? badges?.freshness?.age_minutes ?? null,
    },
    serving: {
      latest_ts: servingTs,
      // Âge du dernier point "serving" (prédictions / exports) en minutes
      latest_age_min: servingMs ? Math.floor((nowMs - servingMs) / 60000) : null,
    },
    features_4h: {
      last_success_at: lastRunAt,
      // Âge de la dernière exécution du job features-4h en minutes
      age_min: jobMs ? Math.floor((nowMs - jobMs) / 60000) : null,
    },
    deltas: {
      // Écart entre serving et badges (serving - badges) en minutes
      serving_vs_badges_min: diffMin(servingMs, badgesMs),
      // Écart entre job features-4h et serving
      job_vs_serving_min: diffMin(jobMs, servingMs),
      // Écart entre job features-4h et badges
      job_vs_badges_min: diffMin(jobMs, badgesMs),
    },
  };

  // Pas de payloads bruts, pas de headers ni d’URLs : on renvoie uniquement ce résumé
  return res.status(200).json(result);
}