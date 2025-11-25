// pages/api/debug/badges-proxy.ts
//
// =============================================================================
// Proxy de DEBUG pour les "badges" depuis Cloud Run (backend Vélib' Forecast)
// -----------------------------------------------------------------------------
// Cette route Next.js est utilisée comme ENDPOINT DE DEBUG côté frontend pour
// aller chercher, via Cloud Run, l’état courant des "badges" (ex. statuts,
// compteurs, indicateurs synthétiques).
//
// Rôles principaux :
//   - appeler l’endpoint /badges du backend Cloud Run en mode "latest" ;
//   - injecter un token d’API en header Authorization (jamais exposé au client) ;
//   - forcer un fetch "frais" (no-store) pour éviter la mise en cache ;
//   - renvoyer au client la réponse JSON du backend, ou un wrapper minimal
//     si le backend ne renvoie pas du JSON propre.
//
// Variables d’environnement utilisées :
//   - CLOUD_RUN_BASE : base URL du service Cloud Run
//       ex. "https://velib-api-xyz-ue.a.run.app"
//       → nettoyée pour retirer les "/" finaux.
//   - API_TOKEN      : token d’API injecté côté Netlify (ou build), jamais transmis
//       au navigateur tel quel. Utilisé uniquement en Authorization Bearer.
//
// Sécurité & confidentialité :
//   - en cas de mauvaise configuration, on renvoie un message générique
//     "Upstream not configured" (sans nom de variable d’env) ;
//   - en cas d’échec de fetch, on renvoie "Upstream fetch failed" sans détails ;
//   - en cas de réponse non-JSON, on renvoie un wrapper textuel tronqué,
//     sans jamais loguer ni exposer l’URL ni les headers exacts.
// =============================================================================

import type { NextApiRequest, NextApiResponse } from "next";

// Base Cloud Run (sans slash final)
const BASE = (process.env.CLOUD_RUN_BASE || "").replace(/\/+$/, "");

// Token d’API uniquement côté serveur (jamais renvoyé au client)
const API_TOKEN = process.env.API_TOKEN || ""; // injecté côté Netlify, jamais exposé

/**
 * Handler API Next.js pour /api/debug/badges-proxy.
 *
 * Usage attendu :
 *   - Route de debug pour récupérer les derniers "badges" calculés côté backend.
 *   - Appelle l’endpoint Cloud Run : `${CLOUD_RUN_BASE}/badges?mode=latest&_ts=<timestamp>`.
 *
 * Comportement :
 *   1. Vérifie la présence de CLOUD_RUN_BASE (BASE).
 *   2. Construit l’URL d’upstream avec un timestamp (_ts) pour contourner la mise en cache.
 *   3. Appelle fetch() en GET avec :
 *        - "accept: application/json"
 *        - header Authorization Bearer si API_TOKEN est défini
 *        - cache: "no-store" pour forcer un appel frais.
 *   4. Lit la réponse en texte :
 *        - essaie de parser en JSON → renvoie tel quel (status + body JSON) ;
 *        - sinon, encapsule dans un wrapper minimal { ok, status, body }.
 *
 * Erreurs :
 *   - Si BASE est vide  → 500 { error: "Upstream not configured" }.
 *   - Si fetch échoue   → 502 { error: "Upstream fetch failed" }.
 *
 * @param req Requête HTTP entrante (non utilisée ici, endpoint en lecture seule).
 * @param res Réponse HTTP Next.js.
 */
export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (!BASE) {
    // Ne divulgue pas le nom de la variable manquante
    return res.status(500).json({ error: "Upstream not configured" });
  }

  // Timestamp dans la query pour éviter toute mise en cache aggressive côté proxy/CDN
  const url = `${BASE}/badges?mode=latest&_ts=${Date.now()}`;

  try {
    const upstream = await fetch(url, {
      method: "GET",
      headers: {
        accept: "application/json",
        ...(API_TOKEN ? { authorization: `Bearer ${API_TOKEN}` } : {}),
      },
      // on force le fetch frais pour un endpoint de debug
      cache: "no-store",
    });

    // On forward le status et le JSON sans écho de l’URL ni des headers
    const text = await upstream.text();
    let body: any;

    try {
      body = JSON.parse(text);
    } catch {
      // Si ce n’est pas du JSON, renvoie un wrapper minimal (toujours sans secrets)
      return res
        .status(upstream.status)
        .json({ ok: upstream.ok, status: upstream.status, body: text.slice(0, 2000) });
    }

    return res.status(upstream.status).json(body);
  } catch (e: any) {
    // Message générique pour éviter toute fuite d’info
    return res.status(502).json({ error: "Upstream fetch failed" });
  }
}