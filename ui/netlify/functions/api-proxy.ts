// netlify/functions/api-proxy.ts
//
// =============================================================================
// Proxy Netlify → Cloud Run (API backend Vélib' Forecast)
// -----------------------------------------------------------------------------
// Cette fonction Netlify sert de proxy HTTP entre le frontend (hébergé sur
// Netlify) et le backend déployé sur Cloud Run.
//
// Rôles principaux :
//   - reconstruire l'URL cible à partir du path Netlify et des variables d'env,
//   - filtrer / normaliser les en-têtes avant de les forwarder,
//   - éviter la double compression (content-encoding / transfer-encoding),
//   - injecter un token Bearer global si nécessaire,
//   - renvoyer au client la réponse Cloud Run en base64 (convention Netlify).
//
// Convention de routage :
//   - Appel côté client :  /.netlify/functions/api-proxy/...path...
//   - Proxy reconstruit :  <CLOUD_RUN_BASE>/<API_PREFIX>/...path...
//
// Variables d'environnement attendues :
//   - CLOUD_RUN_BASE   : base URL du service Cloud Run (ex. "https://api-xyz.run.app")
//                        → sera nettoyée pour retirer un éventuel "/" final.
//   - API_PREFIX       : préfixe d'API optionnel (ex. "api/v1")
//                        → sera nettoyé pour retirer les "/" en début/fin.
//   - API_GLOBAL_TOKEN : token global optionnel, ajouté en "Authorization: Bearer ..."
//
// En cas d'erreur interne (fetch, URL invalide, etc.), la fonction renvoie un
// HTTP 502 avec un JSON { error: "proxy_error", detail }.
// =============================================================================

import type { Handler, HandlerEvent } from "@netlify/functions";

// Base Cloud Run nettoyée (sans slash final)
const BASE = (process.env.CLOUD_RUN_BASE || "").replace(/\/$/, "");

// Préfixe d'API (sans slash au début / à la fin)
const API_PREFIX = (process.env.API_PREFIX || "").replace(/^\/|\/$/g, "");

// Token global éventuel, formaté en header Bearer
const BEARER = process.env.API_GLOBAL_TOKEN ? `Bearer ${process.env.API_GLOBAL_TOKEN}` : "";

/**
 * Construit les en-têtes à forwarder vers Cloud Run.
 *
 * - reprend les en-têtes envoyés par le client (event.headers),
 * - filtre les en-têtes hop-by-hop (host, connection, transfer-encoding),
 * - force "accept: application/json",
 * - remplace "accept-encoding" par "identity" pour éviter la double compression,
 * - injecte un header Authorization Bearer global si le client n'en a pas fourni.
 *
 * @param src  En-têtes d'origine (Netlify / client).
 * @returns    Instance de Headers prête à être utilisée par fetch().
 */
function buildForwardHeaders(src: Record<string, string | undefined> | undefined): Headers {
  const h = new Headers();
  if (!src) return h;

  for (const [k, v] of Object.entries(src)) {
    if (!v) continue;
    const kl = k.toLowerCase();
    // Retire les en-têtes hop-by-hop ou spécifiques au transport
    if (kl === "host" || kl === "connection" || kl === "transfer-encoding") continue;
    h.set(k, v);
  }

  // Toujours demander du JSON côté backend (notre API répond en JSON)
  if (!h.has("accept")) h.set("accept", "application/json");

  // Empêche la double compression : on demande une réponse "brute"
  // Netlify ajoutera lui-même les en-têtes de compression si nécessaire.
  h.set("accept-encoding", "identity");

  // Injecte le token global si aucun header Authorization n'est présent
  if (!h.has("authorization") && BEARER) {
    h.set("authorization", BEARER);
  }

  return h;
}

/**
 * Handler Netlify principal pour le proxy.
 *
 * Étapes :
 *   1. Vérifie la présence de CLOUD_RUN_BASE.
 *   2. Reconstruit le path "passthrough" en retirant le préfixe
 *      "/.netlify/functions/api-proxy".
 *   3. Reconstruit l'URL complète vers Cloud Run :
 *        BASE + "/" + API_PREFIX (optionnel) + passthrough + query string.
 *   4. Adapte la méthode + le body (prise en charge du base64 Netlify).
 *   5. Appelle fetch() vers Cloud Run avec les en-têtes filtrés / normalisés.
 *   6. Reconstruit la réponse au format attendu par Netlify :
 *        - body en base64
 *        - headers nettoyés (pas de transfer-encoding, connection, etc.)
 *
 * En cas d'erreur technique (fetch / DNS / autre), renvoie :
 *   statusCode: 502
 *   body      : { "error": "proxy_error", "detail": "<message>" }
 */
export const handler: Handler = async (event: HandlerEvent) => {
  try {
    // Garde-fou : sans base Cloud Run, on ne peut rien forwarder
    if (!BASE) {
      return { statusCode: 500, body: JSON.stringify({ error: "CLOUD_RUN_BASE not set" }) };
    }

    // Préfixe de la fonction côté Netlify (partie à enlever dans l'URL reçue)
    const fnPrefix = "/.netlify/functions/api-proxy";

    // Path brut reçu par Netlify (ex. "/.netlify/functions/api-proxy/stations")
    const rawPath = event.path || "/";

    // Path "fonctionnel" à passer au backend (sans le préfixe Netlify)
    const passthrough = rawPath.startsWith(fnPrefix) ? rawPath.slice(fnPrefix.length) : rawPath;

    // Query string brute (si présente)
    const qs = event.rawQuery ? `?${event.rawQuery}` : "";

    // URL finale vers Cloud Run :
    //   BASE + "/" + API_PREFIX (optionnel) + passthrough + query string
    const upstreamUrl = `${BASE}${API_PREFIX ? `/${API_PREFIX}` : ""}${passthrough}${qs}`;

    // Méthode HTTP normalisée (GET, POST, etc.)
    const method = (event.httpMethod || "GET").toUpperCase();

    // En-têtes forwardés (filtrés + token global éventuel)
    const headers = buildForwardHeaders(event.headers);

    // Body : seulement pour les méthodes avec payload (pas GET / HEAD)
    // Netlify fournit body potentiellement encodé en base64.
    const body =
      event.body && method !== "GET" && method !== "HEAD"
        ? (event.isBase64Encoded ? Buffer.from(event.body, "base64") : event.body)
        : undefined;

    console.log("[proxy] →", method, upstreamUrl);

    // Appel vers Cloud Run
    const res = await fetch(upstreamUrl, { method, headers, body: body as any, redirect: "manual" });

    // Récupère la réponse sous forme de buffer (pour ensuite encoder en base64)
    const ab = await res.arrayBuffer();
    const buf = Buffer.from(ab);

    // Entêtes propres pour le client (Netlify / navigateur)
    const out: Record<string, string> = {};
    res.headers.forEach((v, k) => {
      const kl = k.toLowerCase();

      // On retire les en-têtes de transport que Netlify gérera lui-même
      if (kl === "transfer-encoding" || kl === "connection" || kl === "content-encoding") return;

      // On laisse Netlify calculer le Content-Length à partir de notre body
      if (kl === "content-length") return;

      out[k] = v;
    });

    // On force un Content-Length cohérent avec le buffer renvoyé
    out["Content-Length"] = String(buf.length);

    // Log des réponses non-OK pour debug (sans tout loguer) :
    if (!res.ok) {
      const peek = buf.toString("utf8").slice(0, 1000);
      console.error(`[proxy] upstream ${res.status} ${res.statusText} — body:`, peek);
    }

    // Réponse finale au format Netlify
    return {
      statusCode: res.status,
      headers: out,
      body: buf.toString("base64"),
      isBase64Encoded: true,
    };
  } catch (e: any) {
    // Erreur technique côté proxy (fetch, DNS, variables d'env, etc.)
    console.error("[proxy] ERROR", e);
    return { statusCode: 502, body: JSON.stringify({ error: "proxy_error", detail: String(e?.message || e) }) };
  }
};
