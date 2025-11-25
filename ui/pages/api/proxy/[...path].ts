// /ui/pages/api/proxy/[...path].ts
//
// =============================================================================
// Proxy générique Next.js → Cloud Run (backend Vélib' Forecast)
// -----------------------------------------------------------------------------
// Cette route d’API Next.js agit comme un PROXY côté serveur entre :
//   - le frontend Next.js (appel vers /api/proxy/...),
//   - le backend déployé sur Cloud Run.
//
// Objectifs principaux :
//   - reconstruire l’URL cible à partir du catch-all [...path] + query string ;
//   - forwarder méthode, en-têtes et corps de la requête telle quelle ;
//   - injecter un jeton privé serveur → backend (jamais exposé au navigateur) ;
//   - gérer un timeout global pour éviter les requêtes pendantes ;
//   - normaliser la réponse (status, headers, body) sans fuite d’infos sensibles.
//
// En cas de problème upstream, la route renvoie des erreurs génériques
// ("Upstream not configured", "Invalid request body", "Bad gateway") sans jamais
// exposer les détails de configuration (URL, secrets, stack trace, etc.).
// =============================================================================

import type { NextApiRequest, NextApiResponse } from "next";

/*─────────────────────────────── Config privée (serveur) ───────────────────────────────*/
// Base URL du backend Cloud Run (nettoyée pour supprimer les "/" finaux)
const BASE = (process.env.CLOUD_RUN_BASE || "").replace(/\/+$/, "");

// Token d’API privé, injecté côté serveur uniquement (jamais renvoyé au client)
const API_TOKEN = process.env.API_TOKEN || "";

// Nom du header d’authentification utilisé vers le backend (ex: Authorization)
const AUTH_HEADER = process.env.API_AUTH_HEADER || "Authorization";

// Préfixe injecté devant le token (ex: "Bearer ")
const TOKEN_PREFIX = process.env.API_TOKEN_PREFIX ?? "Bearer ";

// Timeout global (en millisecondes) pour les appels vers le backend
const DEFAULT_TIMEOUT_MS = Number(process.env.API_PROXY_TIMEOUT_MS ?? "30000");

// Désactive le bodyParser (on forward le corps tel quel)
export const config = {
  api: {
    bodyParser: false,
    externalResolver: true,
  },
};

/*──────────────────────────── Helpers ───────────────────────────────*/

/**
 * Lit le corps brut d’une requête Next.js et le renvoie sous forme de Buffer.
 *
 * - Utilisé pour les méthodes avec payload (POST, PUT, PATCH, DELETE, etc.).
 * - Pour GET/HEAD, le handler ne fait pas appel à cette fonction.
 *
 * @param req Requête HTTP Next.js entrante.
 * @returns   Promise résolue avec le Buffer contenant le corps complet.
 */
function readRawBody(req: NextApiRequest): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (c) => chunks.push(Buffer.isBuffer(c) ? c : Buffer.from(c)));
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

/**
 * Construit l’URL complète vers le backend à partir de la requête Next.js.
 *
 * Règles :
 *   - path est issu du paramètre catch-all [...path] → "seg1/seg2/..." ;
 *   - la query string est reconstruite à partir de req.url (partie après "?") ;
 *   - le path est préfixé par BASE (CLOUD_RUN_BASE).
 *
 * Exemple :
 *   /api/proxy/stations?foo=bar
 *     → ${BASE}/stations?foo=bar
 *
 * @param req Requête HTTP entrante.
 * @returns   URL absolue vers l’endpoint backend.
 */
function buildUpstreamUrl(req: NextApiRequest) {
  const segs = Array.isArray(req.query.path)
    ? req.query.path
    : [req.query.path].filter(Boolean) as string[];
  const path = segs.join("/");
  const q = req.url?.split("?")[1];
  return `${BASE}/${path}${q ? `?${q}` : ""}`;
}

/**
 * Filtre et reconstruit les en-têtes à forwarder vers le backend.
 *
 * Politique :
 *   - ne propage pas l’Authorization d’entrée (évite l’injection de token) ;
 *   - forwarde uniquement les en-têtes utiles pour le backend :
 *       • Content-Type
 *       • Accept (valeur par défaut: application/json)
 *       • If-None-Match / If-Modified-Since (caching conditionnel) ;
 *   - injecte le jeton privé API_TOKEN dans AUTH_HEADER, avec TOKEN_PREFIX ;
 *   - force "Accept-Encoding: identity" pour éviter la double compression.
 *
 * @param req Requête HTTP entrante.
 * @returns   Dictionnaire d’en-têtes normalisés pour l’appel upstream.
 */
function filterIncomingHeaders(req: NextApiRequest): Record<string, string> {
  const h: Record<string, string> = {};

  // Types acceptés
  const ct = req.headers["content-type"];
  if (ct) h["Content-Type"] = String(ct);
  h["Accept"] = req.headers["accept"] ? String(req.headers["accept"]) : "application/json";

  // Caching conditionnel safe
  if (req.headers["if-none-match"]) h["If-None-Match"] = String(req.headers["if-none-match"]);
  if (req.headers["if-modified-since"])
    h["If-Modified-Since"] = String(req.headers["if-modified-since"]);

  // Pas d'Authorization entrant (évite l'injection)
  // Ajoute le jeton privé serveur si dispo
  if (API_TOKEN) {
    if (AUTH_HEADER.toLowerCase() === "authorization") {
      h["Authorization"] = `${TOKEN_PREFIX}${API_TOKEN}`;
    } else {
      h[AUTH_HEADER] = `${TOKEN_PREFIX}${API_TOKEN}`;
    }
  }

  // Pas de compression côté upstream → on laisse Next appliquer la sienne
  h["Accept-Encoding"] = "identity";

  return h;
}

/**
 * Enveloppe une Promise avec un timeout.
 *
 * Si la Promise ne se résout pas dans le délai imparti, la Promise retournée
 * est rejetée avec une erreur "Upstream timeout".
 *
 * @param p  Promise d’origine (fetch vers le backend, par exemple).
 * @param ms Délai maximal en millisecondes.
 * @returns  Promise qui se comporte comme p, mais avec timeout.
 */
function withTimeout<T>(p: Promise<T>, ms: number): Promise<T> {
  return new Promise((resolve, reject) => {
    const id = setTimeout(() => reject(new Error("Upstream timeout")), ms);
    p.then((v) => { clearTimeout(id); resolve(v); })
     .catch((e) => { clearTimeout(id); reject(e); });
  });
}

/*─────────────────────────────── Handler principal ───────────────────────────────*/

/**
 * Handler principal pour /api/proxy/[...path].
 *
 * Pipeline :
 *   1. Vérifie la configuration (BASE doit être définie).
 *   2. Normalise la méthode HTTP (GET, POST, etc.).
 *   3. Construit l’URL cible via buildUpstreamUrl().
 *   4. Lit le corps brut si nécessaire (méthodes ≠ GET/HEAD).
 *   5. Construit les en-têtes upstream via filterIncomingHeaders().
 *   6. Appelle fetch() vers le backend avec un timeout global.
 *   7. Forwarde au client :
 *        - le status HTTP,
 *        - les en-têtes filtrés (sans transfer-encoding / content-encoding / content-length),
 *        - le corps binaire (Buffer) avec un Content-Length cohérent.
 *
 * Gestion des erreurs :
 *   - Config manquante        → 500 { error: "Upstream not configured" }
 *   - Corps invalide          → 400 { error: "Invalid request body" }
 *   - Erreur réseau / timeout → 502 { error: "Bad gateway" }
 *
 * Aucun détail de l’URL backend ni des secrets n’est renvoyé dans les réponses.
 */
export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  // Config manquante → réponse générique (ne pas exposer les valeurs)
  if (!BASE) {
    res.status(500).json({ error: "Upstream not configured" });
    return;
  }

  const method = (req.method || "GET").toUpperCase();
  const url = buildUpstreamUrl(req);

  let body: Buffer | undefined;
  if (!["GET", "HEAD"].includes(method)) {
    try {
      body = await readRawBody(req);
    } catch {
      res.status(400).json({ error: "Invalid request body" });
      return;
    }
  }

  const headers = filterIncomingHeaders(req);
  const bodyInit = body ? new Uint8Array(body) : undefined;

  try {
    const upstream = await withTimeout(
      fetch(url, { method, headers, body: bodyInit, redirect: "manual" }),
      DEFAULT_TIMEOUT_MS
    );

    res.status(upstream.status);

    upstream.headers.forEach((value, key) => {
      const k = key.toLowerCase();
      if (k === "transfer-encoding" || k === "connection") return;
      if (k === "content-encoding") return;
      if (k === "content-length") return;
      res.setHeader(key, value);
    });

    // Pas de mise en cache par défaut côté client pour cet endpoint proxy
    res.setHeader("Cache-Control", "no-store, private, max-age=0");

    if (method === "HEAD" || upstream.status === 304) {
      res.end();
      return;
    }

    const buf = Buffer.from(await upstream.arrayBuffer());
    res.setHeader("Content-Length", String(buf.length));
    res.send(buf);
  } catch {
    // Réponse minimale, sans echo d’URL ni de détails serveur
    res.status(502).json({ error: "Bad gateway" });
  }
}