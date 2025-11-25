// ui/lib/http.ts
//
// =============================================================================
// Utilitaires HTTP centralisés pour le frontend Vélib’ Forecast.
//
// Rôle :
// - Centraliser toute la logique réseau : base URL, timeout, retries, parsing JSON,
//   gestion des erreurs et du cache ETag (localStorage).
// - Fournir une API minimale (`json`, `getJSON`, `postJSON`) consommée par tous
//   les services (`/forecast`, `/monitoring/...`, etc.).
// - Offrir quelques helpers spécifiques aux prévisions (`selectForecastRows`,
//   `getForecastRows`).
//
// Contraintes :
// - Aucunes dépendances à React ou au DOM : module utilitaire pur.
// - Aucun Cloud Run URL en dur : tout passe par les variables d’environnement
//   ou le proxy Netlify (`/.netlify/functions/api-proxy`).
// - Toute évolution d’API backend doit idéalement se refléter via ce module
//   pour garder un point de passage unique.
// =============================================================================

/*─────────────────────────────── Base URL (proxy Netlify) ───────────────────────────────*/

/**
 * Concatène proprement une base URL et un chemin, en évitant les doubles slashs.
 */
function joinUrl(base: string, path: string) {
  const b = base.replace(/\/+$/, "");
  const p = path.replace(/^\/+/, "");
  return `${b}/${p}`;
}

/**
 * Base URL utilisée pour tous les appels API.
 *
 * Priorité :
 *  1. `NEXT_PUBLIC_API_BASE` (env public Next.js).
 *  2. Proxy Netlify (`/.netlify/functions/api-proxy`) pour dev / prod.
 *
 * On ne met jamais d’URL Cloud Run en dur ici pour éviter :
 * - les soucis de scan de secrets,
 * - les divergences entre environnements.
 */
export const API_BASE: string = (
  process.env.NEXT_PUBLIC_API_BASE && process.env.NEXT_PUBLIC_API_BASE.trim()
) || "/.netlify/functions/api-proxy";

// Jeton public optionnel (local/dev uniquement, jamais sensible)
const PUBLIC_TOKEN = process.env.NEXT_PUBLIC_API_TOKEN ?? "";

// Timeout HTTP par défaut (en millisecondes)
export const DEFAULT_TIMEOUT_MS = Number(
  process.env.NEXT_PUBLIC_HTTP_TIMEOUT_MS ?? "30000"
); // 30s

if (typeof window !== "undefined") {
  // Petit log debug inoffensif côté client
  console.log("[http] API_BASE =", API_BASE);
  console.log("[http] timeout =", DEFAULT_TIMEOUT_MS, "ms");
}

/*─────────────────────────────── Types & Helpers ───────────────────────────────*/

/**
 * Extension de `RequestInit` avec :
 * - `timeoutMs` : timeout spécifique (ms),
 * - `noCache`   : force `cache: "no-store"` si true.
 */
type JsonInit = RequestInit & {
  timeoutMs?: number;
  noCache?: boolean;
};

/**
 * Erreur HTTP spécialisée, utilisée pour remonter un statut + body éventuel.
 */
export class HttpError extends Error {
  status: number;
  body?: string;
  constructor(status: number, message: string, body?: string) {
    super(`${status} ${message}`);
    this.name = "HttpError";
    this.status = status;
    this.body = body;
  }
}

/**
 * Crée un AbortController avec timeout.
 *
 * - `ms` : durée avant abort (DEFAULT_TIMEOUT_MS par défaut).
 * - Retourne `{ signal, done }` où `done()` nettoie le timer.
 */
function withTimeout(ms = DEFAULT_TIMEOUT_MS) {
  const ctrl = new AbortController();
  const id = setTimeout(() => ctrl.abort(), ms);
  return { signal: ctrl.signal, done: () => clearTimeout(id) };
}

/**
 * Résout un chemin relatif en URL absolue (via API_BASE),
 * ou laisse passer une URL absolue (`http(s)://`).
 */
function resolveUrl(pathOrUrl: string) {
  const hasProtocol = /^https?:\/\//i.test(pathOrUrl);
  const url = hasProtocol ? pathOrUrl : joinUrl(API_BASE, pathOrUrl);
  return { url };
}

/**
 * Ajoute un timestamp `_ts=...` à l’URL pour contourner d’éventuels caches
 * agressifs (proxies, CDN) lorsque `noCache` est souhaité.
 */
function addTs(url: string) {
  const ts = `_ts=${Date.now()}`;
  return `${url}${url.includes("?") ? "&" : "?"}${ts}`;
}

/**
 * Petit wrapper fetch avec retry en cas de timeout/abort.
 *
 * - `tries` : nombre de tentatives max (par défaut 2).
 * - Retries uniquement sur abort/timeout, pas sur erreurs HTTP 4xx/5xx.
 */
async function fetchWithRetry(input: RequestInfo, init: RequestInit, tries = 2) {
  try {
    return await fetch(input, init);
  } catch (e: any) {
    if (tries > 1 && (e?.name === "AbortError" || /aborted|timeout/i.test(String(e)))) {
      console.warn("[http] retry after abort/timeout →", input);
      return fetchWithRetry(input, init, tries - 1);
    }
    throw e;
  }
}

/*─────────────────────────────── Core JSON fetch ───────────────────────────────*/

/**
 * Fetch JSON "standard" avec :
 * - timeout configurable,
 * - ajout d’un timestamp `_ts` (anti-cache),
 * - gestion des erreurs HTTP via `HttpError`,
 * - parsing JSON défensif.
 *
 * Paramètres :
 * - `path` : chemin relatif ou URL absolue.
 * - `init` : options fetch + `timeoutMs` + `noCache`.
 *
 * Retour :
 * - `Promise<T>` avec le JSON parsé.
 * - Throw `HttpError` si réponse non OK ou JSON invalide.
 */
export async function json<T>(path: string, init: JsonInit = {}): Promise<T> {
  const { url: url0 } = resolveUrl(path);
  const url = addTs(url0);
  const { timeoutMs = DEFAULT_TIMEOUT_MS, noCache = true, ...rest } = init;
  const t = withTimeout(timeoutMs);

  try {
    const headers = new Headers(rest.headers || {});
    headers.set("accept", "application/json");

    // Ne fixe Content-Type que si on envoie un body
    if (rest.method && rest.method !== "GET" && rest.method !== "HEAD") {
      headers.set("content-type", headers.get("content-type") || "application/json");
    }

    // Token public optionnel (dev)
    if (PUBLIC_TOKEN) headers.set("authorization", `Bearer ${PUBLIC_TOKEN}`);

    const res = await fetchWithRetry(url, {
      ...rest,
      headers,
      cache: noCache ? "no-store" : rest.cache,
      credentials: "same-origin",
      signal: t.signal,
    });

    const text = await res.text();
    if (!res.ok) throw new HttpError(res.status, res.statusText || "HTTP Error", text);

    try {
      return JSON.parse(text) as T;
    } catch {
      throw new HttpError(500, "Invalid JSON response", text);
    }
  } finally {
    t.done();
  }
}

/*─────────────────────────────── ETag caching (with fallback) ───────────────────────────────*/

/**
 * Variante de `json` avec support du cache ETag côté `localStorage`.
 *
 * Principe :
 * - Clé ETag = `etag:<url0>`, Clé body = `cache:<url0>`.
 * - Envoie `If-None-Match` si un ETag local existe.
 * - Si 304 + body en cache → retourne le cache.
 * - Si 304 mais rien en cache → refetch forcé sans ETag.
 * - Sinon, en cas de 200 :
 *     • parse le JSON,
 *     • sauvegarde ETag + body dans localStorage (si dispo).
 *
 * Paramètres :
 * - `pathOrUrl` : chemin relatif ou URL absolue.
 * - `init` : options fetch + `timeoutMs` + `noCache`.
 */
export async function fetchJsonWithEtag<T>(
  pathOrUrl: string,
  init: JsonInit = {}
): Promise<T> {
  const { url: url0 } = resolveUrl(pathOrUrl);
  const url = addTs(url0);
  const { timeoutMs = DEFAULT_TIMEOUT_MS, noCache = false, ...rest } = init;
  const t = withTimeout(timeoutMs);

  const etagKey = `etag:${url0}`;
  const bodyKey = `cache:${url0}`;
  const hasWindow = typeof window !== "undefined";
  const prevEtag = hasWindow ? localStorage.getItem(etagKey) : null;

  try {
    const headers = new Headers(rest.headers || {});
    headers.set("accept", "application/json");
    if (PUBLIC_TOKEN) headers.set("authorization", `Bearer ${PUBLIC_TOKEN}`);
    if (prevEtag) headers.set("if-none-match", prevEtag);

    const res = await fetchWithRetry(url, {
      ...rest,
      headers,
      cache: noCache ? "no-store" : rest.cache,
      credentials: "same-origin",
      signal: t.signal,
    });

    // 304 mais rien en cache local → refetch forcé sans ETag
    if (res.status === 304 && hasWindow) {
      const cached = localStorage.getItem(bodyKey);
      if (cached) return JSON.parse(cached) as T;

      const headers2 = new Headers(rest.headers || {});
      headers2.set("accept", "application/json");
      if (PUBLIC_TOKEN) headers2.set("authorization", `Bearer ${PUBLIC_TOKEN}`);

      const res2 = await fetchWithRetry(addTs(url0), {
        ...rest,
        headers: headers2,
        cache: "no-store",
        credentials: "same-origin",
        signal: t.signal,
      });
      const text2 = await res2.text();
      if (!res2.ok) throw new HttpError(res2.status, res2.statusText || "HTTP Error", text2);
      const data2 = JSON.parse(text2) as T;
      const etag2 = res2.headers.get("etag");
      if (etag2) {
        try {
          localStorage.setItem(etagKey, etag2);
          localStorage.setItem(bodyKey, text2);
        } catch {}
      }
      return data2;
    }

    const text = await res.text();
    if (!res.ok) throw new HttpError(res.status, res.statusText || "HTTP Error", text);

    const data = JSON.parse(text) as T;
    const etag = res.headers.get("etag");
    if (hasWindow && etag) {
      try {
        localStorage.setItem(etagKey, etag);
        localStorage.setItem(bodyKey, text);
      } catch {}
    }
    return data;
  } finally {
    t.done();
  }
}

/*─────────────────────────────── Shortcuts ───────────────────────────────*/

/**
 * Raccourci GET JSON typé.
 *
 * Exemple :
 *   const data = await getJSON<MyType>("/monitoring/intro");
 */
export const getJSON = <T>(
  path: string,
  init: Omit<JsonInit, "method" | "body"> = {}
) => json<T>(path, { ...init, method: "GET" });

/**
 * Raccourci POST JSON typé.
 *
 * - Sérialise `body` en JSON si non null/undefined.
 * - Laisse passer les options supplémentaires de `init`.
 */
export const postJSON = <T>(
  path: string,
  body?: unknown,
  init: Omit<JsonInit, "method"> = {}
) =>
  json<T>(path, {
    ...init,
    method: "POST",
    body: body == null ? undefined : JSON.stringify(body),
  });

/*─────────────────────────────── Forecast helpers ───────────────────────────────*/

/**
 * Sélectionne le tableau de lignes de prévisions dans un payload JSON
 * potentiellement multi-formats.
 *
 * Schémas supportés :
 * - `[...]`
 * - `{ data: [...] }`
 * - `{ data: { "15": [...] } }` (clé = horizon en minutes)
 * - `{ predictions: [...] }` (legacy)
 *
 * Si la forme est inattendue, log un warning et renvoie `[]`.
 */
export function selectForecastRows(payload: any, horizonMin = 15): any[] {
  if (Array.isArray(payload)) return payload;                 // root array
  if (Array.isArray(payload?.data)) return payload.data;      // { data: [...] }
  const k = String(horizonMin);                               // { data: { "15": [...] } }
  if (payload?.data?.[k] && Array.isArray(payload.data[k])) return payload.data[k];
  if (Array.isArray(payload?.predictions)) return payload.predictions; // legacy
  console.warn("[selectForecastRows] unexpected payload shape:", payload);
  return [];
}

/**
 * Helper haut niveau : récupère les lignes de prévision pour un horizon donné
 * via `/forecast/latest?h=...`, puis applique `selectForecastRows`.
 *
 * - Tolère un 404 (fichier GCS manquant) en renvoyant `[]`.
 * - Toute autre erreur est relancée.
 */
export async function getForecastRows(horizonMin = 15) {
  try {
    const payload = await getJSON(`/forecast/latest?h=${horizonMin}`);
    return selectForecastRows(payload, horizonMin);
  } catch (e: any) {
    if (e?.status === 404) return []; // tolerate missing GCS file
    throw e;
  }
}
