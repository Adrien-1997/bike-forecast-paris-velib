// ui/lib/http.ts
// Centralized HTTP utility — no cache/dedupe/retries, with timeout + JSON helpers.

export const API_BASE =
  (process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8081").replace(/\/$/, "");

// (optional debug – safe to keep)
console.log("[http] API_BASE =", process.env.NEXT_PUBLIC_API_BASE || "<empty>");

type JsonInit = RequestInit & {
  timeoutMs?: number;
  noCache?: boolean;
};

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

// ───────────────────────────────
// Helpers
// ───────────────────────────────

function withTimeout(ms = 10_000) {
  const ctrl = new AbortController();
  const id = setTimeout(() => ctrl.abort(), ms);
  return { signal: ctrl.signal, done: () => clearTimeout(id) };
}

function resolveUrl(pathOrUrl: string) {
  const hasProtocol = /^https?:\/\//i.test(pathOrUrl);
  const base = hasProtocol ? "" : API_BASE;
  const path = hasProtocol ? pathOrUrl : `${API_BASE}${pathOrUrl}`;
  return { base, path };
}

function addTs(url: string) {
  const ts = `_ts=${Date.now()}`;
  return `${url}${url.includes("?") ? "&" : "?"}${ts}`;
}

// ───────────────────────────────
// Core JSON request
// ───────────────────────────────

export async function json<T>(path: string, init: JsonInit = {}): Promise<T> {
  const { path: url0 } = resolveUrl(path);
  const url = addTs(url0);

  const { timeoutMs = 10_000, noCache = true, ...rest } = init;
  const t = withTimeout(timeoutMs);

  try {
    const res = await fetch(url, {
      ...rest,
      headers: {
        accept: "application/json",
        "Content-Type": "application/json",
        ...(rest.headers || {}),
      },
      cache: noCache ? "no-store" : rest.cache,
      signal: t.signal,
    });

    const text = await res.text();

    if (!res.ok) {
      console.error("[http] error", res.status, res.statusText, text.slice(0, 500));
      throw new HttpError(res.status, res.statusText, text);
    }

    try {
      return JSON.parse(text) as T;
    } catch {
      console.error("[http] non-JSON body:", text.slice(0, 800));
      throw new HttpError(500, "Invalid JSON response", text);
    }
  } finally {
    t.done();
  }
}

// ───────────────────────────────
// ETag JSON request (with localStorage)
// ───────────────────────────────

/**
 * fetchJsonWithEtag
 * - Accepte chemin relatif ("/monitoring/...") OU URL absolue ("https://...").
 * - Utilise If-None-Match / ETag + localStorage pour cache offline-léger.
 * - Renvoie le JSON parsé. En 304 → retourne la copie cache.
 */
export async function fetchJsonWithEtag<T>(
  pathOrUrl: string,
  init: JsonInit = {}
): Promise<T> {
  const { path: url0 } = resolveUrl(pathOrUrl);
  const url = addTs(url0);

  const { timeoutMs = 10_000, noCache = false, ...rest } = init;
  const t = withTimeout(timeoutMs);

  // Clés de cache
  const etagKey = `etag:${url0}`;
  const bodyKey = `cache:${url0}`;

  // Récup ETag existant
  const prevEtag = typeof window !== "undefined" ? localStorage.getItem(etagKey) : null;

  try {
    const res = await fetch(url, {
      ...rest,
      headers: {
        accept: "application/json",
        "Content-Type": "application/json",
        ...(prevEtag ? { "If-None-Match": prevEtag } : {}),
        ...(rest.headers || {}),
      },
      // Ici on laisse le cache navigateur par défaut si noCache=false
      cache: noCache ? "no-store" : rest.cache,
      signal: t.signal,
    });

    if (res.status === 304 && typeof window !== "undefined") {
      const cached = localStorage.getItem(bodyKey);
      if (cached) {
        try {
          return JSON.parse(cached) as T;
        } catch {
          // Cache corrompu → on ignore et on continue
        }
      }
      // Pas de cache exploitable → on tente quand même de lire le body (certains proxies envoient 304 + body)
    }

    const text = await res.text();

    if (!res.ok) {
      // Log court
      console.error("[http][etag] error", res.status, res.statusText, text.slice(0, 500));
      throw new HttpError(res.status, res.statusText, text);
    }

    try {
      const data = JSON.parse(text) as T;

      // Stocke ETag + body si présents
      const etag = res.headers.get("ETag");
      if (typeof window !== "undefined" && etag) {
        try {
          localStorage.setItem(etagKey, etag);
          localStorage.setItem(bodyKey, text);
        } catch {
          // quota full / privacy mode → ignorer silencieusement
        }
      }

      return data;
    } catch {
      console.error("[http][etag] non-JSON body:", text.slice(0, 800));
      throw new HttpError(500, "Invalid JSON response", text);
    }
  } finally {
    t.done();
  }
}

// ───────────────────────────────
// Convenience wrappers
// ───────────────────────────────

export const getJSON = <T>(
  path: string,
  init: Omit<JsonInit, "method" | "body"> = {}
) => json<T>(path, { ...init, method: "GET" });

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

// ───────────────────────────────
// Forecast payload normalizer
// ───────────────────────────────

/**
 * Normalize forecast payload to a flat array of rows, regardless of shape:
 * - [{...}]
 * - { data: { "15": [ {...} ] }, generated_at, horizons }
 * - { predictions: [ {...} ] }
 */
export function selectForecastRows(payload: any, horizonMin = 15): any[] {
  if (Array.isArray(payload)) return payload;
  const k = String(horizonMin);
  if (payload?.data?.[k] && Array.isArray(payload.data[k])) return payload.data[k];
  if (Array.isArray(payload?.predictions)) return payload.predictions;
  console.warn("[selectForecastRows] unexpected payload shape:", payload);
  return [];
}

/** Convenience: always return an array of forecast rows for a given horizon. */
export async function getForecastRows(horizonMin = 15) {
  const payload = await getJSON(`/forecast/latest?h=${horizonMin}`);
  return selectForecastRows(payload, horizonMin);
}
