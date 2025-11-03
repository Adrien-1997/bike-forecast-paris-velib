// ui/lib/http.ts
// Centralized HTTP utility — timeout, JSON parsing, ETag caching, and error handling.

/*─────────────────────────────── Base URL & Token ───────────────────────────────*/

// En production (Netlify), on passe par le rewrite /api → évite CORS.
// En dev, si tu as un proxy Next: http://localhost:3000/api/proxy ; sinon /api marche aussi.
function joinUrl(base: string, path: string) {
  const b = base.replace(/\/+$/, "");
  const p = path.replace(/^\/+/, "");
  return `${b}/${p}`;
}

const PUBLIC_TOKEN = process.env.NEXT_PUBLIC_API_TOKEN ?? "";

export const API_BASE =
  (process.env.NEXT_PUBLIC_API_BASE && process.env.NEXT_PUBLIC_API_BASE.trim()) ||
  (process.env.NODE_ENV === "production" ? "/api" : "http://localhost:3000/api/proxy");

export const DEFAULT_TIMEOUT_MS = Number(
  process.env.NEXT_PUBLIC_HTTP_TIMEOUT_MS ?? "30000"
); // 30s par défaut

console.log("[http] API_BASE =", API_BASE);
console.log("[http] timeout =", DEFAULT_TIMEOUT_MS, "ms");

/*─────────────────────────────── Types & Helpers ───────────────────────────────*/

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

function withTimeout(ms = DEFAULT_TIMEOUT_MS) {
  const ctrl = new AbortController();
  const id = setTimeout(() => ctrl.abort(), ms);
  return { signal: ctrl.signal, done: () => clearTimeout(id) };
}

function resolveUrl(pathOrUrl: string) {
  const hasProtocol = /^https?:\/\//i.test(pathOrUrl);
  const url = hasProtocol ? pathOrUrl : joinUrl(API_BASE, pathOrUrl);
  return { url };
}

function addTs(url: string) {
  const ts = `_ts=${Date.now()}`;
  return `${url}${url.includes("?") ? "&" : "?"}${ts}`;
}

async function fetchWithRetry(input: RequestInfo, init: RequestInit, tries = 2) {
  try {
    return await fetch(input, init);
  } catch (e: any) {
    if (tries > 1 && (e?.name === "AbortError" || /aborted/i.test(String(e)))) {
      console.warn("[http] retry after abort/timeout →", input);
      return fetchWithRetry(input, init, tries - 1);
    }
    throw e;
  }
}

/*─────────────────────────────── Core JSON fetch ───────────────────────────────*/

export async function json<T>(path: string, init: JsonInit = {}): Promise<T> {
  const { url: url0 } = resolveUrl(path);
  const url = addTs(url0);
  const { timeoutMs = DEFAULT_TIMEOUT_MS, noCache = true, ...rest } = init;
  const t = withTimeout(timeoutMs);

  try {
    const headers = new Headers(rest.headers || {});
    headers.set("accept", "application/json");
    headers.set("Content-Type", "application/json");
    if (PUBLIC_TOKEN) headers.set("Authorization", `Bearer ${PUBLIC_TOKEN}`);

    const res = await fetchWithRetry(url, {
      ...rest,
      headers,
      cache: noCache ? "no-store" : rest.cache,
      signal: t.signal,
    });

    const text = await res.text();
    if (!res.ok) throw new HttpError(res.status, res.statusText, text);

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
  const prevEtag =
    typeof window !== "undefined" ? localStorage.getItem(etagKey) : null;

  try {
    const headers = new Headers(rest.headers || {});
    headers.set("accept", "application/json");
    headers.set("Content-Type", "application/json");
    if (PUBLIC_TOKEN) headers.set("Authorization", `Bearer ${PUBLIC_TOKEN}`);
    if (prevEtag) headers.set("If-None-Match", prevEtag);

    const res = await fetchWithRetry(url, {
      ...rest,
      headers,
      cache: noCache ? "no-store" : rest.cache,
      signal: t.signal,
    });

    // 304 mais aucun cache local → re-fetch complet
    if (res.status === 304 && typeof window !== "undefined") {
      const cached = localStorage.getItem(bodyKey);
      if (cached) return JSON.parse(cached) as T;

      const headers2 = new Headers(rest.headers || {});
      headers2.set("accept", "application/json");
      headers2.set("Content-Type", "application/json");
      if (PUBLIC_TOKEN) headers2.set("Authorization", `Bearer ${PUBLIC_TOKEN}`);

      const res2 = await fetchWithRetry(addTs(url0), {
        ...rest,
        headers: headers2,
        cache: "no-store",
        signal: t.signal,
      });
      const text2 = await res2.text();
      if (!res2.ok) throw new HttpError(res2.status, res2.statusText, text2);
      const data2 = JSON.parse(text2) as T;
      const etag2 = res2.headers.get("ETag");
      if (etag2) {
        try {
          localStorage.setItem(etagKey, etag2);
          localStorage.setItem(bodyKey, text2);
        } catch {}
      }
      return data2;
    }

    const text = await res.text();
    if (!res.ok) throw new HttpError(res.status, res.statusText, text);

    const data = JSON.parse(text) as T;
    const etag = res.headers.get("ETag");
    if (typeof window !== "undefined" && etag) {
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

/*─────────────────────────────── Forecast helpers ───────────────────────────────*/

// Robust row selector for forecast payloads (supports multiple shapes)
export function selectForecastRows(payload: any, horizonMin = 15): any[] {
  // Root array
  if (Array.isArray(payload)) return payload;

  // Current format: { data: [...] }
  if (Array.isArray(payload?.data)) return payload.data;

  // Legacy keyed by horizon: { data: { "15": [...] } }
  const k = String(horizonMin);
  if (payload?.data?.[k] && Array.isArray(payload.data[k])) return payload.data[k];

  // Other legacy: { predictions: [...] }
  if (Array.isArray(payload?.predictions)) return payload.predictions;

  console.warn("[selectForecastRows] unexpected payload shape:", payload);
  return [];
}

export async function getForecastRows(horizonMin = 15) {
  try {
    const payload = await getJSON(`/forecast/latest?h=${horizonMin}`);
    return selectForecastRows(payload, horizonMin);
  } catch (e: any) {
    // GCS file missing → API returns 404; return [] for UI resilience
    if (e?.status === 404) return [];
    throw e;
  }
}
