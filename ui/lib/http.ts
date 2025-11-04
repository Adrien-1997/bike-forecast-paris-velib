// ui/lib/http.ts
// Centralized HTTP utility — timeout, JSON parsing, ETag caching, and error handling.

/*─────────────────────────────── Base URL (proxy Netlify) ───────────────────────────────*/

function joinUrl(base: string, path: string) {
  const b = base.replace(/\/+$/, "");
  const p = path.replace(/^\/+/, "");
  return `${b}/${p}`;
}

/**
 * Always prefer NEXT_PUBLIC_API_BASE; fallback to Netlify Function proxy.
 * No hard-coded Cloud Run URL to avoid secret-scan or env drift.
 */
export const API_BASE: string = (
  process.env.NEXT_PUBLIC_API_BASE && process.env.NEXT_PUBLIC_API_BASE.trim()
) || "/.netlify/functions/api-proxy";

// Optional public token (for local/dev only)
const PUBLIC_TOKEN = process.env.NEXT_PUBLIC_API_TOKEN ?? "";

// Default timeout
export const DEFAULT_TIMEOUT_MS = Number(
  process.env.NEXT_PUBLIC_HTTP_TIMEOUT_MS ?? "30000"
); // 30s

if (typeof window !== "undefined") {
  // Harmless debug (client only)
  console.log("[http] API_BASE =", API_BASE);
  console.log("[http] timeout =", DEFAULT_TIMEOUT_MS, "ms");
}

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
    if (tries > 1 && (e?.name === "AbortError" || /aborted|timeout/i.test(String(e)))) {
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

    // Only set Content-Type when sending a body
    if (rest.method && rest.method !== "GET" && rest.method !== "HEAD") {
      headers.set("content-type", headers.get("content-type") || "application/json");
    }

    // Optional public token for dev
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

    // 304 but nothing cached → fetch fresh
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

export function selectForecastRows(payload: any, horizonMin = 15): any[] {
  if (Array.isArray(payload)) return payload;                 // root array
  if (Array.isArray(payload?.data)) return payload.data;      // { data: [...] }
  const k = String(horizonMin);                               // { data: { "15": [...] } }
  if (payload?.data?.[k] && Array.isArray(payload.data[k])) return payload.data[k];
  if (Array.isArray(payload?.predictions)) return payload.predictions; // legacy
  console.warn("[selectForecastRows] unexpected payload shape:", payload);
  return [];
}

export async function getForecastRows(horizonMin = 15) {
  try {
    const payload = await getJSON(`/forecast/latest?h=${horizonMin}`);
    return selectForecastRows(payload, horizonMin);
  } catch (e: any) {
    if (e?.status === 404) return []; // tolerate missing GCS file
    throw e;
  }
}
