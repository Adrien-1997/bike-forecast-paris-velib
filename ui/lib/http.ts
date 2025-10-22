// ui/lib/http.ts
// Centralized HTTP utility — timeout, JSON parsing, ETag caching, and error handling.

export const API_BASE = (
  process.env.NEXT_PUBLIC_API_BASE ||
  "https://velib-api-160046094975.europe-west1.run.app"
).replace(/\/$/, "");

export const API_TOKEN = process.env.NEXT_PUBLIC_API_TOKEN || "";

// Debug logs
console.log("[http] API_BASE =", API_BASE);
if (API_TOKEN) console.log("[http] using token (set)");
else console.warn("[http] no API token defined");

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

/*─────────────────────────────── Helpers ───────────────────────────────*/

function withTimeout(ms = 10_000) {
  const ctrl = new AbortController();
  const id = setTimeout(() => ctrl.abort(), ms);
  return { signal: ctrl.signal, done: () => clearTimeout(id) };
}

function resolveUrl(pathOrUrl: string) {
  const hasProtocol = /^https?:\/\//i.test(pathOrUrl);
  const path = hasProtocol ? pathOrUrl : `${API_BASE}${pathOrUrl}`;
  return { path };
}

function addTs(url: string) {
  const ts = `_ts=${Date.now()}`;
  return `${url}${url.includes("?") ? "&" : "?"}${ts}`;
}

/*─────────────────────────────── Core JSON fetch ───────────────────────────────*/

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
        ...(API_TOKEN ? { Authorization: `Bearer ${API_TOKEN}` } : {}),
        ...(rest.headers || {}),
      },
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

/*─────────────────────────────── ETag caching ───────────────────────────────*/

export async function fetchJsonWithEtag<T>(
  pathOrUrl: string,
  init: JsonInit = {}
): Promise<T> {
  const { path: url0 } = resolveUrl(pathOrUrl);
  const url = addTs(url0);
  const { timeoutMs = 10_000, noCache = false, ...rest } = init;
  const t = withTimeout(timeoutMs);

  const etagKey = `etag:${url0}`;
  const bodyKey = `cache:${url0}`;
  const prevEtag =
    typeof window !== "undefined" ? localStorage.getItem(etagKey) : null;

  try {
    const res = await fetch(url, {
      ...rest,
      headers: {
        accept: "application/json",
        "Content-Type": "application/json",
        ...(API_TOKEN ? { Authorization: `Bearer ${API_TOKEN}` } : {}),
        ...(prevEtag ? { "If-None-Match": prevEtag } : {}),
        ...(rest.headers || {}),
      },
      cache: noCache ? "no-store" : rest.cache,
      signal: t.signal,
    });

    if (res.status === 304 && typeof window !== "undefined") {
      const cached = localStorage.getItem(bodyKey);
      if (cached) return JSON.parse(cached) as T;
    }

    const text = await res.text();
    if (!res.ok) throw new HttpError(res.status, res.statusText, text);

    const data = JSON.parse(text) as T;
    const etag = res.headers.get("ETag");
    if (typeof window !== "undefined" && etag) {
      try {
        localStorage.setItem(etagKey, etag);
        localStorage.setItem(bodyKey, text);
      } catch {
        /* ignore quota or privacy errors */
      }
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
  if (Array.isArray(payload)) return payload;
  const k = String(horizonMin);
  if (payload?.data?.[k] && Array.isArray(payload.data[k])) return payload.data[k];
  if (Array.isArray(payload?.predictions)) return payload.predictions;
  console.warn("[selectForecastRows] unexpected payload shape:", payload);
  return [];
}

export async function getForecastRows(horizonMin = 15) {
  const payload = await getJSON(`/forecast/latest?h=${horizonMin}`);
  return selectForecastRows(payload, horizonMin);
}
