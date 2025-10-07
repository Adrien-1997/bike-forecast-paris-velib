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

// ───────────────────────────────
// Helpers
// ───────────────────────────────

function withTimeout(ms = 10_000) {
  const ctrl = new AbortController();
  const id = setTimeout(() => ctrl.abort(), ms);
  return { signal: ctrl.signal, done: () => clearTimeout(id) };
}

// ───────────────────────────────
// Core JSON request
// ───────────────────────────────

export async function json<T>(path: string, init: JsonInit = {}): Promise<T> {
  const tsParam = `_ts=${Date.now()}`; // bust browser cache
  const url = `${API_BASE}${path}${path.includes("?") ? "&" : "?"}${tsParam}`;

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
      throw new Error(`${res.status} ${res.statusText}`);
    }

    try {
      return JSON.parse(text) as T;
    } catch {
      console.error("[http] non-JSON body:", text.slice(0, 800));
      throw new Error("Invalid JSON response");
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
