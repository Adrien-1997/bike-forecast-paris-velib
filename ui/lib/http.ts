// ui/lib/http.ts
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://127.0.0.1:8081';

type JsonInit = RequestInit & { timeoutMs?: number; noCache?: boolean; dedupeKey?: string };

const inflight = new Map<string, Promise<any>>();

function withTimeout(ms = 10_000) {
  const ctrl = new AbortController();
  const id = setTimeout(() => ctrl.abort(), ms);
  return { signal: ctrl.signal, done: () => clearTimeout(id) };
}

async function _doJson<T>(path: string, init: JsonInit = {}): Promise<T> {
  const url = `${API_BASE}${path}${path.includes('?') ? '&' : '?'}_ts=${Date.now()}`;

  const { timeoutMs = 10_000, noCache = true, ...rest } = init;
  const t = withTimeout(timeoutMs);
  try {
    const res = await fetch(url, {
      ...rest,
      headers: {
        accept: 'application/json',
        'Content-Type': 'application/json',
        ...(rest.headers || {}),
      },
      cache: noCache ? 'no-store' : rest.cache,
      signal: t.signal,
    });

    const text = await res.text();
    if (!res.ok) {
      console.error('[http] error', res.status, res.statusText, text.slice(0, 500));
      throw new Error(`${res.status} ${res.statusText}`);
    }
    try {
      return JSON.parse(text) as T;
    } catch {
      console.error('[http] non-JSON body:', text.slice(0, 800));
      throw new Error('Non-JSON response');
    }
  } finally {
    t.done();
  }
}

// Retry simple (exponentiel) + dédoublonnage d’appels identiques
export async function json<T>(path: string, init: JsonInit = {}, retries = 1): Promise<T> {
  const key = init.dedupeKey ?? `${path}::${init.method ?? 'GET'}::${init.body ?? ''}`;
  if (inflight.has(key)) return inflight.get(key)!;

  const run = async (): Promise<T> => {
    try {
      return await _doJson<T>(path, init);
    } catch (e) {
      if (retries > 0) {
        await new Promise(r => setTimeout(r, 500 * Math.pow(2, 1 - retries)));
        return run.bind(null)() as Promise<T>;
      }
      throw e;
    }
  };

  const p = run().finally(() => inflight.delete(key));
  inflight.set(key, p);
  return p;
}
