// ui/lib/services/monitoring.ts
import { fetchJsonWithEtag } from "@/lib/http";
import type {
  MonitoringManifest,
  PerfDailyResponse,
  PerfSegmentsResponse,
} from "@/lib/types";

const base = (process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8081").replace(/\/$/, "");

// ───────────────────────────────
// Helpers (ne jettent jamais)
// ───────────────────────────────
function isAbortOrHttp(e: any) {
  const name = e?.name || e?.constructor?.name || "";
  const status = e?.status;
  return name.includes("AbortError") || status === 404 || status === 500;
}
function safeLog(where: string, e: any) {
  // bruit minimal en dev
  if (process.env.NODE_ENV !== "production") {
    console.warn(`[monitoring:${where}]`, e?.status || e?.name || e?.message || e);
  }
}

// ───────────────────────────────
// Manifest
// ───────────────────────────────
export async function getManifest(): Promise<MonitoringManifest | null> {
  try {
    // Manifest peut être un peu lent en local
    return await fetchJsonWithEtag<MonitoringManifest>(`${base}/monitoring/manifest`, {
      timeoutMs: 25_000,
    });
  } catch (e: any) {
    if (!isAbortOrHttp(e)) safeLog("manifest", e);
    return null;
  }
}

// ───────────────────────────────
// Model Performance
// ───────────────────────────────
export async function getPerfDaily(h: number = 15): Promise<PerfDailyResponse | null> {
  try {
    // h=60 est souvent absent → on n'attend pas trop longtemps
    const timeoutMs = h === 60 ? 6_000 : 20_000;
    return await fetchJsonWithEtag<PerfDailyResponse>(
      `${base}/monitoring/model/perf/daily?h=${h}`,
      { timeoutMs }
    );
  } catch (e: any) {
    if (!isAbortOrHttp(e)) safeLog(`perf/daily?h=${h}`, e);
    return null;
  }
}

export async function getPerfSegments(h: number = 15): Promise<PerfSegmentsResponse> {
  try {
    const timeoutMs = h === 60 ? 6_000 : 20_000;
    return await fetchJsonWithEtag<PerfSegmentsResponse>(
      `${base}/monitoring/model/perf/segments?h=${h}`,
      { timeoutMs }
    );
  } catch (e: any) {
    if (!isAbortOrHttp(e)) safeLog(`perf/segments?h=${h}`, e);
    return [];
  }
}

// ───────────────────────────────
// Network
// ───────────────────────────────
export async function getNetworkDynamics<T = unknown>(): Promise<T | null> {
  try {
    return await fetchJsonWithEtag<T>(`${base}/monitoring/network/dynamics`, {
      timeoutMs: 25_000,
    });
  } catch (e: any) {
    if (!isAbortOrHttp(e)) safeLog("network/dynamics", e);
    return null;
  }
}

export async function getNetworkStations<T = unknown>(): Promise<T | []> {
  try {
    return await fetchJsonWithEtag<T>(`${base}/monitoring/network/stations`, {
      timeoutMs: 25_000,
    });
  } catch (e: any) {
    if (!isAbortOrHttp(e)) safeLog("network/stations", e);
    return [] as unknown as T;
  }
}

// ───────────────────────────────
// Drift
// ───────────────────────────────
export async function getDriftSummary<T = unknown>(): Promise<T | null> {
  try {
    return await fetchJsonWithEtag<T>(`${base}/monitoring/drift/summary`, {
      timeoutMs: 20_000,
    });
  } catch (e: any) {
    if (!isAbortOrHttp(e)) safeLog("drift/summary", e);
    return null;
  }
}

// ───────────────────────────────
// Documentation (dictionary, methodology, exports)
// ───────────────────────────────
export async function getDoc<T = unknown>(
  name: "dictionary" | "methodology" | "exports"
): Promise<T | null> {
  try {
    return await fetchJsonWithEtag<T>(`${base}/monitoring/docs/${name}`, {
      timeoutMs: 15_000,
    });
  } catch (e: any) {
    if (!isAbortOrHttp(e)) safeLog(`docs/${name}`, e);
    return null;
  }
}

// ───────────────────────────────
// Model Explainability & Health
// ───────────────────────────────
export async function getModelExplainability<T = unknown>(): Promise<T | null> {
  try {
    return await fetchJsonWithEtag<T>(`${base}/monitoring/model/explainability`, {
      timeoutMs: 20_000,
    });
  } catch (e: any) {
    if (!isAbortOrHttp(e)) safeLog("model/explainability", e);
    return null;
  }
}

export async function getModelHealth<T = unknown>(): Promise<T | null> {
  try {
    return await fetchJsonWithEtag<T>(`${base}/monitoring/model/health`, {
      timeoutMs: 20_000,
    });
  } catch (e: any) {
    if (!isAbortOrHttp(e)) safeLog("model/health", e);
    return null;
  }
}
