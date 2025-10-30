// ui/lib/services/monitoring/intro.ts
// Service pour /monitoring/intro (ETag + base/token gérés par lib/http)

import { fetchJsonWithEtag } from "@/lib/http";

/* ───────────────────────── Types ───────────────────────── */
export type Led = "ok" | "warn" | "down";

export type IntroDoc = {
  schema_version: string;
  generated_at: string;
  tz: string;
  kpis: {
    stations_active: number | null;
    freshness_p95_min: number | null;
    coverage_7d_pct: number | null;
    psi_global?: number | null;
    /** ex: "vX.Y.Z / vA.B.C" (h15/h60) */
    model_versions: string | null;
  };
  statuses: {
    api_stations: {
      led: Led;
      stations_active: number | null;
    };
    batch_forecast: {
      led: Led;
      age_min: number | null;
      source_generated_at: string | null;
      rows?: number | null;
    };
    weather_provider: {
      led: Led;
      freshness_p95_min: number | null;
      source_generated_at: string | null;
    };
    /** NEW: statut drift basé sur psi_global (seuils 0.10 / 0.20) */
    data_drift?: {
      led: Led;
      psi_global: number | null;
      top_feature?: string | null;
      top_feature_psi?: number | null;
      source_generated_at: string | null;
    };
  };
  activity: Array<{
    label: string;
    value: unknown;
    generated_at: string | null;
  }>;
  sources: Record<string, string>;
};

/* ───────────────────────── Helpers ───────────────────────── */
const path = (suffix = "") => `/monitoring/intro${suffix}`;

/* ───────────────────────── API (ETag) ───────────────────────── */
export const getMonitoringIntro = () =>
  fetchJsonWithEtag<IntroDoc>(path(""));

/** Time-travel support: ?at=YYYY-MM-DDTHH-MM-SSZ */
export const getMonitoringIntroAt = (atIsoStamp: string) =>
  fetchJsonWithEtag<IntroDoc>(`${path("")}?at=${encodeURIComponent(atIsoStamp)}`);
