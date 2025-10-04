// components/BadgesBar.tsx
import type { Badges } from "@/lib/types";

export type BadgesProps = { data?: Badges | null };

function fmt(n: number | null | undefined, d = 1) {
  const v = Number(n);
  return Number.isFinite(v)
    ? new Intl.NumberFormat(undefined, { maximumFractionDigits: d }).format(v)
    : "—";
}

function weatherEmoji(temp?: number | null, precip?: number | null, wind?: number | null) {
  const t = Number(temp), p = Number(precip), w = Number(wind);

  if (Number.isFinite(p) && p >= 2) return "⛈️";
  if (Number.isFinite(p) && p > 0)  return "🌧️";
  if (Number.isFinite(w) && w >= 10) return "🌬️";
  if (Number.isFinite(t) && t <= 3)  return "❄️";
  if (Number.isFinite(t) && t >= 30) return "🥵";
  return "🌤️";
}

export default function BadgesBar({ data }: BadgesProps) {
  // supporte /badges {weather:{...}, freshness:{...}} et l’ancien format plat
  const w  = (data as any)?.weather ?? data ?? {};
  const fr = (data as any)?.freshness ?? {};

  const temp   = w?.temp_C ?? null;
  const precip = w?.precip_mm ?? null;
  const wind   = w?.wind_mps ?? null;
  const ageMin = (fr as any)?.age_minutes ?? (data as any)?.parquet_age_min ?? null;

  const emo = weatherEmoji(temp, precip, wind);

  return (
    <div className="badges">
      <div className="badge">
        <span className="badge-dot" />
        <span>{emo} Météo</span>
        <span className="small">• {fmt(temp, 0)} °C</span>
        <span className="small">• {fmt(precip, 1)} mm</span>
        <span className="small">• {fmt(wind, 1)} m/s</span>
      </div>

      <div className="badge">
        <span className="badge-dot" />
        <span>Parquet</span>
        <span className="small">• fraîcheur {fmt(ageMin, 0)} min</span>
      </div>
    </div>
  );
}