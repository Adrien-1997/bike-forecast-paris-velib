// components/BadgesBar.tsx
import type { Badges } from "@/lib/types";

export type BadgesProps = { data?: Badges | null };

function fmt(n: number | null | undefined, d = 1) {
  const v = Number(n);
  return Number.isFinite(v)
    ? new Intl.NumberFormat(undefined, { maximumFractionDigits: d }).format(v)
    : "—";
}

function parisHHmm(iso?: string | null): string {
  if (!iso) return "—";
  const s = iso.endsWith("Z") ? iso : `${iso}Z`;
  const d = new Date(s);
  return new Intl.DateTimeFormat("fr-FR", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "Europe/Paris",
  }).format(d);
}

/**
 * Emoji météo contextuel :
 * - Priorité : précipitations > vent > température > calme
 * - Jour/Nuit déterminé par `targetIso` (Europe/Paris). Fallback: heure locale si absent.
 */
function weatherEmoji(
  temp?: number | null,
  precip?: number | null,
  wind?: number | null,
  targetIso?: string | null
): string {
  const t = Number(temp);
  const p = Number(precip);
  const w = Number(wind);

  // Heure locale Paris au moment de la PRÉVISION (target_ts_utc)
  let isNight = false;
  if (typeof targetIso === "string" && targetIso.trim() !== "") {
    const s = targetIso.endsWith("Z") ? targetIso : `${targetIso}Z`;
    const d = new Date(s);
    const hourStr = new Intl.DateTimeFormat("fr-FR", {
      hour: "2-digit",
      hour12: false,
      timeZone: "Europe/Paris",
    }).format(d);
    const h = parseInt(hourStr, 10);
    isNight = h < 6 || h >= 21; // nuit 21h→6h
  } else {
    // Fallback : maintenant (rare si target_ts_utc est systématique)
    const nowHour = new Intl.DateTimeFormat("fr-FR", {
      hour: "2-digit",
      hour12: false,
      timeZone: "Europe/Paris",
    }).format(new Date());
    const h = parseInt(nowHour, 10);
    isNight = h < 6 || h >= 21;
  }

  // 🌧️ Précipitations dominantes
  if (Number.isFinite(p)) {
    if (p >= 5)  return isNight ? "🌩️" : "⛈️"; // orage / très forte pluie
    if (p >= 1)  return "🌧️";                 // pluie modérée
    if (p > 0)   return "🌦️";                 // petites averses
  }

  // 🌬️ Vent (si pas de pluie)
  if (Number.isFinite(w) && w >= 14) return "💨"; // très venteux / rafales
  if (Number.isFinite(w) && w >= 8)  return "🌬️"; // vent sensible

  // ❄️ Température (sans pluie/vent fort)
  if (Number.isFinite(t)) {
    if (t <= -2) return "🥶"; // grand froid
    if (t <= 5)  return "❄️"; // froid
    if (t >= 33) return "🥵"; // canicule
    if (t >= 26) return isNight ? "🌙" : "☀️"; // chaud jour = soleil, nuit = lune
  }

  // 🌤️ Conditions calmes
  if (isNight) {
    // Nuit sans pluie ni vent fort : lune
    return "🌙";
  } else {
    // Jour calme : partiellement ensoleillé
    return "🌤️";
  }
}

export default function BadgesBar({ data }: BadgesProps) {
  if (!data) return null;

  // Supporte {weather:{...}, freshness:{...}, meta:{...}} et l’ancien format plat
  const w  = (data as any)?.weather ?? data ?? {};
  const fr = (data as any)?.freshness ?? {};
  const meta = (data as any)?.meta ?? {};

  const temp   = w?.temp_C ?? null;
  const precip = w?.precip_mm ?? null;
  const wind   = w?.wind_mps ?? null;

  // Fraîcheur des données (minutes) — priorité à la normalisée → freshness.age_minutes → fallback legacy
  const ageMin: number | null =
    (meta?.freshness_min_norm as number | undefined) ??
    (fr?.age_minutes as number | undefined) ??
    (data as any)?.parquet_age_min ??
    null;

  // Info complémentaires (tooltip)
  const predTsISO    = meta?.pred_ts_utc ?? null;    // heure de run du modèle
  const targetTsISO  = meta?.target_ts_utc ?? null;  // heure cible de la prévision

  // Emoji météo basé sur la cible (jour/nuit)
  const emo = weatherEmoji(temp, precip, wind, targetTsISO);

  // Palette simple selon âge des données
  // <= 10 min → vert ; 11-20 → amber ; > 20 → rouge
  const age = Number(ageMin);
  const freshClass =
    Number.isFinite(age)
      ? age <= 10
        ? "bg-emerald-100 text-emerald-900"
        : age <= 20
          ? "bg-amber-100 text-amber-900"
          : "bg-rose-100 text-rose-900"
      : "bg-neutral-100 text-neutral-900";

  const freshnessLabel = Number.isFinite(age)
    ? (age === 0 ? "Actualisées à l’instant" : `Données mises à jour il y a ${fmt(age, 0)} min`)
    : "Fraîcheur inconnue";

  const tooltip =
    `Modèle: ${predTsISO ? parisHHmm(predTsISO) : "—"} · ` +
    `Prévision pour: ${targetTsISO ? parisHHmm(targetTsISO) : "—"}`;

  return (
    <div className="badges flex flex-wrap gap-2">
      {/* Météo */}
      <div className="badge inline-flex items-center gap-2 rounded px-2 py-1 bg-neutral-100 text-neutral-900">
        <span className="badge-dot inline-block h-2 w-2 rounded-full bg-neutral-400" />
        <span>{emo} Météo</span>
        <span className="small opacity-80">• {fmt(temp, 0)} °C</span>
        <span className="small opacity-80">• {fmt(precip, 1)} mm</span>
        <span className="small opacity-80">• {fmt(wind, 1)} m/s</span>
      </div>

      {/* Fraîcheur des données (tbin_latest) */}
      <div
        className={`badge inline-flex items-center gap-2 rounded px-2 py-1 ${freshClass}`}
        title={tooltip}
      >
        <span className="badge-dot inline-block h-2 w-2 rounded-full bg-neutral-400" />
        <span>🕓 Données</span>
        <span className="small opacity-80">• {freshnessLabel}</span>
      </div>
    </div>
  );
}