import { useMemo } from "react";

export type KpiItem = {
  label: string;
  value: number | string | null | undefined;
  /** Optional human formatter, e.g. (v)=>`${v?.toFixed(1)}%` */
  fmt?: (v: number | string | null | undefined) => string;
  /** Subtext under the value (e.g., window, units) */
  hint?: string;
  /** Optional delta indicator (percent or abs). Shown as small pill. */
  delta?: { value: number; suffix?: string; tone?: "ok" | "warn" | "err" | "muted" };
  /** If true, visually deemphasize the card */
  muted?: boolean;
};

export type KpiBarProps = {
  items: KpiItem[];
  /** Compact vertical spacing & font sizes */
  dense?: boolean;
  /** Replace content with shimmer skeletons */
  loading?: boolean;
  /** Show a small inline error banner on the right */
  errorText?: string | null;
  /** Optional ARIA label for the whole bar */
  ariaLabel?: string;
  /** If true, allow horizontal scroll on narrow screens (default true) */
  scrollable?: boolean;
};

function defaultFmt(v: KpiItem["value"]) {
  if (typeof v === "number" && Number.isFinite(v)) return v.toLocaleString("fr-FR");
  if (typeof v === "string" && v.trim().length) return v;
  return "—";
}

function DeltaPill({
  value,
  suffix,
  tone = "muted",
}: { value: number; suffix?: string; tone?: "ok" | "warn" | "err" | "muted" }) {
  const sign = value > 0 ? "+" : value < 0 ? "−" : "";
  const abs = Math.abs(value);
  return (
    <span className={`kpi-delta kpi-delta--${tone}`} aria-label={`delta ${sign}${abs}${suffix ?? ""}`}>
      {sign}
      {abs}
      {suffix ?? ""}
    </span>
  );
}

export default function KpiBar({
  items,
  dense = false,
  loading = false,
  errorText = null,
  ariaLabel = "KPI bar",
  scrollable = true,
}: KpiBarProps) {
  const showError = Boolean(errorText);

  const content = useMemo(() => {
    if (loading) {
      // Skeletons: same count as items (or 4 minimum)
      const n = Math.max(4, items.length || 0);
      return Array.from({ length: n }).map((_, i) => (
        <div key={`sk-${i}`} className={`kpi-card ${dense ? "kpi-card--dense" : ""} is-skeleton`}>
          <div className="kpi__label skeleton-line" />
          <div className="kpi__value skeleton-block" />
          <div className="kpi__hint skeleton-line short" />
        </div>
      ));
    }

    return items.map((it, idx) => {
      const fmt = it.fmt ?? defaultFmt;
      const text = fmt(it.value);
      return (
        <div
          key={`${it.label}-${idx}`}
          className={[
            "kpi-card",
            dense ? "kpi-card--dense" : "",
            it.muted ? "is-muted" : "",
          ].join(" ").trim()}
          role="group"
          aria-label={it.label}
        >
          <div className="kpi__label">{it.label}</div>
          <div className="kpi__row">
            <div className="kpi__value">{text}</div>
            {it.delta && Number.isFinite(Number(it.delta.value)) && (
              <DeltaPill
                value={Number(it.delta.value)}
                suffix={it.delta.suffix}
                tone={it.delta.tone}
              />
            )}
          </div>
          {it.hint && <div className="kpi__hint">{it.hint}</div>}
        </div>
      );
    });
  }, [items, dense, loading]);

  return (
    <div className="kpi-bar-wrap">
      <div
        className={[
          "kpi-bar",
          scrollable ? "kpi-bar--scroll" : "",
          dense ? "kpi-bar--dense" : "",
          showError ? "kpi-bar--with-error" : "",
        ].join(" ").trim()}
        role="list"
        aria-label={ariaLabel}
      >
        {content}
      </div>

      {showError && (
        <div className="kpi-error" role="status" aria-live="polite">
          {errorText}
        </div>
      )}
    </div>
  );
}

/* ───────────── Helpers you can import where needed ───────────── */
export function fmtPct(v: number | string | null | undefined, digits = 1) {
  const num = Number(v);
  if (!Number.isFinite(num)) return "—";
  return `${num.toFixed(digits)}%`;
}
export function fmtInt(v: number | string | null | undefined) {
  const num = Number(v);
  if (!Number.isFinite(num)) return "—";
  return num.toLocaleString("fr-FR");
}
