// ui/components/monitoring/KpiBar.tsx
//
// =============================================================================
// Barre de KPIs (cartes horizontales)
//
// Rôle :
// - Afficher une série de KPIs sous forme de cartes alignées horizontalement,
//   chacune avec : label, valeur principale, sous-texte (hint) et delta optionnel.
// - Gérer un état "loading" avec skeletons qui respectent la densité et le nombre
//   de items attendu (min. 4).
// - Afficher un éventuel message d’erreur inline à droite de la barre.
//
// Accessibilité :
// - La barre porte un `aria-label` global (paramétrable).
// - Chaque carte est un `role="group"` étiqueté par son `label`.
// - Le delta est étiqueté via `aria-label` (par ex. "delta +3%").
// =============================================================================

import { useMemo } from "react";

export type KpiItem = {
  /** Libellé du KPI (titre de la carte). */
  label: string;
  /** Valeur brute (nombre ou texte, éventuellement null/undefined). */
  value: number | string | null | undefined;
  /**
   * Formateur optionnel pour la valeur, par ex:
   *   (v) => `${Number(v).toFixed(1)}%`
   * Si non fourni → fallback sur `defaultFmt`.
   */
  fmt?: (v: number | string | null | undefined) => string;
  /** Sous-texte sous la valeur (fenêtre temporelle, unités, etc.). */
  hint?: string;
  /**
   * Delta optionnel (badge à droite de la valeur).
   * - value : valeur numérique de delta,
   * - suffix : texte accolé (%, pts, etc.),
   * - tone   : tonalité visuelle (couleur) : ok/warn/err/muted.
   */
  delta?: { value: number; suffix?: string; tone?: "ok" | "warn" | "err" | "muted" };
  /** Si true, applique un style visuel atténué à la carte. */
  muted?: boolean;
};

export type KpiBarProps = {
  /** Liste des KPIs à afficher. */
  items: KpiItem[];
  /** Mode compact (marges et tailles de police réduites). */
  dense?: boolean;
  /** Si true, affiche des skeletons à la place des valeurs (chargement). */
  loading?: boolean;
  /** Eventuel texte d’erreur affiché à droite de la barre. */
  errorText?: string | null;
  /** Libellé ARIA global pour la barre. */
  ariaLabel?: string;
  /** Si true, autorise le scroll horizontal sur petits écrans (défaut true). */
  scrollable?: boolean;
};

/**
 * Formatage par défaut :
 * - nombres → `toLocaleString("fr-FR")`,
 * - chaînes non vides → laissé tel quel,
 * - sinon → tiret "—".
 */
function defaultFmt(v: KpiItem["value"]) {
  if (typeof v === "number" && Number.isFinite(v)) return v.toLocaleString("fr-FR");
  if (typeof v === "string" && v.trim().length) return v;
  return "—";
}

/**
 * Petit "pill" de delta affiché à côté de la valeur.
 *
 * - Ajoute automatiquement le signe + / − selon la valeur,
 * - Affiche la valeur absolue suivie du suffix (%, pts…),
 * - Applique une classe de tonalité (ok / warn / err / muted).
 */
function DeltaPill({
  value,
  suffix,
  tone = "muted",
}: { value: number; suffix?: string; tone?: "ok" | "warn" | "err" | "muted" }) {
  const sign = value > 0 ? "+" : value < 0 ? "−" : "";
  const abs = Math.abs(value);
  return (
    <span
      className={`kpi-delta kpi-delta--${tone}`}
      aria-label={`delta ${sign}${abs}${suffix ?? ""}`}
    >
      {sign}
      {abs}
      {suffix ?? ""}
    </span>
  );
}

/**
 * KpiBar
 * ------
 * Conteneur principal des cartes de KPIs.
 *
 * - Si `loading === true` → rend des skeletons (même nombre de cartes que `items`
 *   ou 4 minimum).
 * - Sinon → rend chaque `KpiItem` avec son label, sa valeur formattée et hint.
 * - Si `errorText` est truthy → affiche un bandeau d’erreur à droite.
 */
export default function KpiBar({
  items,
  dense = false,
  loading = false,
  errorText = null,
  ariaLabel = "KPI bar",
  scrollable = true,
}: KpiBarProps) {
  const showError = Boolean(errorText);

  // Contenu principal : soit skeletons (loading), soit les cartes réelles.
  const content = useMemo(() => {
    if (loading) {
      // Skeletons : même count que items (ou 4 minimum pour la structure de la barre)
      const n = Math.max(4, items.length || 0);
      return Array.from({ length: n }).map((_, i) => (
        <div
          key={`sk-${i}`}
          className={`kpi-card ${dense ? "kpi-card--dense" : ""} is-skeleton`}
        >
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
          ]
            .join(" ")
            .trim()}
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
        ]
          .join(" ")
          .trim()}
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

/**
 * Formatteur pour pourcentage :
 * - convertit en nombre,
 * - retourne "—" si non fini,
 * - sinon format `X.Y%` avec `digits` décimales.
 */
export function fmtPct(v: number | string | null | undefined, digits = 1) {
  const num = Number(v);
  if (!Number.isFinite(num)) return "—";
  return `${num.toFixed(digits)}%`;
}

/**
 * Formatteur entier :
 * - toLocaleString("fr-FR") si valide,
 * - "—" sinon.
 */
export function fmtInt(v: number | string | null | undefined) {
  const num = Number(v);
  if (!Number.isFinite(num)) return "—";
  return num.toLocaleString("fr-FR");
}
