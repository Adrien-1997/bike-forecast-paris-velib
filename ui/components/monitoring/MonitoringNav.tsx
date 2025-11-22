// ui/components/monitoring/MonitoringNav.tsx

// =============================================================================
// Barre de navigation principale pour les pages de monitoring.
//
// Rôle :
// - Afficher le titre de la page de monitoring (title + sous-titre optionnel).
// - Afficher une méta "Généré le : ..." avec un état visuel (ok / warn / err)
//   en fonction de l'âge du snapshot (generatedAt).
// - Proposer une navigation par groupes : Réseau / Données / Modèle,
//   chacun contenant des liens vers les sous-pages de monitoring.
// - Gérer un petit bloc d’actions supplémentaires à droite (extraActions).
//
// Comportement :
// - Le lien actif est déterminé à partir de `pathname` (Next router).
// - Un groupe est actif si au moins un de ses liens est actif.
// - L’état "généré le" passe en :
//     • ok   si age ≤ 7 jours,
//     • warn si 7 < age ≤ 14 jours,
//     • err  si age > 14 jours.
// =============================================================================

import Link from "next/link";
import { useRouter } from "next/router";
import { useMemo } from "react";

type Action = { label: string; href: string };
type Group = { label: string; items: Action[] };

/**
 * GROUPS
 * ------
 * Structure fixe de la navigation Monitoring :
 * - Réseau      → overview / stations / dynamics
 * - Données     → drift / health
 * - Modèle      → performance / explainability
 *
 * Les labels et href sont consommés par la barre de navigation.
 */
const GROUPS: Group[] = [
  {
    label: "Réseau",
    items: [
      { label: "Aperçu", href: "/monitoring/network/overview" },
      { label: "Stations", href: "/monitoring/network/stations" },
      { label: "Dynamiques", href: "/monitoring/network/dynamics" },
    ],
  },
  {
    label: "Données",
    items: [
      { label: "Dérive", href: "/monitoring/data/drift" },
      { label: "Qualité", href: "/monitoring/data/health" },
    ],
  },
  {
    label: "Modèle",
    items: [
      { label: "Performance", href: "/monitoring/model/performance" },
      { label: "Explicabilité", href: "/monitoring/model/explainability" },
    ],
  },
];

/**
 * Un lien est actif si :
 * - le pathname est exactement égal à href, ou
 * - le pathname commence par href + "/" (ex: /monitoring/network/overview/details).
 */
function isActive(pathname: string, href: string) {
  return pathname === href || pathname.startsWith(href + "/");
}

/**
 * Un groupe est actif si au moins un de ses items est actif.
 */
function isGroupActive(pathname: string, group: Group) {
  return group.items.some((i) => isActive(pathname, i.href));
}

export default function MonitoringNav({
  title,
  subtitle,
  generatedAt,
  extraActions = [],
}: {
  /** Titre principal de la page (H1). */
  title: string;
  /** Sous-titre optionnel (texte sous le H1). */
  subtitle?: string;
  /**
   * Timestamp ISO de génération des artefacts utilisés par la page.
   * Sert à calculer l'age (en jours) et colorer le point (ok / warn / err).
   */
  generatedAt?: string | null;
  /** Actions supplémentaires à droite (boutons-lien). */
  extraActions?: Action[];
}) {
  const { pathname } = useRouter();

  /**
   * Informations dérivées de `generatedAt` :
   * - label human-readable en fr-FR,
   * - état visuel : ok / warn / err selon age en jours.
   *
   * Si `generatedAt` est absent ou invalide → null (rien n'est affiché).
   */
  const genInfo = useMemo(() => {
    if (!generatedAt) return null;
    const ts = new Date(generatedAt).getTime();
    if (Number.isNaN(ts)) return null;

    const ageMs = Date.now() - ts;
    const ageDays = ageMs / (1000 * 60 * 60 * 24);

    let state: "ok" | "warn" | "err" = "ok";
    if (ageDays > 14) state = "err";
    else if (ageDays > 7) state = "warn";

    const label = new Date(ts).toLocaleString("fr-FR");
    return { label, state };
  }, [generatedAt]);

  return (
    <header className="mn-header">
      {/* ───────────── Barre de titre ───────────── */}
      <div className="mn-titlebar">
        <div className="title">
          <h1>{title}</h1>
          {subtitle && <span className="meta">{subtitle}</span>}
          {genInfo && (
            <span className="mn-meta inline">
              <span
                className={`dot dot--${genInfo.state}`}
                aria-hidden="true"
              />
              Généré le : {genInfo.label}
            </span>
          )}
        </div>
      </div>

      {/* ───────────── Barre de navigation ───────────── */}
      <div
        className="toolbar-wrap"
        role="menubar"
        aria-label="Sections du monitoring"
      >
        <div className="groups">
          {GROUPS.map((group) => {
            const groupActive = isGroupActive(pathname, group);
            return (
              <div
                key={group.label}
                className={`nav-group ${groupActive ? "active" : ""}`}
                role="none"
              >
                {/* Bouton "tête" du groupe (style bouton, mais sans ouverture/fermeture
                    de dropdown au clic : le dropdown est toujours visible en desktop). */}
                <button
                  type="button"
                  className={
                    groupActive
                      ? "btn btn--primary nav-trigger"
                      : "btn btn--ghost nav-trigger"
                  }
                  aria-controls={`dropdown-${group.label}`}
                  aria-haspopup="true"
                  aria-expanded={undefined}
                  onMouseDown={(e) => e.preventDefault()} // évite de garder le focus "collé"
                >
                  {group.label}
                </button>

                {/* Liste des pages du groupe (dropdown stylé via CSS) */}
                <div
                  id={`dropdown-${group.label}`}
                  className="dropdown"
                  role="menu"
                  aria-label={`Pages de ${group.label}`}
                >
                  {group.items.map((item) => {
                    const active = isActive(pathname, item.href);
                    return (
                      <Link
                        key={item.href}
                        href={item.href}
                        className={active ? "dropdown-item active" : "dropdown-item"}
                        aria-current={active ? "page" : undefined}
                      >
                        {item.label}
                      </Link>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>

        {/* Actions supplémentaires (boutons alignés à droite de la barre) */}
        {extraActions.length > 0 && (
          <div className="extras">
            {extraActions.map((a) => {
              const active = isActive(pathname, a.href);
              return (
                <Link
                  key={a.href}
                  href={a.href}
                  className={active ? "btn btn--primary" : "btn btn--ghost"}
                  aria-current={active ? "page" : undefined}
                  onMouseDown={(e) => e.preventDefault()} // évite le focus persistant sur click
                >
                  {a.label}
                </Link>
              );
            })}
          </div>
        )}
      </div>
    </header>
  );
}
