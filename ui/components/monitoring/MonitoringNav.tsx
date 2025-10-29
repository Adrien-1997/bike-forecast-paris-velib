import Link from "next/link";
import { useRouter } from "next/router";
import { useMemo } from "react";

type Action = { label: string; href: string };
type Group = { label: string; items: Action[] };

const GROUPS: Group[] = [
  {
    label: "Réseau",
    items: [
      { label: "Aperçu", href: "/monitoring/network/overview" },
      { label: "Stations", href: "/monitoring/network/stations" },
      { label: "Dynamique", href: "/monitoring/network/dynamics" },
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

function isActive(pathname: string, href: string) {
  return pathname === href || pathname.startsWith(href + "/");
}
function isGroupActive(pathname: string, group: Group) {
  return group.items.some((i) => isActive(pathname, i.href));
}

export default function MonitoringNav({
  title,
  subtitle,
  generatedAt,
  extraActions = [],
}: {
  title: string;
  subtitle?: string;
  generatedAt?: string | null;
  extraActions?: Action[];
}) {
  const { pathname } = useRouter();

  const genInfo = useMemo(() => {
    if (!generatedAt) return null;
    const ts = new Date(generatedAt).getTime();
    if (Number.isNaN(ts)) return null;
    const ageMs = Date.now() - ts;
    const frais = ageMs <= 24 * 60 * 60 * 1000;
    const label = new Date(ts).toLocaleString("fr-FR");
    return { label, frais };
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
                className={`dot ${genInfo.frais ? "dot--ok" : "dot--stale"}`}
                aria-hidden="true"
              />
              Généré le : {genInfo.label}
            </span>
          )}
        </div>
      </div>

      {/* ───────────── Barre de navigation ───────────── */}
      <div className="toolbar-wrap" role="menubar" aria-label="Sections du monitoring">
        <div className="groups">
          {GROUPS.map((group) => {
            const groupActive = isGroupActive(pathname, group);
            return (
              <div
                key={group.label}
                className={`nav-group ${groupActive ? "active" : ""}`}
                role="none"
              >
                <button
                  type="button"
                  className={
                    groupActive
                      ? "btn btn--primary nav-trigger"
                      : "btn btn--ghost nav-trigger"
                  }
                  aria-controls={`dropdown-${group.label}`}
                  aria-haspopup="true"
                  aria-expanded={undefined} // ouverture gérée par CSS (pas de persistance)
                  onMouseDown={(e) => e.preventDefault()} // ❗ empêche le focus “collant” au clic
                >
                  {group.label}
                </button>

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

        {/* Actions supplémentaires (sans focus persistant) */}
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
                  onMouseDown={(e) => e.preventDefault()} // idem : pas de focus sticky
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
