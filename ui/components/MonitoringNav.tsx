// ui/components/MonitoringNav.tsx
import Link from "next/link";
import { useRouter } from "next/router";

type Crumb = { label: string; href?: string };
type Action = { label: string; href: string };

type Group = {
  label: string;
  items: Action[];
};

const GROUPS: Group[] = [
  {
    label: "Network",
    items: [
      { label: "Overview", href: "/monitoring/network/overview" },
      { label: "Stations", href: "/monitoring/network/stations" },
      { label: "Dynamics", href: "/monitoring/network/dynamics" },
    ],
  },
  {
    label: "Data",
    items: [
      { label: "Drift", href: "/monitoring/data/drift" },
      { label: "Health", href: "/monitoring/data/health" },
    ],
  },
  {
    label: "Model",
    items: [
      { label: "Performance", href: "/monitoring/model/performance" },
      { label: "Explainability", href: "/monitoring/model/explainability" },
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
  crumbs,
  extraActions = [],
}: {
  title: string;
  subtitle?: string;
  generatedAt?: string | null;
  crumbs: Crumb[];
  extraActions?: Action[];
}) {
  const { pathname } = useRouter();

  return (
    <header className="header">
      <div className="page">
        {/* Breadcrumbs */}
        <nav className="breadcrumbs">
          {crumbs.map((c, i) =>
            c.href ? (
              <Link key={i} href={c.href}>
                {c.label}
              </Link>
            ) : (
              <span key={i}>{c.label}</span>
            )
          )}
          <span className="right footer-note">
            {generatedAt ? `Généré : ${new Date(generatedAt).toLocaleString("fr-FR")}` : "—"}
          </span>
        </nav>

        {/* Title */}
        <div className="title">
          <h1>{title}</h1>
          {subtitle && <span className="meta">{subtitle}</span>}
        </div>

        {/* Tabs with grouped dropdowns */}
        <div className="toolbar toolbar-wrap" role="menubar" aria-label="Monitoring sections">
          <div className="groups">
            {GROUPS.map((group) => {
              const groupActive = isGroupActive(pathname, group);
              return (
                <div
                  key={group.label}
                  className={`nav-group ${groupActive ? "active" : ""}`}
                  role="menuitem"
                  tabIndex={0}
                  aria-haspopup="true"
                  aria-expanded={groupActive ? true : undefined}
                >
                  <span className={groupActive ? "btn btn-primary" : "btn btn-ghost nav-trigger"}>
                    {group.label}
                  </span>

                  <div className="dropdown" role="menu" aria-label={`${group.label} pages`}>
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

          {/* Extra actions (affichés à droite, sans menu) */}
          {extraActions.length > 0 && (
            <div className="extras">
              {extraActions.map((a) => {
                const active = isActive(pathname, a.href);
                return (
                  <Link
                    key={a.href}
                    href={a.href}
                    className={active ? "btn btn-primary" : "btn btn-ghost"}
                    aria-current={active ? "page" : undefined}
                  >
                    {a.label}
                  </Link>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Styles spécifiques au dropdown (peu intrusifs) */}
      <style jsx>{`
        .toolbar-wrap {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 12px;
          flex-wrap: wrap;
        }
        .groups {
          display: flex;
          gap: 8px;
          align-items: center;
          flex-wrap: wrap;
        }
        .extras {
          display: flex;
          gap: 8px;
          align-items: center;
        }
        .nav-group {
          position: relative;
        }
        .nav-trigger {
          cursor: default;
        }
        .dropdown {
          position: absolute;
          top: calc(100% + 6px);
          left: 0;
          min-width: 180px;
          padding: 6px;
          border-radius: 12px;
          background: var(--panel, #111);
          box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
          border: 1px solid var(--hairline, rgba(255, 255, 255, 0.08));
          display: none;
          z-index: 50;
        }
        .nav-group:hover .dropdown,
        .nav-group:focus-within .dropdown {
          display: grid;
          gap: 4px;
        }
        .dropdown-item {
          padding: 8px 10px;
          border-radius: 10px;
          text-decoration: none;
          display: block;
          white-space: nowrap;
          font-size: 0.95rem;
          opacity: 0.9;
        }
        .dropdown-item:hover {
          background: rgba(255, 255, 255, 0.06);
          opacity: 1;
        }
        .dropdown-item.active {
          background: var(--accentA, rgba(0, 153, 255, 0.18));
          outline: 1px solid var(--accent, #09f);
          opacity: 1;
        }

        /* Mobile: le menu s’ouvre au focus (tap) également */
        @media (hover: none) {
          .nav-trigger { cursor: pointer; }
        }
      `}</style>
    </header>
  );
}
