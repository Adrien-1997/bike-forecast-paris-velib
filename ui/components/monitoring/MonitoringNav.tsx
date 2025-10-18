// ui/components/monitoring/MonitoringNav.tsx
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

  const showBreadcrumbs = (crumbs?.length ?? 0) > 0 || !!generatedAt;

  return (
    <header className="mn-header">
      <div className="mn-page">
        {/* Breadcrumbs (only if we have crumbs or a timestamp) */}
        {showBreadcrumbs && (
          <nav className="breadcrumbs" aria-label="Breadcrumb">
            {crumbs.map((c, i) =>
              c.href ? (
                <Link key={i} href={c.href} className="crumb">
                  {c.label}
                </Link>
              ) : (
                <span key={i} className="crumb current" aria-current="page">
                  {c.label}
                </span>
              )
            )}
            {generatedAt && (
              <span className="right footer-note">
                Generated: {new Date(generatedAt).toLocaleString("en-GB")}
              </span>
            )}
          </nav>
        )}

        {/* Title */}
        <div className="title">
          <h1>{title}</h1>
          {subtitle && <span className="meta">{subtitle}</span>}
        </div>

        {/* The three grouped tabs */}
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
                  <span className={groupActive ? "btn btn--primary" : "btn btn--ghost nav-trigger"}>
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

          {/* Right-side actions (optional) */}
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
                  >
                    {a.label}
                  </Link>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </header>
  );
}