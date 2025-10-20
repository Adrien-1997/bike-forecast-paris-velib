// ui/components/monitoring/MonitoringNav.tsx
import Link from "next/link";
import { useRouter } from "next/router";
import { useState } from "react";

type Action = { label: string; href: string };
type Group = { label: string; items: Action[] };

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
  extraActions = [],
}: {
  title: string;
  subtitle?: string;
  generatedAt?: string | null;
  extraActions?: Action[];
}) {
  const { pathname } = useRouter();
  const [openGroup, setOpenGroup] = useState<string | null>(null);

  const handleEnter = (label: string) => setOpenGroup(label);
  const handleLeave = () => setOpenGroup(null);

  return (
    <header className="mn-header">
      {/* ───────────── Title & generated time ───────────── */}
      <div className="mn-titlebar">
        <div className="title">
          <h1>{title}</h1>
          {subtitle && <span className="meta">{subtitle}</span>}
        </div>
        {generatedAt && (
          <div className="mn-meta">
            <span className="generated-at">
              <span className="dot" aria-hidden="true"></span>
              Généré : {new Date(generatedAt).toLocaleString("fr-FR")}
            </span>
          </div>
        )}
      </div>

      {/* ───────────── Toolbar ───────────── */}
      <div
        className="toolbar-wrap"
        role="menubar"
        aria-label="Monitoring sections"
      >
        <div className="groups">
          {GROUPS.map((group) => {
            const groupActive = isGroupActive(pathname, group);
            const isOpen = openGroup === group.label;

            return (
              <div
                key={group.label}
                className={`nav-group ${groupActive ? "active" : ""} ${
                  isOpen ? "open" : ""
                }`}
                role="menuitem"
                tabIndex={0}
                aria-haspopup="true"
                aria-expanded={isOpen || groupActive ? true : undefined}
                onMouseEnter={() => handleEnter(group.label)}
                onMouseLeave={handleLeave}
              >
                <button
                  type="button"
                  className={
                    groupActive
                      ? "btn btn--primary nav-trigger"
                      : "btn btn--ghost nav-trigger"
                  }
                  aria-controls={`dropdown-${group.label}`}
                >
                  {group.label}
                </button>

                <div
                  id={`dropdown-${group.label}`}
                  className="dropdown"
                  role="menu"
                  aria-label={`${group.label} pages`}
                >
                  {group.items.map((item) => {
                    const active = isActive(pathname, item.href);
                    return (
                      <Link
                        key={item.href}
                        href={item.href}
                        className={
                          active
                            ? "dropdown-item active"
                            : "dropdown-item"
                        }
                        aria-current={active ? "page" : undefined}
                        onClick={() => setOpenGroup(null)}
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

        {/* ───────────── Extra actions (restaurées) ───────────── */}
        {extraActions.length > 0 && (
          <div className="extras">
            {extraActions.map((a) => {
              const active = isActive(pathname, a.href);
              return (
                <Link
                  key={a.href}
                  href={a.href}
                  className={
                    active ? "btn btn--primary" : "btn btn--ghost"
                  }
                  aria-current={active ? "page" : undefined}
                  style={{ textDecoration: "none", fontWeight: 600 }}
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
