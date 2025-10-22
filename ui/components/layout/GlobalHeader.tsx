import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/router";
import { createPortal } from "react-dom";

export type HeaderItem = { label: string; href: string };

function isPathActive(path: string, href: string) {
  if (!href.startsWith("#")) return path === href || path.startsWith(href + "/");
  return false;
}

export default function GlobalHeader({
  items,
  brandHref = "/",
}: {
  items?: HeaderItem[];
  brandHref?: string;
}) {
  const { pathname } = useRouter();
  const [activeHash, setActiveHash] = useState<string>("");
  const [open, setOpen] = useState(false);
  const headerRef = useRef<HTMLElement | null>(null);
  const [hasDOM, setHasDOM] = useState(false); // évite le mismatch SSR des portals

  useEffect(() => setHasDOM(true), []);

  const list =
    items && items.length
      ? items
      : [
          { label: "Accueil", href: "/" },
          { label: "App", href: "/app" },
          { label: "Monitoring", href: "/monitoring" },
        ];

  const hashTargets = useMemo(
    () => list.filter((i) => i.href.startsWith("#")).map((i) => i.href.replace("#", "")),
    [list]
  );

  const isActive = (href: string) =>
    href.startsWith("#") ? activeHash === href : isPathActive(pathname, href);

  const closeMenu = () => setOpen(false);

  // Suivi des sections hash pour l'état actif
  useEffect(() => {
    if (hashTargets.length === 0 || typeof window === "undefined") return;
    const onHashChange = () => setActiveHash(window.location.hash || "");
    window.addEventListener("hashchange", onHashChange, { passive: true });
    onHashChange();

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort(
            (a, b) =>
              (a.target as HTMLElement).offsetTop - (b.target as HTMLElement).offsetTop
          )[0];
        if (visible) setActiveHash(`#${(visible.target as HTMLElement).id}`);
      },
      { rootMargin: "-30% 0px -60% 0px", threshold: [0, 1] }
    );

    hashTargets.forEach((id) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });

    return () => {
      window.removeEventListener("hashchange", onHashChange);
      observer.disconnect();
    };
  }, [hashTargets]);

  // Fermer le menu si on repasse desktop
  useEffect(() => {
    const onResize = () => {
      if (window.innerWidth >= 980) setOpen(false);
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // Lock scroll + classe utilitaire quand menu ouvert
  useEffect(() => {
    if (typeof document === "undefined") return;
    const body = document.body;
    const prev = body.style.overflow;
    body.style.overflow = open ? "hidden" : prev || "";
    body.classList.toggle("menu-open", open);
    return () => {
      body.style.overflow = prev || "";
      body.classList.remove("menu-open");
    };
  }, [open]);

  // Auto-hide du header au scroll
  useEffect(() => {
    const el = headerRef.current;
    if (!el) return;
    let prevY = window.scrollY;

    const onScroll = () => {
      const y = window.scrollY;
      const goingDown = y > prevY && y > 10;
      el.classList.toggle("is-hidden", goingDown && !open);
      prevY = y;
    };

    el.classList.add("autohide");
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", onScroll);
      el.classList.remove("autohide", "is-hidden");
    };
  }, [open]);

  // Échap pour fermer
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  // Fermer au clic extérieur (en plus du backdrop)
  useEffect(() => {
    if (!open) return;

    const onPointerDown = (e: PointerEvent) => {
      const headerEl =
        headerRef.current ?? document.querySelector<HTMLElement>(".site-header");
      if (!headerEl) return;
      const drawer = headerEl.querySelector(".mobile-drawer");
      const burger = headerEl.querySelector(".burger");
      if (
        (drawer && (drawer === e.target || drawer.contains(e.target as Node))) ||
        (burger && (burger === e.target || burger.contains(e.target as Node)))
      ) {
        return;
      }
      setOpen(false);
    };

    document.addEventListener("pointerdown", onPointerDown, true);
    return () => document.removeEventListener("pointerdown", onPointerDown, true);
  }, [open]);

  return (
    <>
      <header ref={headerRef} className="site-header">
        <div className="container nav">
          {/* Branding */}
          <Link href={brandHref} className="brand" aria-label="Accueil" onClick={closeMenu}>
            <img
              src="/favicon.svg"
              alt=""
              className="mark"
              width={40}
              height={40}
              decoding="async"
              loading="eager"
              draggable={false}
            />
            <span className="brandtext" aria-label="Vélo Paris">
              <span className="brand-velo">vélo</span>
              <span className="brand-sep">/</span>
              <span className="brand-paris">paris</span>
            </span>
          </Link>

          {/* Nav desktop */}
          <nav className="nav-desktop" aria-label="Navigation principale">
            <ul>
              {list.map((it) => (
                <li key={it.href}>
                  <Link
                    href={it.href}
                    className={isActive(it.href) ? "nav-trigger active" : "nav-trigger"}
                    aria-current={isActive(it.href) ? "page" : undefined}
                  >
                    {it.label}
                  </Link>
                </li>
              ))}
            </ul>
          </nav>

          {/* Burger (2 traits parallèles) */}
          <button
            className={open ? "burger close" : "burger"}
            aria-label={open ? "Fermer le menu" : "Ouvrir le menu"}
            aria-controls="mobile-menu"
            aria-expanded={open ? "true" : "false"}
            onClick={() => setOpen((v) => !v)}
          >
            <span className="icon" aria-hidden="true">
              <svg width="28" height="28" viewBox="0 0 24 24">
                <g className="lines">
                  <rect x="3" y="7" width="18" height="2" rx="1" />
                  <rect x="3" y="17" width="18" height="2" rx="1" />
                </g>
                <g className="cross">
                  <rect x="4" y="11" width="16" height="2" rx="1" transform="rotate(45 12 12)" />
                  <rect x="4" y="11" width="16" height="2" rx="1" transform="rotate(-45 12 12)" />
                </g>
              </svg>
            </span>
          </button>
        </div>

        {/* Mobile drawer */}
        <div
          id="mobile-menu"
          className={open ? "mobile-drawer open" : "mobile-drawer"}
          role="dialog"
          aria-modal="true"
        >
          <nav className="mobile-nav">
            <ul>
              {list.map((it) => (
                <li key={it.href}>
                  <Link
                    href={it.href}
                    className={isActive(it.href) ? "nav-trigger active" : "nav-trigger"}
                    aria-current={isActive(it.href) ? "page" : undefined}
                    onClick={closeMenu}
                  >
                    {it.label}
                  </Link>
                </li>
              ))}
            </ul>
          </nav>
        </div>
      </header>

      {/* Backdrop global porté au <body> (floute toute la page, pas le menu) */}
      {hasDOM &&
        createPortal(
          <button
            className={open ? "backdrop visible" : "backdrop"}
            aria-hidden={!open}
            tabIndex={-1}
            onClick={closeMenu}
          />,
          document.body
        )}
    </>
  );
}
