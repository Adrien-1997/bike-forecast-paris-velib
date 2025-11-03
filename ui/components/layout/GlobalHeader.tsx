// ui/components/layout/GlobalHeader.tsx
import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/router";
import { createPortal } from "react-dom";

export type HeaderItem = { label: string; href: string };

/** Active pour routes (pages), pas pour ancres. */
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
  const router = useRouter();
  const { pathname } = router;

  // ── États
  const [open, setOpen] = useState(false);
  const [hasDOM, setHasDOM] = useState(false);
  const headerRef = useRef<HTMLElement | null>(null);

  const burgerRef = useRef<HTMLButtonElement | null>(null);
  const firstLinkRef = useRef<HTMLAnchorElement | null>(null);

  const [activeHash, setActiveHash] = useState<string>("");

  useEffect(() => setHasDOM(true), []);

  const list = useMemo<HeaderItem[]>(
    () =>
      items && items.length
        ? items
        : [
            { label: "Accueil", href: "/" },
            { label: "App", href: "/app" },
            { label: "Monitoring", href: "/monitoring" },
          ],
    [items]
  );

  const hashTargets = useMemo(
    () => list.filter((i) => i.href.startsWith("#")).map((i) => i.href.slice(1)),
    [list]
  );

  const isActive = (href: string) =>
    href.startsWith("#") ? activeHash === href : isPathActive(pathname, href);

  const closeMenu = () => setOpen(false);

  // ===== Scroll-spy via IntersectionObserver =====
  useEffect(() => {
    if (typeof window === "undefined" || hashTargets.length === 0) return;

    const headerH = (() => {
      const doc = document.documentElement;
      const comp = getComputedStyle(doc);
      const raw = comp.getPropertyValue("--header-h").trim();
      const px = raw.endsWith("px") ? parseFloat(raw) : NaN;
      return Number.isFinite(px) ? px : 60;
    })();

    const topOffset = Math.ceil(window.innerHeight * 0.3) + headerH;

    const observer = new IntersectionObserver(
      (entries) => {
        const visibles = entries
          .filter((e) => e.isIntersecting)
          .sort(
            (a, b) =>
              (a.target as HTMLElement).getBoundingClientRect().top -
              (b.target as HTMLElement).getBoundingClientRect().top
          );

        if (visibles.length > 0) {
          const id = (visibles[0].target as HTMLElement).id;
          if (id) setActiveHash("#" + id);
          return;
        }

        let bestId = "";
        let bestTop = -Infinity;
        for (const id of hashTargets) {
          const el = document.getElementById(id);
          if (!el) continue;
          const top = el.getBoundingClientRect().top;
          if (top <= topOffset && top > bestTop) {
            bestTop = top;
            bestId = id;
          }
        }
        if (bestId) setActiveHash("#" + bestId);
      },
      {
        root: null,
        rootMargin: `-${topOffset}px 0px -60% 0px`,
        threshold: [0, 0.25, 0.5, 1],
      }
    );

    hashTargets.forEach((id) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });

    const setFromUrl = () => {
      const h = window.location.hash;
      if (h && hashTargets.includes(h.slice(1))) setActiveHash(h);
    };
    setFromUrl();

    const onHash = () => setFromUrl();
    window.addEventListener("hashchange", onHash, { passive: true });

    return () => {
      window.removeEventListener("hashchange", onHash);
      observer.disconnect();
    };
  }, [hashTargets]);

  // Fermer menu si resize desktop
  useEffect(() => {
    const onResize = () => {
      if (window.innerWidth >= 980) setOpen(false);
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // Scroll-lock body
  useEffect(() => {
    if (typeof window === "undefined") return;
    const body = document.body;

    if (open) {
      const scrollY = window.scrollY;
      body.dataset.prevScrollY = String(scrollY);
      body.style.position = "fixed";
      body.style.top = `-${scrollY}px`;
      body.style.left = "0";
      body.style.right = "0";
      body.style.width = "100%";
      body.classList.add("menu-open");
      requestAnimationFrame(() => firstLinkRef.current?.focus());
    } else {
      const prev = body.dataset.prevScrollY;
      body.style.position = "";
      body.style.top = "";
      body.style.left = "";
      body.style.right = "";
      body.style.width = "";
      body.classList.remove("menu-open");
      if (prev) window.scrollTo(0, parseInt(prev, 10));
      burgerRef.current?.focus();
    }

    return () => {
      body.style.position = "";
      body.style.top = "";
      body.style.left = "";
      body.style.right = "";
      body.style.width = "";
      body.classList.remove("menu-open");
    };
  }, [open]);

  // Auto-hide header
  useEffect(() => {
    const el = headerRef.current;
    if (!el) return;
    let prevY = window.scrollY;

    const onScroll = () => {
      const y = window.scrollY;
      if (!open) {
        const goingDown = y > prevY && y > 10;
        el.classList.toggle("is-hidden", goingDown);
      } else {
        el.classList.remove("is-hidden");
      }
      prevY = y;
    };

    el.classList.add("autohide");
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", onScroll);
      el.classList.remove("autohide", "is-hidden");
    };
  }, [open]);

  // ESC pour fermer
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  // Clic extérieur
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

  // Fermer navigation interne
  useEffect(() => {
    const handleStart = () => setOpen(false);
    router.events.on("routeChangeStart", handleStart);
    return () => router.events.off("routeChangeStart", handleStart);
  }, [router.events]);

  const ariaCurrentFor = (href: string): "page" | "location" | undefined => {
    if (href.startsWith("#")) return isActive(href) ? "location" : undefined;
    return isActive(href) ? "page" : undefined;
  };

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
                    aria-current={ariaCurrentFor(it.href)}
                    onClick={() => {
                      if (it.href.startsWith("#")) setActiveHash(it.href);
                    }}
                  >
                    {it.label}
                  </Link>
                </li>
              ))}
            </ul>
          </nav>

          {/* Burger */}
          <button
            ref={burgerRef}
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
                  <rect
                    x="4"
                    y="11"
                    width="16"
                    height="2"
                    rx="1"
                    transform="rotate(45 12 12)"
                  />
                  <rect
                    x="4"
                    y="11"
                    width="16"
                    height="2"
                    rx="1"
                    transform="rotate(-45 12 12)"
                  />
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
              {list.map((it, idx) => (
                <li key={it.href}>
                  <Link
                    href={it.href}
                    className={isActive(it.href) ? "nav-trigger active" : "nav-trigger"}
                    aria-current={ariaCurrentFor(it.href)}
                    onClick={() => {
                      if (it.href.startsWith("#")) setActiveHash(it.href);
                      closeMenu();
                    }}
                    ref={idx === 0 ? firstLinkRef : undefined}
                  >
                    {it.label}
                  </Link>
                </li>
              ))}
            </ul>
          </nav>
        </div>
      </header>

      {/* Backdrop global */}
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
