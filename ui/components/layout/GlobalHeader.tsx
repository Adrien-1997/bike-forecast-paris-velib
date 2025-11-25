// ui/components/layout/GlobalHeader.tsx
//
// =============================================================================
// Header global du site (marque "vélo/paris" + navigation principale).
//
// Rôle :
// - Affiche le branding (logo + texte) renvoyant vers l'accueil.
// - Gère la navigation principale (desktop + mobile drawer).
// - Met en évidence le lien actif (route ou ancre) :
//   • par path (pathname) pour les pages,
//   • par "scroll-spy" IntersectionObserver pour les sections (#ancres).
// - Gère un menu mobile avec : scroll-lock, backdrop, focus management, ESC,
//   clic extérieur, auto-hide du header au scroll, fermeture sur changement de route.
//
// Important :
// - Ne gère PAS l'état "active" dans la logique de scroll globale du site
//   (seulement pour le header).
// - Le style (autohide, backdrop, burger…) est entièrement géré via CSS.
// =============================================================================

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/router";
import { createPortal } from "react-dom";

export type HeaderItem = { label: string; href: string };

/**
 * Détermine si une route (href) doit être considérée comme "active"
 * pour un chemin donné :
 * - uniquement pour les liens de page (pas pour les ancres `#`),
 * - match exact ou préfixe (ex: /monitoring active pour /monitoring/overview).
 */
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

  // ─────────────────────────────
  // États & refs
  // ─────────────────────────────
  const [open, setOpen] = useState(false);     // état du menu mobile (drawer)
  const [hasDOM, setHasDOM] = useState(false); // true côté client (pour portal)
  const headerRef = useRef<HTMLElement | null>(null);

  const burgerRef = useRef<HTMLButtonElement | null>(null);   // bouton burger
  const firstLinkRef = useRef<HTMLAnchorElement | null>(null); // 1er lien du menu mobile (focus)

  const [activeHash, setActiveHash] = useState<string>("");   // ancre active (scroll-spy)

  // Indique que l'on est bien côté client (DOM disponible) → requerant pour createPortal
  useEffect(() => setHasDOM(true), []);

  // Liste des items, ou fallback par défaut si aucune prop `items` n'est fournie.
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

  // Liste des IDs ciblés par des ancres (#id) pour le scroll-spy
  const hashTargets = useMemo(
    () => list.filter((i) => i.href.startsWith("#")).map((i) => i.href.slice(1)),
    [list]
  );

  // Fonction générique pour savoir si un lien est actif (page ou ancre)
  const isActive = (href: string) =>
    href.startsWith("#") ? activeHash === href : isPathActive(pathname, href);

  // Ferme le menu mobile
  const closeMenu = () => setOpen(false);

  // ========================================================================
  // Scroll-spy via IntersectionObserver (gestion des ancres actives)
  // ========================================================================
  useEffect(() => {
    if (typeof window === "undefined" || hashTargets.length === 0) return;

    // Récupère la hauteur du header via la CSS custom property --header-h,
    // fallback à 60 px si non définie.
    const headerH = (() => {
      const doc = document.documentElement;
      const comp = getComputedStyle(doc);
      const raw = comp.getPropertyValue("--header-h").trim();
      const px = raw.endsWith("px") ? parseFloat(raw) : NaN;
      return Number.isFinite(px) ? px : 60;
    })();

    // Décalage vertical utilisé pour le calcul de la section "dominante"
    const topOffset = Math.ceil(window.innerHeight * 0.3) + headerH;

    const observer = new IntersectionObserver(
      (entries) => {
        // On ordonne les sections visibles par leur position vertical (top)
        const visibles = entries
          .filter((e) => e.isIntersecting)
          .sort(
            (a, b) =>
              (a.target as HTMLElement).getBoundingClientRect().top -
              (b.target as HTMLElement).getBoundingClientRect().top
          );

        // Si au moins une section est visible, on prend la plus haute dans la vue
        if (visibles.length > 0) {
          const id = (visibles[0].target as HTMLElement).id;
          if (id) setActiveHash("#" + id);
          return;
        }

        // Fallback : aucune section n'est intersectée → on choisit celle
        // dont le top est le plus proche du haut de la fenêtre, tout en étant
        // au-dessus du seuil `topOffset`.
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

    // On observe toutes les sections ciblées par des ancres
    hashTargets.forEach((id) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });

    // Au chargement, on synchronise l'état actif avec le hash de l'URL s'il existe
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

  // Fermer le menu mobile si on repasse en "desktop" (>= 980 px)
  useEffect(() => {
    const onResize = () => {
      if (window.innerWidth >= 980) setOpen(false);
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // ========================================================================
  // Scroll-lock du body quand le menu mobile est ouvert
  // ========================================================================
  useEffect(() => {
    if (typeof window === "undefined") return;
    const body = document.body;

    if (open) {
      // On fige le body (position: fixed) pour bloquer le scroll,
      // tout en mémorisant la position courante pour la restaurer à la fermeture.
      const scrollY = window.scrollY;
      body.dataset.prevScrollY = String(scrollY);
      body.style.position = "fixed";
      body.style.top = `-${scrollY}px`;
      body.style.left = "0";
      body.style.right = "0";
      body.style.width = "100%";
      body.classList.add("menu-open");

      // Focus initial sur le premier lien du menu mobile
      requestAnimationFrame(() => firstLinkRef.current?.focus());
    } else {
      // Restaure l'état du body et la position de scroll précédente
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

    // Cleanup pour éviter de laisser le body dans un état fixé si le composant unmount
    return () => {
      body.style.position = "";
      body.style.top = "";
      body.style.left = "";
      body.style.right = "";
      body.style.width = "";
      body.classList.remove("menu-open");
    };
  }, [open]);

  // ========================================================================
  // Auto-hide du header au scroll (caché en scroll down, affiché en scroll up)
  // ========================================================================
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
        // Si le menu est ouvert, le header reste visible
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

  // ESC pour fermer le menu mobile
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  // Clic extérieur (en "capturing") pour fermer le drawer mobile
  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: PointerEvent) => {
      const headerEl =
        headerRef.current ?? document.querySelector<HTMLElement>(".site-header");
      if (!headerEl) return;
      const drawer = headerEl.querySelector(".mobile-drawer");
      const burger = headerEl.querySelector(".burger");

      // Si clic dans le drawer ou sur le burger → on ne ferme pas
      if (
        (drawer && (drawer === e.target || drawer.contains(e.target as Node))) ||
        (burger && (burger === e.target || burger.contains(e.target as Node)))
      ) {
        return;
      }
      // Sinon → fermer le menu
      setOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown, true);
    return () => document.removeEventListener("pointerdown", onPointerDown, true);
  }, [open]);

  // Fermer le menu sur changement de route (navigation interne Next)
  useEffect(() => {
    const handleStart = () => setOpen(false);
    router.events.on("routeChangeStart", handleStart);
    return () => router.events.off("routeChangeStart", handleStart);
  }, [router.events]);

  /**
   * aria-current selon le type de lien :
   * - "page" pour les routes actives,
   * - "location" pour les ancres actives,
   * - undefined sinon.
   */
  const ariaCurrentFor = (href: string): "page" | "location" | undefined => {
    if (href.startsWith("#")) return isActive(href) ? "location" : undefined;
    return isActive(href) ? "page" : undefined;
  };

  return (
    <>
      <header ref={headerRef} className="site-header">
        <div className="container nav">
          {/* Branding (logo + texte "vélo/paris") */}
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

          {/* Navigation desktop */}
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

          {/* Bouton burger (mobile) */}
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

        {/* Drawer mobile (menu plein écran) */}
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

      {/* Backdrop global pour le menu mobile (portal vers document.body) */}
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
