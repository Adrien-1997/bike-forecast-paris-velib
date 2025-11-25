// ui/pages/_app.tsx
//
// -----------------------------------------------------------------------------
// Custom App (Next.js) — point d'entrée UI
//
// Rôle :
//   - Injecter les styles globaux (globals/header/footer).
//   - Brancher la font Urbanist via next/font et appliquer la variable CSS
//     directement sur <html> (urbanist.variable).
//   - Fournir un "layout" contextuel en fonction de la route :
//       * landing     → homepage / pages marketing,
//       * monitoring  → pages de monitoring (/monitoring/...),
//       * app         → application principale (/app/...).
//   - Charger dynamiquement les feuilles CSS spécifiques à chaque contexte
//     (landing.css, monitoring.css, app.css + bundles associés).
//   - Gérer le mode "no chrome" pour l’embarqué (iframe, /app/embed, query
//     ?embed=1 / ?nochrome=1) : retire header, footer, halo décoratif.
//   - Poser un shim Buffer côté browser pour les libs qui en ont besoin.
// -----------------------------------------------------------------------------
// NOTE IMPORTANTE :
//   Le code ci-dessous a été uniquement documenté (commentaires).
//   Aucune logique / signature n’a été modifiée.
// -----------------------------------------------------------------------------

import type { AppProps } from "next/app";
import { useEffect, useLayoutEffect, useMemo, useState } from "react";
import { useRouter } from "next/router";
import Head from "next/head";

// Global base styles (importés une seule fois ici)
import "@/styles/globals.css";
import "@/styles/header.css";
import "@/styles/footer.css";

// Font (next/font) → expose --font-urbanist
// La variable CSS est appliquée directement sur <html> plus bas.
import { Urbanist } from "next/font/google";
const urbanist = Urbanist({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800"],
  variable: "--font-urbanist",
  display: "swap",
});

// Browser Buffer shim
// Certaines libs s’attendent à trouver `Buffer` dans window (environnement Node).
// On ajoute un polyfill minimal côté client si absent.
import { Buffer } from "buffer";
if (typeof window !== "undefined" && !(window as any).Buffer) {
  (window as any).Buffer = Buffer;
}

// Common layout (non-landing)
import GlobalHeader from "@/components/layout/GlobalHeader";
import GlobalFooter from "@/components/layout/GlobalFooter";

// Contexte visuel / CSS de la page
type Ctx = "landing" | "monitoring" | "app";

export default function MyApp({ Component, pageProps }: AppProps) {
  const router = useRouter();

  /**
   * Déduit le contexte à partir du chemin de la route :
   *   - /monitoring/... → "monitoring"
   *   - /app/...        → "app"
   *   - sinon           → "landing"
   */
  const computeCtx = (path: string): Ctx =>
    path.startsWith("/monitoring")
      ? "monitoring"
      : path.startsWith("/app")
      ? "app"
      : "landing";

  // Init rapide pour éviter le "flash" de mauvais contexte au premier render.
  const [ctx, setCtx] = useState<Ctx>(() => computeCtx(router.pathname));

  // Applique la classe de variable Urbanist directement sur <html>.
  // useLayoutEffect côté client, useEffect côté SSR pour éviter les warnings.
  const useIsoLayoutEffect =
    typeof window !== "undefined" ? useLayoutEffect : useEffect;
  useIsoLayoutEffect(() => {
    const html =
      typeof document !== "undefined" ? document.documentElement : null;
    if (!html) return;
    html.classList.add(urbanist.variable);
    return () => {
      html.classList.remove(urbanist.variable);
    };
  }, []);

  // Suivi des navigations client → mise à jour du contexte (landing/app/monitoring).
  useEffect(() => {
    const updateCtx = (url: string) =>
      setCtx(computeCtx(url.split(/[?#]/)[0]));

    router.events.on("routeChangeStart", updateCtx);
    router.events.on("routeChangeComplete", updateCtx);
    return () => {
      router.events.off("routeChangeStart", updateCtx);
      router.events.off("routeChangeComplete", updateCtx);
    };
  }, [router.events]);

  // Chargement des feuilles de style contextuelles (landing.css / monitoring.css / app.css).
  // Les <link> sont marqués data-dynamic-style="true" pour pouvoir être purgés.
  useEffect(() => {
    // purge avant reload (évite l’accumulation de <link> lors des changements de contexte)
    document
      .querySelectorAll('link[data-dynamic-style="true"]')
      .forEach((el) => el.remove());

    const addCSS = (href: string) => {
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = href;
      link.dataset.dynamicStyle = "true";
      document.head.appendChild(link);
    };

    // CSS principal selon contexte
    addCSS(`/css/${ctx}.css`);

    // Bundles spécifiques (monitoring / app)
    if (ctx === "monitoring") {
      addCSS("/css/monitoringnav.css");
      addCSS("/css/kpibar.css");
      addCSS("/css/switchtoggle.css"); // ✅ nouvelle feuille pour SwitchBar
      addCSS("/css/loadingbar.css");
    } else if (ctx === "app") {
      addCSS("/css/mapview.css");
      addCSS("/css/loadingbar.css");
      addCSS("/css/switchtoggle.css"); // ✅ nouvelle feuille pour SwitchBar
    }

    // Nettoyage (au cas où le contexte change de nouveau)
    return () => {
      document
        .querySelectorAll('link[data-dynamic-style="true"]')
        .forEach((el) => el.remove());
    };
  }, [ctx]);

  // Header commun (hors landing) : items de navigation principaux.
  const sharedHeaderItems = useMemo(
    () => [
      { label: "Accueil", href: "/" },
      { label: "Carte", href: "/app" },
      { label: "Monitoring", href: "/monitoring" },
    ],
    []
  );

  // ─────────────────────────────────────────────────────────────
  // NO-CHROME: désactive header/footer/halo pour les pages embarquées
  // ─────────────────────────────────────────────────────────────
  // Cas couverts :
  //   - propriété statique Component.noChrome === true,
  //   - query ?embed=1 ou ?nochrome=1,
  //   - route dédiée /app/embed,
  //   - détection iframe (window.self !== window.top).
  const [inFrame, setInFrame] = useState(false);
  useEffect(() => {
    try {
      setInFrame(window.self !== window.top);
    } catch {
      // En sandbox cross-origin, on considère que c’est de l’embed.
      setInFrame(true);
    }
  }, []);

  const noChromeFlag = (Component as any)?.noChrome === true;
  const viaQuery =
    (router.query?.embed as string) === "1" ||
    (router.query?.nochrome as string) === "1";
  const isEmbedRoute = router.pathname === "/app/embed";

  // Flag final : si true, on ne rend ni halo, ni header, ni footer.
  const noChrome = noChromeFlag || viaQuery || isEmbedRoute || inFrame;

  return (
    <>
      {/* <Head> global pour l’app (viewport, theme-color, favicon, preconnect éventuel) */}
      <Head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta
          name="theme-color"
          content="#0b1220"
          media="(prefers-color-scheme: dark)"
        />
        <meta
          name="theme-color"
          content="#ffffff"
          media="(prefers-color-scheme: light)"
        />
        <link rel="icon" href="/favicon.svg" />
        {/* <link
          rel="preconnect"
          href="https://velib-ui-160046094975.europe-west1.run.app"
          crossOrigin=""
        /> */}
      </Head>

      {/* Classe racine = contexte (landing / app / monitoring),
          utilisée par les feuilles /css/*.css pour adapter la mise en page. */}
      <div className={ctx}>
        {/* halo décoratif global — désactivé sur landing ET en mode noChrome */}
        {ctx !== "landing" && !noChrome && (
          <div className="fx--page" aria-hidden="true" />
        )}

        {/* Header / Footer globaux hors landing, mais pas en noChrome */}
        {ctx !== "landing" && !noChrome && (
          <GlobalHeader items={sharedHeaderItems} />
        )}
        {/* Rendu de la page courante (Next.js) */}
        <Component {...pageProps} />
        {ctx !== "landing" && !noChrome && <GlobalFooter />}
      </div>
    </>
  );
}
