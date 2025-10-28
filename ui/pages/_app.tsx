// ui/pages/_app.tsx
import type { AppProps } from "next/app";
import { useEffect, useLayoutEffect, useMemo, useState } from "react";
import { useRouter } from "next/router";
import Head from "next/head";

// Global base styles
import "@/styles/globals.css";
import "@/styles/header.css";
import "@/styles/footer.css";

// Font (next/font) → expose --font-urbanist
import { Urbanist } from "next/font/google";
const urbanist = Urbanist({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800"],
  variable: "--font-urbanist",
  display: "swap",
});

// Browser Buffer shim
import { Buffer } from "buffer";
if (typeof window !== "undefined" && !(window as any).Buffer) {
  (window as any).Buffer = Buffer;
}

// Common layout (non-landing)
import GlobalHeader from "@/components/layout/GlobalHeader";
import GlobalFooter from "@/components/layout/GlobalFooter";

type Ctx = "landing" | "monitoring" | "app";

export default function MyApp({ Component, pageProps }: AppProps) {
  const router = useRouter();

  const computeCtx = (path: string): Ctx =>
    path.startsWith("/monitoring")
      ? "monitoring"
      : path.startsWith("/app")
      ? "app"
      : "landing";

  // Init rapide pour éviter le flash mauvais contexte
  const [ctx, setCtx] = useState<Ctx>(() => computeCtx(router.pathname));

  // Applique la classe de variable Urbanist directement sur <html>
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

  // Suivi des navigations client → mise à jour du contexte
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

  // Chargement des feuilles de style contextuelles
  useEffect(() => {
    // purge avant reload
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

    // Bundles spécifiques
    if (ctx === "monitoring") {
      addCSS("/css/monitoringnav.css");
      addCSS("/css/kpibar.css");
      addCSS("/css/switchtoggle.css"); // ✅ nouvelle feuille pour SwitchBar
      addCSS("/css/loadingbar.css");
    } else if (ctx === "app") {
      addCSS("/css/mapview.css");
      addCSS("/css/loadingbar.css");
    }

    return () => {
      document
        .querySelectorAll('link[data-dynamic-style="true"]')
        .forEach((el) => el.remove());
    };
  }, [ctx]);

  // Header commun (hors landing)
  const sharedHeaderItems = useMemo(
    () => [
      { label: "Accueil", href: "/" },
      { label: "Carte", href: "/app" },
      { label: "Monitoring", href: "/monitoring" },
    ],
    []
  );

  return (
    <>
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

      <div className={ctx}>
        {/* halo décoratif global — désactivé sur la landing pour laisser le fond défiler */}
        {ctx !== "landing" && <div className="fx--page" aria-hidden="true" />}

        {/* Header / Footer globaux hors landing */}
        {ctx !== "landing" && <GlobalHeader items={sharedHeaderItems} />}
        <Component {...pageProps} />
        {ctx !== "landing" && <GlobalFooter />}
      </div>
    </>
  );
}
