// ui/pages/_app.tsx
import type { AppProps } from "next/app";
import { useEffect, useLayoutEffect, useMemo, useState } from "react";
import { useRouter } from "next/router";
import Head from "next/head";
import "@/styles/globals.css";

// ✅ header/footer en CSS global
import "@/styles/header.css";
import "@/styles/footer.css";

// ✅ next/font — Urbanist
import { Urbanist } from "next/font/google";
const urbanist = Urbanist({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800"],
  variable: "--font-urbanist",
  display: "swap",
});

// ✅ Buffer shim (browser)
import { Buffer } from "buffer";
if (typeof window !== "undefined" && !(window as any).Buffer) {
  (window as any).Buffer = Buffer;
}

// ✅ Header & Footer communs (hors landing)
import GlobalHeader from "@/components/layout/GlobalHeader";
import GlobalFooter from "@/components/layout/GlobalFooter";

export default function MyApp({ Component, pageProps }: AppProps) {
  const router = useRouter();

  // Helper : déduit le contexte selon le pathname
  const computeCtx = (path: string): "landing" | "monitoring" | "app" => {
    if (path.startsWith("/monitoring")) return "monitoring";
    if (path.startsWith("/app")) return "app";
    return "landing";
  };

  // ⚡ init immédiate pour éviter un flash de styles
  const [ctx, setCtx] = useState<"landing" | "monitoring" | "app">(
    () => computeCtx(router.pathname)
  );

  const useIsoLayoutEffect =
    typeof window !== "undefined" ? useLayoutEffect : useEffect;

  // ✅ applique la variable de police Urbanist sur <html>
  useIsoLayoutEffect(() => {
    if (typeof document === "undefined") return;
    const html = document.documentElement;
    html.classList.add(urbanist.variable);
    return () => html.classList.remove(urbanist.variable);
  }, []);

  // ✅ met à jour le contexte lors des navigations client
  useEffect(() => {
    const onRoute = (url: string) => {
      const pathname = url.split("?")[0].split("#")[0];
      setCtx(computeCtx(pathname));
    };
    router.events.on("routeChangeComplete", onRoute);
    return () => router.events.off("routeChangeComplete", onRoute);
  }, [router.events]);

  // ✅ charge dynamiquement la feuille de style contextuelle
  useEffect(() => {
    // Nettoie les anciens liens dynamiques
    document
      .querySelectorAll('link[data-dynamic-style="true"]')
      .forEach((el) => el.remove());

    // charge le CSS contextuel (landing / monitoring / app)
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = `/css/${ctx}.css`;
    link.dataset.dynamicStyle = "true";
    document.head.appendChild(link);

    // ➕ MonitoringNav + KpiBar + LoadingBar CSS (uniquement en contexte monitoring)
    if (ctx === "monitoring") {
      const mnLink = document.createElement("link");
      mnLink.rel = "stylesheet";
      mnLink.href = `/css/monitoringnav.css`;
      mnLink.dataset.dynamicStyle = "true";
      document.head.appendChild(mnLink);

      const kpiBarLink = document.createElement("link");
      kpiBarLink.rel = "stylesheet";
      kpiBarLink.href = `/css/kpibar.css`;
      kpiBarLink.dataset.dynamicStyle = "true";
      document.head.appendChild(kpiBarLink);

      const lbLink = document.createElement("link");
      lbLink.rel = "stylesheet";
      lbLink.href = `/css/loadingbar.css`;
      lbLink.dataset.dynamicStyle = "true";
      document.head.appendChild(lbLink);
    }

    // ➕ Map view + LoadingBar CSS si contexte "app"
    if (ctx === "app") {
      const mapLink = document.createElement("link");
      mapLink.rel = "stylesheet";
      mapLink.href = `/css/mapview.css`;
      mapLink.dataset.dynamicStyle = "true";
      document.head.appendChild(mapLink);

      const lbLink = document.createElement("link");
      lbLink.rel = "stylesheet";
      lbLink.href = `/css/loadingbar.css`;
      lbLink.dataset.dynamicStyle = "true";
      document.head.appendChild(lbLink);
    }

    return () => {
      document
        .querySelectorAll('link[data-dynamic-style="true"]')
        .forEach((el) => el.remove());
    };
  }, [ctx]);

  // ✅ items du header commun (toutes pages ≠ landing)
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
      </Head>

      <div className={ctx}>
        {/* halo décoratif global */}
        <div className="fx--page" aria-hidden="true" />

        {/* Header commun à toutes les pages sauf la landing */}
        {ctx !== "landing" && <GlobalHeader items={sharedHeaderItems} />}

        {/* Contenu principal */}
        <Component {...pageProps} />

        {/* Footer commun à toutes les pages sauf la landing */}
        {ctx !== "landing" && <GlobalFooter />}
      </div>
    </>
  );
}
