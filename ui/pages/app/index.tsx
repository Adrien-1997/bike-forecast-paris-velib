// ui/pages/app/index.tsx
//
// =============================================================================
// Page "App" principale (avec chrome) pour Vélo Paris
// -----------------------------------------------------------------------------
// Cette page sert de **containeur** pour l’application Vélo Paris, en chargeant
// la version embarquée (`/app/embed`) dans un `<iframe>`, avec :
//   - le <Head> géré par Next.js (titre + meta description),
//   - le header / footer globaux fournis par la mise en page du site,
//   - un `<main>` qui occupe la quasi-totalité de la hauteur de la fenêtre,
//   - l’iframe interne qui charge la version sans chrome de l’app.
//
// Avantages de ce pattern :
//   - l’UI principale de l’app (carte, stations, prévisions, etc.) est isolée
//     dans `/app/embed` (layout simplifié, noChrome) ;
//   - cette page `/app` bénéficie du chrome global du site (navigation, footer),
//     tout en encapsulant l’app temps réel dans un iframe full-height.
// =============================================================================

import Head from "next/head";

/**
 * Page Next.js `/app`
 *
 * Rôle :
 *   - Définir le <Head> (titre + description SEO) pour l’app Vélo Paris.
 *   - Afficher un `<main>` occupant presque toute la hauteur du viewport.
 *   - Héberger un `<iframe>` pointant vers `/app/embed`, qui contient :
 *       • la carte Vélib’,
 *       • les prévisions de vélos,
 *       • les KPI/badges, etc.
 *
 * Remarque :
 *   - Le calcul `height: "calc(100vh - 120px)"` suppose que l’ensemble
 *     header + footer représente environ 120px de hauteur. Cette valeur
 *     peut être ajustée dans le layout global si besoin.
 */
export default function AppIndexPage() {
  return (
    <>
      {/* Métadonnées de la page /app (SEO / onglet navigateur) */}
      <Head>
        <title>Vélo Paris App — Disponibilités et prévisions</title>
        <meta
          name="description"
          content="Consultez les disponibilités et prévisions courtes des stations Vélib’ à Paris."
        />
      </Head>

      {/* Conteneur principal de l’app, sans padding latéral ni vertical.
          On laisse la place en hauteur au header/footer globaux. */}
      <main
        className="main app-wrapper"
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "stretch",
          padding: "0",
          height: "calc(100vh - 120px)", // ajuste selon la hauteur de ton header/footer
        }}
      >
        {/* Iframe interne :
            - src="/app/embed" charge la version "noChrome" de l’app Vélo Paris ;
            - flex: 1 → occupe tout l’espace vertical disponible dans <main> ;
            - border: "none" → pas de bordure autour de l’iframe ;
            - referrerPolicy="no-referrer" → ne transmet pas le referer complet. */}
        <iframe
          src="/app/embed"
          title="Vélo Paris App"
          style={{
            flex: 1,
            border: "none",
            width: "100%",
            height: "100%",
          }}
          loading="lazy"
          referrerPolicy="no-referrer"
        />
      </main>
    </>
  );
}