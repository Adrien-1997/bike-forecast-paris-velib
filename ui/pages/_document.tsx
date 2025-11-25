// ui/pages/_document.tsx
// -----------------------------------------------------------------------------
// Custom Document (Next.js)
// -----------------------------------------------------------------------------
// Rôle de ce fichier :
//   - Surcharger le Document par défaut de Next.js pour contrôler la structure
//     HTML racine de l’application.
//   - Poser <Html>, <Head>, <body> globaux, indépendamment des pages.
//   - Injecter ici les styles globaux nécessaires côté "document", par ex.
//     la feuille de style Leaflet utilisée par les cartes (react-leaflet).
//
// IMPORTANT :
//   - Le code de rendu est strictement identique, seules des annotations ont
//     été ajoutées pour documentation.
//   - Aucune logique / signature / JSX n’a été modifiée.
// -----------------------------------------------------------------------------

import Document, { Html, Head, Main, NextScript } from "next/document";

export default class MyDocument extends Document {
  render() {
    return (
      // Langue principale du document (SEO, accessibilité, spellcheck, etc.)
      <Html lang="fr">
        <Head>
          {/* 
            Feuille de styles Leaflet globale

            - chargée ici une seule fois pour toute l’app.
            - utilisée par react-leaflet dans les pages de monitoring et /app.
            - l’attribut integrity garantit l’intégrité (Subresource Integrity).
            - crossOrigin="" est requis par Leaflet pour certaines configurations.
          */}
          <link
            rel="stylesheet"
            href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
            integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
            crossOrigin=""
          />
        </Head>
        <body>
          {/* 
            <Main /> : point d’ancrage principal des pages Next.js
            (le contenu des pages est rendu ici).
          */}
          <Main />
          {/* 
            <NextScript /> : scripts injectés par Next (hydration, bundle JS, etc.)
          */}
          <NextScript />
        </body>
      </Html>
    );
  }
}
