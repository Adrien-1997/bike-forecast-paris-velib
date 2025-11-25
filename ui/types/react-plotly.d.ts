// ui/types/react-plotly.d.ts
// -----------------------------------------------------------------------------
// Déclaration de module TypeScript pour "react-plotly.js".
//
// Rôle :
//   - Fournir un typage local pour le composant Plot de la librairie
//     "react-plotly.js", utilisé dans les pages de monitoring et l’UI.
//   - Éviter les erreurs TS lorsque le module n’exporte pas de types
//     officiels ou que ceux-ci ne sont pas installés dans le projet.
//
// Détails :
//   - Déclare le module "react-plotly.js" comme exposant un composant React
//     par défaut (`Plot`), typé avec l’interface PlotProps.
//   - PlotProps reprend les principaux props utilisés dans le projet :
//       • data : tableau de traces Plotly (Partial<Data>[]),
//       • layout : configuration de mise en page (Partial<Layout>),
//       • config : options Plotly (Partial<Config>),
//       • style / className : pour le styling React,
//       • useResizeHandler : activation de la gestion responsive,
//       • onInitialized / onUpdate : callbacks recevant data + layout.
//
// Usage :
//   - Import standard dans les composants :
//       import Plot from "react-plotly.js";
//   - Les types Layout / Config / Data proviennent de "plotly.js".
//
// Remarque :
//   - Ce fichier est uniquement une déclaration (.d.ts), il n’est pas
//     bundlé dans le JS final mais sert à la complétion & au lint TypeScript.
// -----------------------------------------------------------------------------

declare module "react-plotly.js" {
  import { ComponentType, CSSProperties } from "react";
  import type { Layout, Config, Data } from "plotly.js";

  export interface PlotProps {
    data: Partial<Data>[];
    layout?: Partial<Layout>;
    config?: Partial<Config>;
    style?: CSSProperties;
    className?: string;
    useResizeHandler?: boolean;
    onInitialized?: (figure: { data: Partial<Data>[]; layout: Partial<Layout> }) => void;
    onUpdate?: (figure: { data: Partial<Data>[]; layout: Partial<Layout> }) => void;
  }

  const Plot: ComponentType<PlotProps>;
  export default Plot;
}
