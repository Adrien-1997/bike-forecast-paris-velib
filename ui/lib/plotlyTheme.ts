// ui/lib/plotlyTheme.ts
//
// =============================================================================
// Thème Plotly centralisé pour l’app Vélib’ Forecast.
//
// Rôle :
// - Fournir une configuration Plotly cohérente (theme global) pour tous les
//   graphiques de monitoring et de l’app.
// - Exposer :
//     • `chartConfig` : options d’interaction / rendu (mode bar, responsive…)
//     • `chartLayout` : layout de base (marges, couleurs, axes…) surchargeable
//       via un objet `overrides`.
//
// Contraintes :
// - Aucune logique métier ici : uniquement du styling / ergonomie.
// - Le thème s’appuie sur les variables CSS globales (`--mn-font`, `--text`,
//   `--text-dim`, etc.) afin de rester aligné avec le design system du site.
// - `chartLayout(overrides)` doit rester pur et facilement sérialisable.
// =============================================================================

import type * as Plotly from "plotly.js";

/* ──────────────────────────── CONFIG ──────────────────────────── */
/**
 * Configuration Plotly générique appliquée à tous les graphiques.
 *
 * - `displayModeBar: false` : masque la barre d’outils flottante
 *   (zoom, save, etc.) pour garder une UI épurée.
 * - `responsive: true` : force le redimensionnement automatique
 *   des graphiques (utile pour les layouts flex / grid).
 * - `scrollZoom: false` : désactive le zoom à la molette (evite les
 *   zooms accidentels lors du scroll de page).
 */
export const chartConfig: Partial<Plotly.Config> = {
  displayModeBar: false,
  responsive: true,
  scrollZoom: false,
};

/* ──────────────────────────── LAYOUT ──────────────────────────── */
/**
 * Layout de base Plotly pour l’ensemble des graphiques.
 *
 * Usage :
 *   const layout = chartLayout({
 *     title: { text: "My chart" },
 *     yaxis: { rangemode: "tozero" },
 *   });
 *
 * - `overrides` permet de surcharger n’importe quelle propriété du layout
 *   (par horizon, par page, etc.) tout en conservant un thème global.
 */
export function chartLayout(
  overrides: Partial<Plotly.Layout> = {}
): Partial<Plotly.Layout> {
  const base: Partial<Plotly.Layout> = {
    autosize: true,

    // Marges plus généreuses (gauche/haut/bas) pour éviter le clipping
    // des labels et des titres d’axes.
    margin: { l: 52, r: 12, t: 34, b: 48 },

    // Fond transparent : laisse apparaître le fond de la page (glass / dark).
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",

    // Mode hover unifié sur l’axe X (bulle unique).
    hovermode: "x unified",

    // Police alignée sur le design global (définie en CSS).
    font: {
      family: "var(--mn-font)",
      size: 12,
      color: "var(--text)",
    },

    // Palette de couleurs héritée de CSS si souhaité (sinon palette Plotly).
    // On peut enrichir ici si besoin :
    // colorway: ["var(--mn-blue)", "var(--mn-orange)", "var(--mn-green)"],

    // Légende compacte en haut, horizontale.
    legend: {
      orientation: "h",
      yanchor: "bottom",
      y: 1.02,
      xanchor: "left",
      x: 0,
      font: {
        size: 11,
        color: "var(--text-dim)",
      },
    },

    // Axe X : ticks externes, grille discrète, texte dim.
    xaxis: {
      title: {
        font: { size: 12, color: "var(--text-dim)" },
        standoff: 12,
      },
      tickfont: { color: "var(--text-dim)" },
      gridcolor: "rgba(255,255,255,.08)",
      zeroline: false,
      automargin: true,
      ticks: "outside",
      ticklen: 6,
    },

    // Axe Y — mêmes principes que l’axe X.
    yaxis: {
      title: {
        font: { size: 12, color: "var(--text-dim)" },
        standoff: 8,
      },
      tickfont: { color: "var(--text-dim)" },
      gridcolor: "rgba(255,255,255,.08)",
      zeroline: false,
      rangemode: "tozero",
      automargin: true,
      ticks: "outside",
      ticklen: 6,
    },
  };

  // Surcharge finale : permet à chaque graphique de customiser le layout
  // tout en réutilisant le thème par défaut.
  return { ...base, ...overrides };
}
