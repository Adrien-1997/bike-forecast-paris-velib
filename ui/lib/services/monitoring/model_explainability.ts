// ui/lib/services/monitoring/model_explainability.ts
//
// =============================================================================
// Service front pour /monitoring/model/explainability.
//
// Rôle :
// - Fournir des fonctions typées pour appeler les endpoints de monitoring
//   "Model Explainability" côté API (résidus, calibration, incertitude,
//   importance des features…).
// - Centraliser les types utilisés par l’UI pour ces écrans, en cohérence
//   avec les artefacts JSON produits côté backend.
//
// Endpoints consommés :
//   GET /monitoring/model/explainability/overview?h=H
//   GET /monitoring/model/explainability/residuals?h=H
//   GET /monitoring/model/explainability/calibration?h=H
//   GET /monitoring/model/explainability/uncertainty?h=H
//   GET /monitoring/model/explainability/feature_importance?h=H
//
// Convention :
// - Tous les appels passent par `fetchJsonWithEtag`, qui gère déjà :
//     • la base URL,
//     • le token éventuel,
//     • les en-têtes ETag / If-None-Match.
// - Chaque doc JSON est versionné via `schema_version` et timestampé avec
//   `generated_at`.
// =============================================================================

import { fetchJsonWithEtag } from "@/lib/http";

/* ── Types : vue d’ensemble ─────────────────────────────────────────────── */

/**
 * Vue d’ensemble de la couche "explainability" pour un horizon donné.
 *
 * Sert principalement à :
 * - contextualiser les autres artefacts (résidus, calibration…),
 * - afficher des métadonnées sur la fenêtre de performance utilisée.
 */
export type Overview = {
  /** Version de schéma JSON. */
  schema_version: string;
  /** Timestamp de génération (UTC, ISO). */
  generated_at: string;
  /** Timezone logique utilisée pour l’affichage (ex: "Europe/Paris"). */
  tz: string;

  /** Jour d’ancrage de la fenêtre de performance (ex: "2025-11-21"). */
  anchor_day_perf: string | null;
  /** Nombre total de lignes de perf dans la fenêtre. */
  perf_rows: number;
  /** Nombre de stations couvertes dans la fenêtre. */
  perf_stations: number;
  /** Premier timestamp de perf (UTC, ISO). */
  ts_min_perf: string | null;
  /** Dernier timestamp de perf (UTC, ISO). */
  ts_max_perf: string | null;

  /** true si y_pred est présent dans les exports de perf. */
  has_y_pred: boolean;
  /** true si des intervalles d’incertitude sont disponibles. */
  has_uncertainty: boolean;

  // Backend ≥ 1.3
  /** Horizon en minutes (ex: 15, 60…). */
  horizon_min?: number;
  /** Taille de la fenêtre de perf (en jours) si connue. */
  window_days?: number | null;
};

/* ── Types : Résidus ────────────────────────────────────────────────────── */

/**
 * Un bin d’histogramme de résidus.
 */
export type ResidHistBin = {
  /** Borne gauche du bin (en unités de résidu). */
  bin_left: number;
  /** Borne droite du bin. */
  bin_right: number;
  /** Nombre d’observations dans ce bin. */
  count: number;
};

/**
 * Document de résidus pour un horizon donné :
 * - histogramme (hist),
 * - QQ plot (qq: théorie vs empirique),
 * - ACF des résidus,
 * - hétéroscédasticité par quantile,
 * - épisodes de résidus consécutifs par station.
 */
export type ResidualsDoc = {
  /** Version de schéma JSON. */
  schema_version: string;
  /** Timestamp de génération (UTC, ISO). */
  generated_at: string;

  /** Histogramme des résidus. */
  hist: ResidHistBin[];

  /** QQ-plot : th = quantiles théoriques, emp = quantiles observés. */
  qq: { th: number[]; emp: number[] };

  /** Autocorrélation des résidus (lags successifs). */
  acf: number[];

  /** Hétéroscédasticité par quantile de ŷ / y. */
  hetero: Array<{ quantile: string; mae: number; n: number }>;

  /**
   * Épisodes de résidus longs par station (runs de même signe, etc.).
   * Permet de repérer des "patterns" station-spécifiques.
   */
  episodes: Array<{ station_id: string; max_run: number; n: number }>;

  // Backend ≥ 1.3
  /** Horizon en minutes. */
  horizon_min?: number;
  /** Fenêtre de perf (en jours) si connue. */
  window_days?: number | null;
};

/* ── Types : Calibration ────────────────────────────────────────────────── */

/**
 * Document de calibration pour un horizon donné.
 *
 * Permet de visualiser la relation entre ŷ et y :
 * - fit global d’une droite (alpha + beta * y_pred),
 * - calibration binned,
 * - calibration par heure locale,
 * - biais par niveau / par station.
 */
export type CalibrationDoc = {
  /** Version de schéma JSON. */
  schema_version: string;
  /** Timestamp de génération (UTC, ISO). */
  generated_at: string;

  /**
   * Fit global d’une droite de calibration (y_true ≈ alpha + beta * y_pred).
   * alpha / beta peuvent être null si le fit échoue.
   */
  fit: { alpha: number | null; beta: number | null; n: number };

  /**
   * Calibration binned :
   * - chaque ligne correspond à un bin de quantile,
   * - compare moyenne de y_pred vs y_true.
   */
  binned: Array<{
    quantile: string;
    y_pred_mean: number;
    y_true_mean: number;
    n: number;
  }>;

  /**
   * Calibration par heure locale (permet de voir des biais temporels).
   */
  by_hour: Array<{
    hour: number;
    alpha: number | null;
    beta: number | null;
    n: number;
  }>;

  /**
   * Erreur relative par "niveau" (faible / moyen / haut, etc.).
   */
  rel_error_levels: Array<{
    level: string;
    mape_like: number;
    n: number;
  }>;

  /**
   * Biais par station :
   * - station_id + name,
   * - bias (erreur moyenne signée),
   * - localisation (lat/lon),
   * - nombre de points.
   */
  bias_by_station: Array<{
    station_id: string;
    name: string | null;
    bias: number | null;
    lat: number | null;
    lon: number | null;
    n: number;
  }>;

  // Backend ≥ 1.3
  /** Horizon en minutes. */
  horizon_min?: number;
  /** Fenêtre de perf (en jours) si connue. */
  window_days?: number | null;
};

/* ── Types : Incertitude ────────────────────────────────────────────────── */

/**
 * Document d’incertitude (couverture des intervalles prédits).
 */
export type UncertaintyDoc = {
  /** Version de schéma JSON. */
  schema_version: string;
  /** Timestamp de génération (UTC, ISO). */
  generated_at: string;

  /**
   * Couverture empirique :
   * - empirical : fraction d’observations dont y_true tombe dans l’intervalle,
   * - n         : nombre de points considérés.
   * Peut être null si les intervalles ne sont pas disponibles.
   */
  coverage: { empirical: number; n: number } | null;

  /** Méthode utilisée pour l’incertitude (nom côté backend). */
  method?: string;

  /** Niveau nominal cible (ex: 0.8, 0.9). */
  nominal?: number | null;

  // Backend ≥ 1.3
  /** Horizon en minutes. */
  horizon_min?: number;
  /** Fenêtre de perf (en jours) si connue. */
  window_days?: number | null;
};

/* ── Types : Feature Importance ────────────────────────────────────────────
   Compat sur deux modes :
   - Ancien (surrogate RF + permutations) :
       rows = { feature, importance, std }
   - Nouveau (XGBoost natif) :
       rows = { feature, gain, weight, cover }
--------------------------------------------------------------------------- */

/**
 * Importance par feature, union des deux modes :
 * - Surrogate RF : importance / std,
 * - XGBoost natif : gain / weight / cover.
 */
export type FeatureImportanceRow = {
  /** Nom de la feature telle qu’exposée par le modèle. */
  feature: string;

  // Surrogate RF (ancien)
  /** Delta MAE (positif) par permutations. */
  importance?: number;
  /** Écart-type de l’importance (stabilité des permutations). */
  std?: number;

  // XGBoost natif (nouveau)
  /** Gain moyen apporté par la feature (critère interne XGB). */
  gain?: number;
  /** Nombre de splits impliquant cette feature. */
  weight?: number;
  /** Couverture moyenne de la feature. */
  cover?: number;
};

/**
 * Document global d’importance des features pour un horizon donné.
 */
export type FeatureImportanceDoc = {
  /** Version de schéma JSON. */
  schema_version: string;
  /** Timestamp de génération (UTC, ISO). */
  generated_at: string;

  /** Horizon en minutes. */
  horizon_min: number;

  /**
   * Méthode utilisée pour produire les importances (ou statut si indisponible) :
   * - "xgboost_native"            : importances natives XGB,
   * - "surrogate_rf_permutation"  : RF + permutations,
   * - "disabled"                  : explicabilité désactivée,
   * - "unavailable_sklearn"       : dépendances manquantes,
   * - "fit_error" / "permutation_error" : erreurs lors du fit / des permutations,
   * - "no_features" / "no_data"   : pas de features ou pas de données.
   */
  // Étend l’union pour couvrir les deux modes et les statuts d’erreur.
  method:
    | "xgboost_native"
    | "surrogate_rf_permutation"
    | "disabled"
    | "unavailable_sklearn"
    | "fit_error"
    | "permutation_error"
    | "no_features"
    | "no_data";

  /** Lignes d’importance (union des deux formats). */
  rows: FeatureImportanceRow[];

  /** Nombre de features considérées. */
  n_features: number;
  /** Nombre de lignes de perf utilisées. */
  n_rows: number;

  /** Notes / messages backend (warnings, limitations…). */
  notes: string[];
};

/* ── Helpers ───────────────────────────────────────────── */

/**
 * Construit le chemin d’endpoint relatif à
 * `/monitoring/model/explainability`.
 *
 * Exemple :
 *   path("/overview?h=15")
 *   → "/monitoring/model/explainability/overview?h=15"
 */
const path = (s: string) => `/monitoring/model/explainability${s}`;

/* ── API (ETag) ────────────────────────────────────────── */

/**
 * Vue d’ensemble "explainability" pour un horizon donné (en minutes).
 *
 * Endpoint :
 *   GET /monitoring/model/explainability/overview?h=H
 */
export const getExplainOverview = (h: number) =>
  fetchJsonWithEtag<Overview>(path(`/overview?h=${encodeURIComponent(h)}`));

/**
 * Résidus (histogramme, QQ, ACF, hétéroscédasticité, épisodes) pour un
 * horizon donné.
 *
 * Endpoint :
 *   GET /monitoring/model/explainability/residuals?h=H
 */
export const getExplainResiduals = (h: number) =>
  fetchJsonWithEtag<ResidualsDoc>(path(`/residuals?h=${encodeURIComponent(h)}`));

/**
 * Calibration globale / par quantile / par heure pour un horizon donné.
 *
 * Endpoint :
 *   GET /monitoring/model/explainability/calibration?h=H
 */
export const getExplainCalibration = (h: number) =>
  fetchJsonWithEtag<CalibrationDoc>(path(`/calibration?h=${encodeURIComponent(h)}`));

/**
 * Incertitude (couverture des intervalles prédits) pour un horizon donné.
 *
 * Endpoint :
 *   GET /monitoring/model/explainability/uncertainty?h=H
 */
export const getExplainUncertainty = (h: number) =>
  fetchJsonWithEtag<UncertaintyDoc>(path(`/uncertainty?h=${encodeURIComponent(h)}`));

/**
 * Importance des features pour un horizon donné.
 *
 * Endpoint :
 *   GET /monitoring/model/explainability/feature_importance?h=H
 */
export const getExplainFeatureImportance = (h: number) =>
  fetchJsonWithEtag<FeatureImportanceDoc>(path(`/feature_importance?h=${encodeURIComponent(h)}`));
