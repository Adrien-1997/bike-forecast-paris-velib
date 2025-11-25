// ui/lib/services/monitoring/intro.ts
// -----------------------------------------------------------------------------
// Service client pour le "hub" de monitoring : /monitoring/intro
//
// Rôle :
//   - Fournir un document synthétique pour la page d’accueil du monitoring,
//     avec :
//       • un bloc de KPIs globaux (stations, fraîcheur, couverture, etc.),
//       • un statut par sous-système (API stations, batch forecast, météo,
//         drift data), chacun avec un "led" (ok / warn / down),
//       • une liste d’entrées "activity" pour tracer les jobs récents,
//       • la liste des sources techniques (URIs/chemins de génération).
//
//   - Gérer automatiquement l’ETag, la base URL et le token d’authentification
//     via `fetchJsonWithEtag` (défini dans lib/http).
//
//   - Offrir une variante "time travel" via `getMonitoringIntroAt` permettant
//     de charger une version passée du document avec ?at=YYYY-MM-DDTHH-MM-SSZ.
//
// Ce service est *read-only* côté front : il ne modifie aucun état serveur.
// -----------------------------------------------------------------------------

import { fetchJsonWithEtag } from "@/lib/http";

/* ───────────────────────── Types ───────────────────────── */
/**
 * LED de statut :
 *  - "ok"   : tout va bien,
 *  - "warn" : dégradé / à surveiller,
 *  - "down" : à considérer comme hors service.
 */
export type Led = "ok" | "warn" | "down";

/**
 * Document racine renvoyé par l’API /monitoring/intro.
 *
 * Les champs sont pensés pour une page de synthèse très lisible :
 *   - `kpis` : KPIs globaux pour situer l’état du système,
 *   - `statuses` : statuts par sous-système (API, batch, météo, drift),
 *   - `activity` : liste d’événements/activités récentes,
 *   - `sources` : métadonnées techniques (origine des données, jobs, URIs…).
 */
export type IntroDoc = {
  /** Version de schéma (pour compatibilité front/back). */
  schema_version: string;
  /** Timestamp de génération du document (ISO, en général UTC). */
  generated_at: string;
  /** Fuseau horaire "logique" de la vue (ex: "Europe/Paris"). */
  tz: string;

  /**
   * KPIs globaux de santé.
   *
   * Tous les champs peuvent être null si la source n’est pas disponible
   * (ou en cas de panne partielle).
   */
  kpis: {
    /** Nombre de stations actives dans le snapshot récent. */
    stations_active: number | null;
    /**
     * Fraîcheur P95 des mesures stations (minutes) sur une fenêtre récente.
     * Permet d’identifier s’il y a du retard sur certaines stations.
     */
    freshness_p95_min: number | null;
    /**
     * Couverture des données sur les 7 derniers jours (% du temps couvert).
     * Exemple : 0.98 → 98 % de la fenêtre couverte.
     */
    coverage_7d_pct: number | null;
    /**
     * PSI global (population stability index) sur une période récente.
     * Optionnel : peut être absent si le job de drift n’a pas tourné.
     */
    psi_global?: number | null;
    /**
     * Versionning des modèles en production.
     * Exemple : "v0.9.2 / v1.0.0" pour horizon 15 min / 60 min.
     */
    model_versions: string | null;
  };

  /**
   * Statuts par sous-système.
   *
   * Chaque bloc suit le même principe :
   *   - `led` : Led ("ok"/"warn"/"down") pour l’affichage visuel,
   *   - métriques associées (fraîcheur, n_rows, station_active, etc.),
   *   - `source_generated_at` pour tracer l’horodatage de la source.
   */
  statuses: {
    /** Statut de l’API "live" des stations (flux temps réel). */
    api_stations: {
      led: Led;
      /** Nombre de stations considérées comme actives par l’API. */
      stations_active: number | null;
    };

    /** Statut du batch de prévision (jobs d’ingestion + forecasting). */
    batch_forecast: {
      led: Led;
      /**
       * Age en minutes du dernier export de forecast (ou équivalent).
       * Permet de détecter un batch en retard ou absent.
       */
      age_min: number | null;
      /** Timestamp de génération de la source (ISO). */
      source_generated_at: string | null;
      /** Nombre de lignes (échantillons) dans la dernière sortie (optionnel). */
      rows?: number | null;
    };

    /** Statut du fournisseur météo (requêtes API / pipeline météo). */
    weather_provider: {
      led: Led;
      /** Fraîcheur P95 de la météo (minutes) sur la fenêtre considérée. */
      freshness_p95_min: number | null;
      /** Timestamp de génération des données météo. */
      source_generated_at: string | null;
    };

    /**
     * Statut du drift data (optionnel).
     *
     * Logique attendue côté backend :
     *   - seuils PSI global typiques :
     *       psi < 0.10 → led = "ok"
     *       0.10–0.20 → led = "warn"
     *       >  0.20   → led = "down"
     *
     * `top_feature` et `top_feature_psi` pointent sur la feature qui dérive le
     * plus, pour orienter le diagnostic rapide dans l’UI.
     */
    data_drift?: {
      led: Led;
      /** PSI global (0..+∞). */
      psi_global: number | null;
      /** Feature la plus "driftée" (ex: "occ_ratio", "hour_of_day"...). */
      top_feature?: string | null;
      /** PSI de la feature la plus "driftée". */
      top_feature_psi?: number | null;
      /** Timestamp de génération du rapport de drift. */
      source_generated_at: string | null;
    };
  };

  /**
   * Entrées "activité" pour alimenter un fil de log synthétique sur la page.
   *
   * Exemple de structure côté backend :
   *   - label       : "Job build_datasets (h=60)"
   *   - value       : 123456 (nombre de lignes, durée, etc.)
   *   - generated_at: timestamp de l’événement
   */
  activity: Array<{
    label: string;
    value: unknown;
    generated_at: string | null;
  }>;

  /**
   * Dictionnaire de métadonnées "sources".
   *
   * Convention générale :
   *   - clé   : nom technique (ex: "stations_daily_export"),
   *   - valeur: description ou chemin/URI (ex: "gs://…/stations/daily").
   *
   * Utilisé pour transparence / troubleshooting dans l’UI.
   */
  sources: Record<string, string>;
};

/* ───────────────────────── Helpers ───────────────────────── */
/**
 * Construit le chemin de base pour le service /monitoring/intro.
 *
 * Permet de rajouter facilement des suffixes si de nouvelles routes sont
 * ajoutées sous le même préfixe (ex: /monitoring/intro/debug).
 */
const path = (suffix = "") => `/monitoring/intro${suffix}`;

/* ───────────────────────── API (ETag) ───────────────────────── */
/**
 * Récupère le document "courant" /monitoring/intro.
 *
 * Utilise `fetchJsonWithEtag` pour :
 *   - gérer les en-têtes ETag / If-None-Match,
 *   - éviter des recharges complètes si l’ETag n’a pas changé,
 *   - propager automatiquement baseURL + token.
 */
export const getMonitoringIntro = () =>
  fetchJsonWithEtag<IntroDoc>(path(""));

/**
 * Variante "time-travel" : charge le document /monitoring/intro
 * à une date/heure donnée (ISO, ex: "2025-10-23T10-00-00Z").
 *
 * Le backend doit gérer le paramètre `at` pour renvoyer la version
 * historisée ou la plus proche dans le temps.
 */
export const getMonitoringIntroAt = (atIsoStamp: string) =>
  fetchJsonWithEtag<IntroDoc>(`${path("")}?at=${encodeURIComponent(atIsoStamp)}`);
