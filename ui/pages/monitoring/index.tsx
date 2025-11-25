// ui/pages/monitoring/index.tsx
//
// -----------------------------------------------------------------------------
// Page "hub" du Monitoring : vue d’ensemble / porte d’entrée
//
// Rôle :
//   - Afficher un résumé synthétique de l’état du système (réseau + données + modèle),
//   - Centraliser quelques KPIs globaux (stations actives, fraîcheur, couverture, versions),
//   - Donner un statut visuel par sous-système (API, batch forecast, météo),
//   - Proposer des raccourcis vers les sections clés : réseau, données, modèle,
//   - Reposer sur le document /monitoring/intro fourni par le backend.
//
// Particularités :
//   - La logique de récupération des données est minimale : 1 seule requête
//     vers `getMonitoringIntro()`.
//   - Tous les helpers de formatage sont locaux à la page pour garder
//     l’affichage robuste aux null/NaN.
//   - La LED de statut (ok/warn/down) est mappée sur des classes CSS
//     `dot`, `dot--ok`, `dot--warn`, `dot--down` définies dans monitoring.css.
// -----------------------------------------------------------------------------

import Head from "next/head";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import MonitoringNav from "@/components/monitoring/MonitoringNav";
import LoadingBar, { type LoadingBarStatus } from "@/components/common/LoadingBar";
import KpiBar from "@/components/monitoring/KpiBar";
import { getMonitoringIntro, type IntroDoc } from "@/lib/services/monitoring/intro";

/* ─────────────────────── Helpers format ─────────────────────── */
/**
 * Formate un pourcentage sur d décimales.
 * - Retourne "—" si la valeur n’est pas numérique.
 * - Ne fait pas de conversion 0→0 %, on suppose que x est déjà en %.
 */
const fmtPct = (x: unknown, d = 1) =>
  x == null || Number.isNaN(Number(x)) ? "—" : `${Number(x).toFixed(d)}%`;

/**
 * Formate une durée en minutes sous forme lisible :
 *   < 60  → "X.X min"
 *   < 24h → "Y.Y h"
 *   sinon → "Z.Z j"
 */
const fmtMin = (x: unknown, d = 1): string => {
  if (x == null || Number.isNaN(Number(x))) return "—";
  const m = Number(x);
  if (m < 60) return `${m.toFixed(d)} min`;
  const h = m / 60;
  if (h < 24) return `${h.toFixed(1)} h`;
  const j = h / 24;
  return `${j.toFixed(1)} j`;
};

/**
 * Formate un entier avec locale fr-FR.
 * - Retourne "—" si la valeur n’est pas numérique.
 */
const fmtInt = (x: unknown) =>
  x == null || Number.isNaN(Number(x)) ? "—" : `${Math.round(Number(x)).toLocaleString("fr-FR")}`;

/**
 * Formate un timestamp ISO en datetime local (fr-FR).
 * - Retourne null si absent.
 */
const fmtDateTime = (iso?: string | null) => (iso ? new Date(iso).toLocaleString("fr-FR") : null);

/**
 * Mappe un statut logique ("ok" | "warn" | "down") vers une classe CSS
 * utilisée pour la pastille colorée (LED).
 */
const ledClass = (s: "ok" | "warn" | "down") =>
  s === "ok" ? "dot dot--ok" : s === "warn" ? "dot dot--warn" : "dot dot--down";

/* ───────────────────────── Component ───────────────────────── */
/**
 * Page d’introduction du Monitoring.
 *
 * Contenu :
 *   - Hero avec titre + texte explicatif,
 *   - KpiBar de synthèse (stations, fraîcheur, versions, couverture),
 *   - 3 cartes de navigation rapide (Réseau / Modèle / Données),
 *   - Bloc "Statut du système" avec LEDs,
 *   - Bloc "Activité récente" basé sur quelques KPIs,
 *   - Bloc "Conseils" avec liens rapides.
 *
 * Source des données :
 *   - `getMonitoringIntro()` (document IntroDoc).
 */
export default function MonitoringIntroPage() {
  // État de chargement + erreur globale
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Document de monitoring (IntroDoc) retourné par l’API
  const [doc, setDoc] = useState<IntroDoc | null>(null);

  // Chargement initial : 1 requête vers /monitoring/intro
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setLoading(true);
        setError(null);
        const d = await getMonitoringIntro();
        if (alive) setDoc(d);
      } catch (e: any) {
        if (alive) setError(String(e?.message ?? e));
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  // Statut de la barre de chargement (haut de page)
  const barStatus: LoadingBarStatus = loading ? "loading" : error ? "error" : "success";

  // KPIs de la hero bar (KpiBar)
  const kpiItems = useMemo(() => {
    const k = doc?.kpis;
    return [
      { label: "Stations actives", value: fmtInt(k?.stations_active) },
      { label: "Fraîcheur p95", value: fmtMin(k?.freshness_p95_min) },
      { label: "Version modèle", value: k?.model_versions ?? "—" },
      { label: "Couverture 7 j", value: fmtPct(k?.coverage_7d_pct) },
    ];
  }, [doc]);

  // Datetime de génération (formaté fr-FR)
  const generatedAt = fmtDateTime(doc?.generated_at) ?? null;

  return (
    <div className="monitoring">
      <Head>
        <title>Monitoring — Vue d’ensemble</title>
        <meta
          name="description"
          content="Vue d’ensemble : état du réseau, qualité des données et performance du modèle."
        />
      </Head>

      {/* Layout principal (padding top calé sur la hauteur du header app) */}
      <main className="page" style={{ paddingTop: "calc(var(--header-h, 70px) + 12px)" }}>
        {/* Barre de navigation Monitoring (sans breadcrumbs) */}
        <MonitoringNav title="Monitoring" generatedAt={generatedAt ?? undefined} />

        {/* Loading + erreur globale */}
        <div className="loadingbar-wrap mb-3">
          <LoadingBar status={barStatus} />
        </div>
        {error && <div className="banner banner--error mt-2">{error}</div>}

        {/* Hero + KPI */}
        <section className="panel hero">
          <div className="hero__title">
            <h2>Bienvenue sur le tableau de bord Monitoring</h2>
            <p className="muted">
              Suivez l’état du réseau, la qualité des données et la performance du modèle — au même endroit.
            </p>
          </div>
          <KpiBar items={kpiItems} dense />
          <div className="kpi-bar-meta">
            {generatedAt ? <>Mise à jour : {generatedAt}</> : <>Environnement actif · valeurs actualisées</>}
          </div>
        </section>

        {/* Sections rapides (cartes de navigation vers les grandes familles) */}
        <section className="grid-3 mt-6">
          <Link href="/monitoring/network/stations" className="card link-card">
            <div className="card__title">Réseau — Stations</div>
            <p className="card__body">Carte par clusters, profils moyens 24 h et distributions récentes.</p>
            <div className="card__cta">Ouvrir →</div>
          </Link>
          <Link href="/monitoring/model/performance" className="card link-card">
            <div className="card__title">Modèle — Performance</div>
            <p className="card__body">Précision et stabilité dans le temps (MAE, biais, dérive).</p>
            <div className="card__cta">Ouvrir →</div>
          </Link>
          <Link href="/monitoring/data/drift" className="card link-card">
            <div className="card__title">Données — Dérive</div>
            <p className="card__body">Dérive des variables d’entrée, valeurs manquantes et cohérence de schéma.</p>
            <div className="card__cta">Ouvrir →</div>
          </Link>
        </section>

        {/* Statut système (LED + texte) */}
        <section className="mt-6 grid-2">
          <div className="panel">
            <h3>Statut du système</h3>
            <ul className="status-list">
              {/* Statut API /stations */}
              <li>
                <span className={ledClass(doc?.statuses?.api_stations?.led ?? "down")} />
                <b> API /stations — </b>
                {(doc?.statuses?.api_stations?.led === "ok" && "Opérationnelle") ||
                  (doc?.statuses?.api_stations?.led === "warn" && "Instable") ||
                  "Indisponible"}
                <span className="muted">
                  {" · "}stations actives {fmtInt(doc?.statuses?.api_stations?.stations_active)}
                </span>
              </li>

              {/* Statut batch de prévisions */}
              <li>
                <span className={ledClass(doc?.statuses?.batch_forecast?.led ?? "down")} />
                <b> Batch de prévisions — </b>
                {(doc?.statuses?.batch_forecast?.led === "ok" && "Planifié") ||
                  (doc?.statuses?.batch_forecast?.led === "warn" && "Retard léger") ||
                  "Retard important"}
                <span className="muted">
                  {" · "}données âgées de {fmtMin(doc?.statuses?.batch_forecast?.age_min, 1)}
                </span>
              </li>

              {/* Statut fournisseur météo */}
              <li>
                <span className={ledClass(doc?.statuses?.weather_provider?.led ?? "down")} />
                <b> Fournisseur météo — </b>
                {(doc?.statuses?.weather_provider?.led === "ok" && "Normal") ||
                  (doc?.statuses?.weather_provider?.led === "warn" && "Ralentissements") ||
                  "Dégradé"}
                <span className="muted">
                  {" · "}fraîcheur : {fmtMin(doc?.statuses?.weather_provider?.freshness_p95_min, 1)}
                </span>
              </li>
            </ul>
            <div className="row mt-3">
              <Link href="/monitoring/data/health" className="btn btn-ghost">
                Santé des données
              </Link>
              <Link href="/monitoring/network/overview" className="btn btn-primary">
                Vue réseau
              </Link>
            </div>
          </div>

          {/* Activité récente simplifiée (quelques KPIs clés) */}
          <div className="panel">
            <h3>Activité récente</h3>
            <ul className="activity">
              <li>
                Version modèle : <b>{doc?.kpis?.model_versions ?? "—"}</b>
                <span className="muted">
                  {" "}— dernier entraînement :{" "}
                  {fmtDateTime(doc?.statuses?.batch_forecast?.source_generated_at) ?? "—"}
                </span>
              </li>
              <li>
                Couverture (7 jours) : <b>{fmtPct(doc?.kpis?.coverage_7d_pct)}</b>
                <span className="muted"> — backfill terminé</span>
              </li>
              <li>
                Stations actives : <b>{fmtInt(doc?.kpis?.stations_active)}</b>
                <span className="muted"> — statut réseau</span>
              </li>
            </ul>
            <div className="row mt-3">
              <Link href="/monitoring/model/explainability" className="btn btn-ghost">
                Explicabilité
              </Link>
              <Link href="/monitoring/model/performance" className="btn btn-primary">
                Performance
              </Link>
            </div>
          </div>
        </section>

        {/* Conseils d’utilisation / raccourcis pratiques */}
        <section className="panel mt-6">
          <h3>Conseils</h3>
          <ul className="tips">
            <li>
              Utilisez les onglets pour naviguer entre <b>Réseau</b>, <b>Données</b> et <b>Modèle</b>.
            </li>
            <li>
              Sur les cartes de stations, activez <b>Auto-fit</b> et <b>Taille = capacité</b> pour explorer plus vite.
            </li>
            <li>Exportez le CSV des stations pour partager les clusters avec les opérations.</li>
          </ul>
          <div className="row mt-2">
            <Link href="/monitoring/data/drift" className="btn btn-ghost">
              Voir la dérive
            </Link>
            <Link href="/monitoring/network/stations" className="btn btn-primary">
              Ouvrir les stations
            </Link>
          </div>
        </section>
      </main>
    </div>
  );
}