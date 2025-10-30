// ui/pages/monitoring/index.tsx
import Head from "next/head";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import MonitoringNav from "@/components/monitoring/MonitoringNav";
import LoadingBar, { type LoadingBarStatus } from "@/components/common/LoadingBar";
import KpiBar from "@/components/monitoring/KpiBar";
import { getMonitoringIntro, type IntroDoc } from "@/lib/services/monitoring/intro";

// ───────────────────────── Helpers format ─────────────────────────
const fmtPct = (x: unknown, d = 1) =>
  x == null || Number.isNaN(Number(x)) ? "—" : `${Number(x).toFixed(d)}%`;

const fmtMin = (x: unknown, d = 1): string => {
  if (x == null || Number.isNaN(Number(x))) return "—";
  const m = Number(x);
  if (m < 60) return `${m.toFixed(d)} min`;
  const h = m / 60;
  if (h < 24) return `${h.toFixed(1)} h`;
  const j = h / 24;
  return `${j.toFixed(1)} j`;
};

const fmtInt = (x: unknown) =>
  x == null || Number.isNaN(Number(x)) ? "—" : `${Math.round(Number(x)).toLocaleString("fr-FR")}`;

const fmtDateTime = (iso?: string | null) => (iso ? new Date(iso).toLocaleString("fr-FR") : null);

const ledClass = (s: "ok" | "warn" | "down") =>
  s === "ok" ? "dot dot--ok" : s === "warn" ? "dot dot--warn" : "dot dot--down";

// ───────────────────────── Component ─────────────────────────
export default function MonitoringIntroPage() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [doc, setDoc] = useState<IntroDoc | null>(null);

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

  const barStatus: LoadingBarStatus = loading ? "loading" : error ? "error" : "success";

  const kpiItems = useMemo(() => {
    const k = doc?.kpis;
    return [
      { label: "Stations actives", value: fmtInt(k?.stations_active) },
      { label: "Fraîcheur p95", value: fmtMin(k?.freshness_p95_min) },
      { label: "Version modèle", value: k?.model_versions ?? "—" },
      { label: "Couverture 7 j", value: fmtPct(k?.coverage_7d_pct) },
    ];
  }, [doc]);

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

      <main className="page" style={{ paddingTop: "calc(var(--header-h, 70px) + 12px)" }}>
        <MonitoringNav title="Monitoring" generatedAt={generatedAt ?? undefined} />

        {/* Loading + erreur avec espacement */}
        <div className="loadingbar-wrap mb-3">
          <LoadingBar status={barStatus} />
        </div>
        {error && <div className="banner banner--error mt-2">{error}</div>}

        {/* Hero + KpiBar */}
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

        {/* Sections rapides */}
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

        {/* Statut système */}
        <section className="mt-6 grid-2">
          <div className="panel">
            <h3>Statut du système</h3>
            <ul className="status-list">
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

        {/* Conseils */}
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
