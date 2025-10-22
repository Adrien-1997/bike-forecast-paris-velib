// ui/pages/monitoring/index.tsx
import Head from "next/head";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import MonitoringNav from "@/components/monitoring/MonitoringNav";
import LoadingBar, { type LoadingBarStatus } from "@/components/common/LoadingBar";
import KpiBar from "@/components/monitoring/KpiBar";

export default function MonitoringIntroPage() {
  const generatedAt: string | null = null;

  // ── LoadingBar logic (uniforme) ───────────────────────────────────────────
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const barStatus: LoadingBarStatus = loading ? "loading" : error ? "error" : "success";

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setLoading(true);
        setError(null);
        // (Futur) Appels API globaux si besoin
        // await Promise.allSettled([ getSomething(), ... ])
      } catch (e: any) {
        if (alive) setError(String(e?.message ?? e));
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, []);

  // ── KPIs (remplace la grille par KpiBar) ──────────────────────────────────
  const kpiItems = useMemo(() => ([
    { label: "Network status", value: "OK" },
    { label: "Data freshness", value: "≤ 5 min" },
    { label: "Model version", value: "v1.8.2" },
    { label: "Coverage (7d)", value: "99.2%" },
  ]), []);

  return (
    <div className="monitoring">
      <Head>
        <title>Monitoring — Overview</title>
        <meta
          name="description"
          content="Monitoring dashboard overview: network health, data quality, and model performance at a glance."
        />
      </Head>

      {/* Contenu principal monitoring (header/footer injectés par _app.tsx) */}
      <main
        className="page"
        style={{ paddingTop: "calc(var(--header-h, 70px) + 12px)" }}
      >
        <MonitoringNav title="Monitoring" generatedAt={generatedAt ?? undefined} />

        {/* Loading / Error → LoadingBar uniforme */}
        <LoadingBar status={barStatus} />
        {error && <div className="banner banner--error mt-2">{error}</div>}

        {/* Hero + KpiBar */}
        <section className="panel hero">
          <div className="hero__title">
            <h2>Welcome to Monitoring</h2>
            <p className="muted">
              Track network health, data quality and model performance — all in one place.
            </p>
          </div>

          {/* ✅ KpiBar compacte, scroll desktop auto, mobile max 3 par ligne via CSS globale */}
          <KpiBar items={kpiItems} dense />

          {/* (meta optionnelle sous la barre) */}
          <div className="kpi-bar-meta">Demo environment · values illustrative</div>
        </section>

        {/* Quick sections */}
        <section className="grid-3 mt-6">
          <Link href="/monitoring/network/stations" className="card link-card">
            <div className="card__title">Network — Stations</div>
            <p className="card__body">
              Map of stations by cluster, 24h centroid profiles and recent distributions.
            </p>
            <div className="card__cta">Open →</div>
          </Link>

          <Link href="/monitoring/model/performance" className="card link-card">
            <div className="card__title">Model — Performance</div>
            <p className="card__body">
              Evaluate predictive accuracy and stability over time (MAE, bias, drift).
            </p>
            <div className="card__cta">Open →</div>
          </Link>

          <Link href="/monitoring/data/drift" className="card link-card">
            <div className="card__title">Data — Drift</div>
            <p className="card__body">
              Monitor input feature drift, missing rates and schema consistency.
            </p>
            <div className="card__cta">Open →</div>
          </Link>
        </section>

        {/* System status + recent activity */}
        <section className="mt-6 grid-2">
          <div className="panel">
            <h3>System status</h3>
            <ul className="status-list">
              <li>
                <span className="dot dot--ok" /> API /stations — <b>Healthy</b>
                <span className="muted"> · median 42ms</span>
              </li>
              <li>
                <span className="dot dot--ok" /> Batch forecast — <b>On schedule</b>
                <span className="muted"> · every 5 min</span>
              </li>
              <li>
                <span className="dot dot--warn" /> Weather provider — <b>Slow</b>
                <span className="muted"> · spikes detected</span>
              </li>
            </ul>
            <div className="row mt-3">
              <Link href="/monitoring/data/health" className="btn btn-ghost">
                Data health
              </Link>
              <Link href="/monitoring/network/overview" className="btn btn-primary">
                Network overview
              </Link>
            </div>
          </div>

          <div className="panel">
            <h3>Recent activity</h3>
            <ul className="activity">
              <li>
                <b>Model v1.8.2</b> deployed to production
                <span className="muted"> — 2 days ago</span>
              </li>
              <li>
                Backfill job completed for <b>last 7 days</b>
                <span className="muted"> — yesterday</span>
              </li>
              <li>
                New station metadata synced <b>(12 updates)</b>
                <span className="muted"> — today</span>
              </li>
            </ul>
            <div className="row mt-3">
              <Link href="/monitoring/model/explainability" className="btn btn-ghost">
                Explainability
              </Link>
              <Link href="/monitoring/model/performance" className="btn btn-primary">
                Performance
              </Link>
            </div>
          </div>
        </section>

        {/* Tips / docs */}
        <section className="panel mt-6">
          <h3>Tips</h3>
          <ul className="tips">
            <li>Use the top tabs to switch between Network, Data and Model sections.</li>
            <li>
              On station maps, enable <b>Auto-fit</b> and <b>Size = capacity</b> for quick exploration.
            </li>
            <li>Export CSV from Stations to share cluster assignments with ops.</li>
          </ul>
          <div className="row mt-2">
            <Link href="/monitoring/data/drift" className="btn btn-ghost">
              See drift
            </Link>
            <Link href="/monitoring/network/stations" className="btn btn-primary">
              Open stations
            </Link>
          </div>
        </section>
      </main>
    </div>
  );
}
