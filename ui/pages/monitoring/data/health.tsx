// ui/pages/monitoring/data/health.tsx
// =============================================================================
// Page Monitoring — Data Health (Santé des données)
// -----------------------------------------------------------------------------
// Cette page regroupe les principaux indicateurs de "santé" des données Vélib’ :
//   - volume (stations, lignes),
//   - couverture (complétude) globale et par heure,
//   - fraîcheur (fraîcheur p95 par station),
//   - anomalies récentes (séquences plates, doublons, bins manquants),
//   - alertes dérivées (SLO OK / non OK).
//
// Elle consomme plusieurs endpoints de monitoring back-end :
//   - /monitoring/data/health/kpis
//   - /monitoring/data/health/station-health
//   - /monitoring/data/health/coverage-by-hour
//   - /monitoring/data/health/anomalies
//   - /monitoring/data/health/alerts
//   - /monitoring/data/freshness/latest
//
// Le rendu utilise :
//   - MonitoringNav / KpiBar / LoadingBar (UI monitoring.css),
//   - Plotly (react-plotly.js) pour les graphiques,
//   - une table responsive pour les stations et anomalies.
// =============================================================================

import Head from "next/head";
import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import type * as Plotly from "plotly.js";
import MonitoringNav from "@/components/monitoring/MonitoringNav";
import LoadingBar, { type LoadingBarStatus } from "@/components/common/LoadingBar";
import KpiBar from "@/components/monitoring/KpiBar";
import { chartLayout, chartConfig } from "@/lib/plotlyTheme";
import { getDataFreshnessLatest, type FreshnessDoc } from "@/lib/services/monitoring/data_freshness";

import {
  getDataHealthKpis,
  getDataHealthStationHealth,
  getDataHealthCoverageByHour,
  getDataHealthAnomalies,
  getDataHealthAlerts,
  type DataKpis,
  type StationHealthRow,
  type CoverageByHourRow,
  type Anomaly,
  type AlertsDoc,
} from "@/lib/services/monitoring/data_health";

/* ───────────────────────── Plotly (client only) ───────────────────────── */
// Graphiques Plotly rendus uniquement côté client (ssr: false) pour éviter
// les accès à window/document pendant le rendu serveur.
const Plot = dynamic(() => import("react-plotly.js").then((m) => m.default), {
  ssr: false,
  loading: () => <div className="empty">Chargement du graphique…</div>,
});

/* ───────────────────────── Helpers ───────────────────────── */

/**
 * Extrait la valeur d’un PromiseSettledResult si status === "fulfilled",
 * sinon renvoie null. Permet de consommer Promise.allSettled() proprement.
 */
function ok<T>(r: PromiseSettledResult<T>): T | null {
  return r.status === "fulfilled" ? r.value : null;
}

/**
 * Formattage pourcentage : 0.123 → "0.1%".
 */
const fmtPct = (x?: number | null, d = 1) =>
  Number.isFinite(Number(x)) ? `${Number(x).toFixed(d)}%` : "—";

/**
 * Formattage entier avec séparateur français : 12345 → "12 345".
 */
const fmtInt = (x?: number | null) =>
  Number.isFinite(Number(x)) ? Number(x).toLocaleString("fr-FR") : "—";

/**
 * Formattage minutes : 12.34 → "12.3 min".
 */
const fmtMin = (x?: number | null, d = 1) =>
  Number.isFinite(Number(x)) ? `${Number(x).toFixed(d)} min` : "—";

/**
 * Contrainte d’une valeur dans [lo, hi].
 */
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

/**
 * Type guard pour savoir si une valeur est un nombre fini.
 */
const isFiniteNum = (x: any): x is number => Number.isFinite(Number(x));

/* ───────────────────────── Page ───────────────────────── */

/**
 * Page /monitoring/data/health
 *
 * Rôle :
 *   - afficher les KPI globaux de santé des données,
 *   - détailler la complétude (par heure + par station),
 *   - lister les anomalies récentes,
 *   - synthétiser les alertes SLA / SLO.
 *
 * Tous les appels aux endpoints de monitoring sont faits au montage via
 * Promise.allSettled() pour isoler les erreurs par sous-bloc.
 */
export default function DataHealthPage() {
  // Données brutes issues des services de monitoring
  const [kpis, setKpis] = useState<DataKpis | null>(null);
  const [stationHealth, setStationHealth] = useState<StationHealthRow[] | null>(null);
  const [covHour, setCovHour] = useState<CoverageByHourRow[] | null>(null);
  const [anomalies, setAnomalies] = useState<Anomaly[] | null>(null);
  const [alerts, setAlerts] = useState<AlertsDoc | null>(null);
  const [freshness, setFreshness] = useState<FreshnessDoc | null>(null);

  // État global de chargement / erreur
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filtres UI
  const [q, setQ] = useState("");
  const [qa, setQa] = useState("");
  const [atype, setAtype] =
    useState<"all" | "flat_sequence" | "duplicates" | "missing_bins">("all");

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setLoading(true);
        const res = await Promise.allSettled([
          getDataHealthKpis(),
          getDataHealthStationHealth(),
          getDataHealthCoverageByHour(),
          getDataHealthAnomalies(),
          getDataHealthAlerts(),
          getDataFreshnessLatest(),
        ]);
        if (!alive) return;

        setKpis(ok(res[0]));
        setStationHealth(ok(res[1]));
        setCovHour(ok(res[2]));
        setAnomalies(ok(res[3]));
        setAlerts(ok(res[4]));
        setFreshness(ok(res[5]));

        // Agrégation des erreurs éventuelles (au moins un appel rejeté)
        const failures = res.filter((r): r is PromiseRejectedResult => r.status === "rejected");
        setError(
          failures.length
            ? failures
                .map((f) => String((f.reason && (f.reason.message ?? f.reason)) || "request failed"))
                .join(" | ") || "Erreur API"
            : null
        );
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

  const generatedAt = kpis?.generated_at ?? undefined;
  const barStatus: LoadingBarStatus = loading ? "loading" : error ? "error" : "success";

  /* ───── KPI BAR ───── */
  // Construction des éléments de la barre de KPI (en haut de page).
  const kpiItems = useMemo(() => {
    const p95 = freshness?.stations?.freshness?.p95_min ?? null;
    return [
      { label: "Stations", value: fmtInt(kpis?.stations) },
      { label: "Lignes", value: fmtInt(kpis?.rows) },
      { label: "Couverture (moy.)", value: fmtPct(kpis?.coverage_global_pct, 1) },
      { label: "Fraîcheur p95", value: fmtMin(p95, 1) }, // ← fraîcheur p95 issue de data_freshness
      { label: "Doublons", value: fmtPct(kpis?.dups_pct, 2) },
    ];
  }, [kpis, freshness]);

  // Ligne de métadonnées (fenêtre, pas de temps, timezone, schéma, timestamp)
  const metaParts: string[] = [];
  if (kpis?.current_days != null) metaParts.push(`Fenêtre : ${kpis.current_days} j`);
  if (kpis?.bin_min != null) metaParts.push(`Pas : ${kpis.bin_min} min`);
  if (kpis?.tz) metaParts.push(`Fuseau horaire : ${kpis.tz}`);
  if (kpis?.schema_version != null) metaParts.push(`Schéma v${kpis.schema_version}`);
  if (generatedAt) metaParts.push(`généré ${generatedAt}`);
  const metaLine = metaParts.join(" · ");

  // Horodatage du dernier bin de fraîcheur (data_freshness)
  const freshBin = freshness?.meta?.bin_t_utc
    ? new Date(freshness.meta.bin_t_utc).toLocaleString("fr-FR")
    : null;

  /* ───── Vues dérivées ───── */

  // Représentation "propre" d’une ligne stationHealth pour l’UI.
  type CleanStation = {
    station_id: string;
    name: string;
    coverage: number | null;
    expected: number | null;
    obs: number | null;
    missing: number | null;
  };

  /**
   * Nettoie et dérive les champs de complétude d’une station :
   *   - station_id : string
   *   - name       : nom normalisé (fallback = id)
   *   - coverage   : couverture en %, recalculée si besoin
   *   - expected   : nombre d’échantillons attendus
   *   - obs        : nombre observé (fallback expected - missing)
   *   - missing    : nombre manquant
   */
  function cleanStationRow(r: StationHealthRow): CleanStation {
    const station_id = String(r.station_id);
    const name = (r.name && String(r.name).trim()) || station_id;
    const expected = isFiniteNum(r.expected) ? Number(r.expected) : null;
    const missing = isFiniteNum(r.missing) ? Number(r.missing) : null;
    const obsFromRow = isFiniteNum(r.obs) ? Number(r.obs) : null;
    const obs =
      obsFromRow ?? (expected !== null && missing !== null ? clamp(expected - missing, 0, expected) : null);

    let coverage: number | null = null;
    if (isFiniteNum(r.coverage_pct)) {
      coverage = clamp(Number(r.coverage_pct), 0, 100);
    } else if (expected && obs !== null && expected > 0) {
      coverage = clamp((obs / expected) * 100, 0, 100);
    }
    return { station_id, name, coverage, expected, obs, missing };
  }

  /**
   * Top 50 des stations à la complétude la plus basse (avec filtre texte).
   */
  const worstStations = useMemo(() => {
    const arr = (stationHealth ?? []).map(cleanStationRow);
    const term = q.trim().toLowerCase();
    const filtered = term
      ? arr.filter((r) => r.name.toLowerCase().includes(term) || r.station_id.toLowerCase().includes(term))
      : arr;

    const withCov = filtered.filter((r) => r.coverage !== null).sort((a, b) => a.coverage! - b.coverage!);
    const noCov = filtered.filter((r) => r.coverage === null);
    return [...withCov, ...noCov].slice(0, 50);
  }, [stationHealth, q]);

  /**
   * Petit résumé numérique des anomalies par type.
   */
  const anomaliesSummary = useMemo(() => {
    const a = anomalies ?? [];
    const flat = a.filter((x) => x.type === "flat_sequence").length;
    const dups = a.filter((x) => x.type === "duplicates").length;
    const miss = a.filter((x) => x.type === "missing_bins").length;
    return { flat, dups, miss };
  }, [anomalies]);

  /**
   * Vue filtrée des anomalies (par type + recherche texte) pour le tableau.
   */
  const anomaliesTable = useMemo(() => {
    let arr = anomalies ?? [];
    if (atype !== "all") arr = arr.filter((a) => a.type === atype);
    const term = qa.trim().toLowerCase();
    if (term) {
      arr = arr.filter((a) => {
        const name = "name" in a ? (a as any).name ?? "" : "";
        return (a.station_id?.toLowerCase?.() ?? "").includes(term) || String(name).toLowerCase().includes(term);
      });
    }
    return arr.slice(0, 200);
  }, [anomalies, atype, qa]);

  /**
   * Données Plotly pour le bar chart "couverture par heure".
   */
  const coverageBarData: Partial<Plotly.PlotData>[] = useMemo(() => {
    const arr = covHour ?? [];
    if (!arr.length) return [];
    const x = arr.map((r) => `${String(r.hour).padStart(2, "0")}h`);
    const y = arr.map((r) => Number(r.coverage_pct));
    return [
      {
        x,
        y,
        type: "bar" as const,
        name: "Couverture (%)",
        hovertemplate: "%{x} — %{y:.1f}%<extra></extra>",
      },
    ];
  }, [covHour]);

  return (
    <div className="monitoring">
      <Head>
        <title>Monitoring — Données / Santé</title>
        <meta name="description" content="Santé des données : fraîcheur, complétude, latence, anomalies." />
      </Head>

      <main className="page" style={{ paddingTop: "calc(var(--header-h, 70px) + 12px)" }}>
        <MonitoringNav
          title="Données — Santé"
          subtitle="Fraîcheur, complétude, latence et anomalies"
          generatedAt={generatedAt}
          extraActions={[{ label: "Drift", href: "/monitoring/data/drift" }]}
        />

        <LoadingBar status={barStatus} />

        {/* KPI bar + méta (fenêtre, pas de temps, timezone, etc.) */}
        <section className="mt-4">
          <KpiBar items={kpiItems} dense />
          {(metaLine || freshBin) && (
            <div className="kpi-bar-meta">
              {metaLine}
              {metaLine && freshBin ? " · " : ""}
              {freshBin ? `bin fraîcheur : ${freshBin}` : ""}
            </div>
          )}
        </section>

        {/* Couverture par heure (bar chart) */}
        <section className="mt-6">
          <h2>Couverture par heure (moyenne)</h2>
          <div className="card plot-card">
            <h3>Couverture (%)</h3>
            {coverageBarData.length ? (
              <Plot
                data={coverageBarData as Plotly.Data[]}
                layout={chartLayout({
                  autosize: true,
                  height: 320,
                  margin: { l: 56, r: 10, t: 10, b: 40 },
                  yaxis: { title: { text: "Couverture (%)" }, range: [0, 100], ticksuffix: "%" },
                  xaxis: { title: { text: "Heure (locale)" } },
                })}
                config={chartConfig}
                className="plot plot--sm"
              />
            ) : (
              <div className="empty">Pas de données de couverture horaire.</div>
            )}
          </div>
          <div className="figure-note small">
            Lecture : moyenne de la complétude par heure locale sur la fenêtre courante.
          </div>
        </section>

        {/* Stations — Top 50 complétude la plus basse */}
        <section className="mt-6">
          <h2>Stations — complétude (Top 50 les plus basses)</h2>

          <div className="filters">
            <input
              className="input"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Filtrer par nom ou id…"
              style={{ width: 340 }}
            />
            <div className="count-small">
              {stationHealth?.length ? `${worstStations.length} / ${stationHealth.length}` : "—"}
            </div>
          </div>

          <div className="card">
            {worstStations.length ? (
              <div className="table-scroll">
                <div
                  className="table-grid"
                  style={{ ["--cols" as any]: "360px 1fr 90px 90px" }}
                >
                  <HeaderCell>Station</HeaderCell>
                  <HeaderCell>Complétude</HeaderCell>
                  <HeaderCell>Obs</HeaderCell>
                  <HeaderCell>Attendu</HeaderCell>

                  {worstStations.map((r) => {
                    const cov = r.coverage !== null ? clamp(r.coverage, 0, 100) : null;
                    const obs = isFiniteNum(r.obs)
                      ? r.obs
                      : r.expected !== null && r.missing !== null
                      ? clamp(r.expected - r.missing, 0, r.expected)
                      : null;
                    return (
                      <Row key={r.station_id}>
                        <div>
                          <div style={{ fontWeight: 600 }}>{r.name}</div>
                          <div className="mono" style={{ fontSize: 12, opacity: 0.7 }}>
                            {r.station_id}
                          </div>
                        </div>
                        <div>
                          <div style={{ display: "grid", gridTemplateColumns: "1fr 64px", alignItems: "center", gap: 10 }}>
                            <div className="bar">
                              <div
                                className="bar__fill bar__fill--ok"
                                style={{ width: `${cov === null ? 0 : cov}%` }}
                                aria-hidden
                              />
                            </div>
                            <div className="table-cell--right" style={{ fontWeight: 700 }}>{fmtPct(cov, 1)}</div>
                          </div>
                        </div>
                        <div className="table-cell--right">{fmtInt(obs)}</div>
                        <div className="table-cell--right">{fmtInt(r.expected)}</div>
                      </Row>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div className="empty">Aucune station à afficher.</div>
            )}
          </div>
        </section>

        {/* Anomalies (séquences plates, doublons, bins manquants) */}
        <section className="mt-6">
          <h2>Anomalies (fenêtre récente)</h2>

          <div className="grid-3">
            <div className="card">
              <h3>Séquences plates</h3>
              <div style={{ fontSize: 18, fontWeight: 700 }}>{fmtInt(anomaliesSummary.flat)}</div>
              <div className="small mt-2">
                Nombre d’agrégats détectés (≥ {kpis?.thresholds?.flat_steps ?? "?"} pas consécutifs).
              </div>
            </div>
            <div className="card">
              <h3>Doublons</h3>
              <div style={{ fontSize: 18, fontWeight: 700 }}>{fmtInt(anomaliesSummary.dups)}</div>
              <div className="small mt-2">Somme des stations avec entrées dupliquées (même (horodatage, station)).</div>
            </div>
            <div className="card">
              <h3>Échantillons manquants</h3>
              <div style={{ fontSize: 18, fontWeight: 700 }}>{fmtInt(anomaliesSummary.miss)}</div>
              <div className="small mt-2">Top stations avec des échantillons manquants dans la fenêtre.</div>
            </div>
          </div>

          {/* Filtres anomalies (type + recherche texte) */}
          <div className="row mt-3" style={{ gap: 8, flexWrap: "wrap" }}>
            <Chip
              active={atype === "all"}
              onClick={() => setAtype("all")}
              label="Tous"
            />
            <Chip
              active={atype === "flat_sequence"}
              onClick={() => setAtype("flat_sequence")}
              label="Séquences plates"
            />
            <Chip
              active={atype === "duplicates"}
              onClick={() => setAtype("duplicates")}
              label="Doublons"
            />
            <Chip
              active={atype === "missing_bins"}
              onClick={() => setAtype("missing_bins")}
              label="Échantillons manquants"
            />

            <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
              <input
                className="input"
                value={qa}
                onChange={(e) => setQa(e.target.value)}
                placeholder="Filtrer par nom ou id…"
                style={{ width: 280 }}
              />
              <div className="count-small" style={{ alignSelf: "center" }}>
                {anomaliesTable.length} éléments
              </div>
            </div>
          </div>

          {/* Tableau anomalies détaillées */}
          <div className="card mt-3">
            {anomaliesTable.length ? (
              <div className="table-scroll">
                <div
                  className="table-grid"
                  style={{ ["--cols" as any]: "320px 160px 1fr" }} // Station | Type | Détails
                >
                  <HeaderCell>Station</HeaderCell>
                  <HeaderCell>Type</HeaderCell>
                  <HeaderCell>Détails</HeaderCell>

                  {anomaliesTable.map((a, i) => {
                    const stationName = "name" in a && a.name ? a.name : undefined;
                    const badgeKind =
                      a.type === "flat_sequence" || a.type === "duplicates" ? "warn" : "info";

                    return (
                      <Row key={i}>
                        {/* Station */}
                        <div>
                          <div style={{ fontWeight: 600 }}>{stationName ?? "—"}</div>
                          <div className="mono" style={{ fontSize: 12, opacity: 0.7 }}>
                            {a.station_id}
                          </div>
                        </div>

                        {/* Type */}
                        <div>
                          <Badge kind={badgeKind as any}>{a.type}</Badge>
                        </div>

                        {/* Détails */}
                        <div style={{ fontSize: 13 }}>
                          {a.type === "flat_sequence" && (
                            <span>
                              {(a as any).steps} pas — {(a as any).duration_min} min,&nbsp;
                              {new Date((a as any).start).toLocaleString("fr-FR")} →{" "}
                              {new Date((a as any).end).toLocaleString("fr-FR")}
                            </span>
                          )}
                          {a.type === "duplicates" && <span>{(a as any).dups} doublons</span>}
                          {a.type === "missing_bins" && (
                            <span>
                              {(a as any).missing}/{(a as any).expected} échantillons manquants
                            </span>
                          )}
                        </div>
                      </Row>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div className="empty">Aucune anomalie à afficher.</div>
            )}
          </div>
        </section>

        {/* Alertes dérivées des signaux de santé */}
        <section className="mt-6">
          <h2>Alertes</h2>
          <div className="card">
            <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
              {alerts?.freshness_p95_ok != null && (
                <Badge kind={alerts.freshness_p95_ok ? "ok" : "warn"}>
                  Fraîcheur p95 {alerts.freshness_p95_ok ? "OK" : "hors SLO"}
                </Badge>
              )}
              {typeof alerts?.coverage_ok === "boolean" && (
                <Badge kind={alerts.coverage_ok ? "ok" : "warn"}>
                  Couverture {alerts.coverage_ok ? "OK" : "basse"}
                </Badge>
              )}
              {typeof alerts?.duplication_alert === "boolean" && (
                <Badge kind={alerts.duplication_alert ? "warn" : "ok"}>
                  Doublons {alerts.duplication_alert ? "élevés" : "faibles"}
                </Badge>
              )}
              {typeof alerts?.flat_sequences_found === "boolean" && (
                <Badge kind={alerts.flat_sequences_found ? "warn" : "ok"}>
                  Séquences plates {alerts.flat_sequences_found ? "détectées" : "aucune"}
                </Badge>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

/* ───────────────────────── UI atoms (compat monitoring.css) ───────────────────────── */

/**
 * Badge coloré pour les alertes / types d’anomalies.
 *
 * kind :
 *   - "ok"   : contexte positif (OK / dans la cible),
 *   - "warn" : attention / alerte,
 *   - "info" : information neutre.
 */
function Badge({
  kind,
  children,
}: {
  kind: "ok" | "warn" | "info";
  children: React.ReactNode;
}) {
  const map: Record<"ok"|"warn"|"info", string> = {
    ok: "badge--ok",
    warn: "badge--warn",
    info: "badge--info",
  };
  return <span className={`badge ${map[kind]}`}>{children}</span>;
}

/**
 * Chip cliquable pour les filtres (type d’anomalie).
 */
function Chip({
  active,
  onClick,
  label,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
}) {
  return (
    <button
      type="button"
      className={`btn btn--ghost chip ${active ? "chip--active" : ""}`}
      aria-pressed={active}
      onClick={onClick}
      title={label}
    >
      {label}
    </button>
  );
}

/**
 * Ligne de tableau générique :
 * - aligne à droite les 2 dernières colonnes (valeurs numériques),
 * - conserve les colonnes de gauche alignées à gauche.
 */
function Row({ children }: { children: React.ReactNode }) {
  const items = Array.isArray(children) ? children : [children];
  return (
    <div className="table-row">
      {items.map((child, i) => {
        const alignRight = items.length >= 4 && i >= items.length - 2;
        return (
          <div
            key={i}
            className={`table-cell ${alignRight ? "table-cell--right" : ""}`}
          >
            {child}
          </div>
        );
      })}
    </div>
  );
}

/**
 * Cellule d’en-tête de tableau (ligne sticky en haut du scroll).
 */
function HeaderCell({ children }: { children: React.ReactNode }) {
  return <div className="table-head table-head--sticky">{children}</div>;
}