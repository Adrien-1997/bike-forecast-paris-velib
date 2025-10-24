// ui/pages/monitoring/model/performance.tsx
import Head from "next/head";
import { useCallback, useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import type * as Plotly from "plotly.js";
import MonitoringNav from "@/components/monitoring/MonitoringNav";
import LoadingBar, { type LoadingBarStatus } from "@/components/common/LoadingBar";
import KpiBar from "@/components/monitoring/KpiBar";
import { chartConfig, chartLayout } from "@/lib/plotlyTheme";

import {
  getPerformanceManifest,
  getPerformanceKpis,
  getPerformanceDailyMetrics,
  getPerformanceByHour,
  getPerformanceByDow,
  getPerformanceByStation,
  getPerformanceLiftCurve,
  getPerformanceHistResiduals,
  type Manifest,
  type KPIs,
  type DailyMetrics,
  type ByHour,
  type ByDow,
  type ByStation,
  type StationRow,
  type LiftCurve,
  type HistResiduals,
} from "@/lib/services/monitoring/model_performance";

/* ───────────────────────── Plotly (client only) ───────────────────────── */
const Plot = dynamic(() => import("react-plotly.js").then((m) => m.default), {
  ssr: false,
  loading: () => (
    <div style={{ height: 320, display: "grid", placeItems: "center", opacity: 0.7 }}>
      Chargement du graphique…
    </div>
  ),
});

/* ───────────────────────── Helpers ───────────────────────── */
function ok<T>(r: PromiseSettledResult<T>): T | null {
  return r.status === "fulfilled" ? r.value : null;
}
function fmtPct(x?: number | null, digits = 1) {
  const v = Number(x);
  return Number.isFinite(v) ? `${v.toFixed(digits)}%` : "—";
}
function fmtNum(x?: number | null, digits = 2) {
  const v = Number(x);
  return Number.isFinite(v) ? v.toFixed(digits) : "—";
}
function fmtInt(x?: number | null) {
  const v = Number(x);
  return Number.isFinite(v) ? v.toLocaleString("fr-FR") : "—";
}
const toStr = (x: any) => (x == null ? "—" : String(x));

function useQueryParamH(defaultH = 15): [number, (h: number) => void] {
  const [h, setH] = useState(defaultH);
  useEffect(() => {
    if (typeof window !== "undefined") {
      const u = new URL(window.location.href);
      const qh = Number(u.searchParams.get("h"));
      if (Number.isFinite(qh) && qh > 0) setH(qh);
    }
  }, []);
  const setter = useCallback((next: number) => {
    setH(next);
    if (typeof window !== "undefined") {
      const u = new URL(window.location.href);
      u.searchParams.set("h", String(next));
      window.history.replaceState({}, "", u.toString());
    }
  }, []);
  return [h, setter];
}

/* ───────────────────────── Page ───────────────────────── */
export default function ModelPerformancePage() {
  const [h, setH] = useQueryParamH(15);

  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [kpis, setKpis] = useState<KPIs | null>(null);
  const [daily, setDaily] = useState<DailyMetrics | null>(null);
  const [byHour, setByHour] = useState<ByHour | null>(null);
  const [byDow, setByDow] = useState<ByDow | null>(null);
  const [byStation, setByStation] = useState<ByStation | null>(null);
  const [lift, setLift] = useState<LiftCurve | null>(null);
  const [hist, setHist] = useState<HistResiduals | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Manifest
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const m = await getPerformanceManifest();
        if (!alive) return;
        setManifest(m);
        if (Array.isArray(m.horizons) && m.horizons.length && !m.horizons.includes(h)) {
          setH(m.horizons[0]);
        }
        setError(null);
      } catch (e: any) {
        if (alive) setError(String(e?.message ?? e));
      }
    })();
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Data for current horizon
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setLoading(true);
        const res = await Promise.allSettled([
          getPerformanceKpis(h),
          getPerformanceDailyMetrics(h),
          getPerformanceByHour(h),
          getPerformanceByDow(h),
          getPerformanceByStation(h),
          getPerformanceLiftCurve(h),
          getPerformanceHistResiduals(h),
        ]);
        if (!alive) return;

        setKpis(ok(res[0]));
        setDaily(ok(res[1]));
        setByHour(ok(res[2]));
        setByDow(ok(res[3]));
        setByStation(ok(res[4]));
        setLift(ok(res[5]));
        setHist(ok(res[6]));

        // Erreur si échec partiel (aligné autres pages)
        const failures = res.filter((r): r is PromiseRejectedResult => r.status === "rejected");
        if (failures.length > 0) {
          const msg = failures
            .map((f) => String((f.reason && (f.reason.message ?? f.reason)) || "request failed"))
            .join(" | ") || "API error";
          setError(msg);
        } else {
          setError(null);
        }
      } catch (e: any) {
        if (alive) setError(String(e?.message ?? e));
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, [h]);

  const generatedAt = kpis?.generated_at ?? manifest?.generated_at ?? null;

  // ✅ Barre d’état uniforme
  const barStatus: LoadingBarStatus = loading ? "loading" : error ? "error" : "success";

  /* ───────── KPI BAR ───────── */
  const kpiItems = useMemo(() => {
    const liftPct = Number.isFinite(Number(kpis?.lift_vs_baseline)) ? Number(kpis!.lift_vs_baseline) * 100 : null;
    return [
      { label: "Stations", value: fmtInt(kpis?.n_stations) },
      { label: "Lignes (n)", value: fmtInt(kpis?.n_rows) },
      { label: "MAE — Modèle", value: fmtNum(kpis?.mae_model, 2) },
      { label: "MAE — Baseline", value: fmtNum(kpis?.mae_baseline, 2) },
      { label: "Lift vs baseline", value: fmtPct(liftPct, 1) },
      { label: "Coverage préd.", value: fmtPct(kpis?.coverage_pred_pct, 1) },
    ];
  }, [kpis]);

  const metaParts: string[] = [];
  const schemaV = kpis?.schema_version ?? manifest?.schema_version;
  if (schemaV != null) metaParts.push(`Schema v${schemaV}`);
  metaParts.push(`Intervalle: ${toStr(kpis?.ts_min_utc ?? "—")} → ${toStr(kpis?.ts_max_utc ?? "—")} (UTC)`);
  const winDays = kpis?.window_days ?? manifest?.window_days;
  if (winDays != null) metaParts.push(`Fenêtre: ${winDays} j`);
  if (generatedAt) metaParts.push(`generated ${generatedAt}`);
  const metaLine = metaParts.join(" · ");

  /* ───────── Dérivations graphiques ───────── */
  const liftCurve: Partial<Plotly.PlotData> | null = useMemo(() => {
    const pts = lift?.points ?? [];
    if (!pts.length) return null;
    const x = pts.map((p) => p.date);
    const y = pts.map((p) =>
      Number.isFinite(Number(p.lift_vs_baseline)) ? Number(p.lift_vs_baseline) * 100 : null
    );
    return { x, y, type: "scatter", mode: "lines", name: "Lift (%)", connectgaps: false, hovertemplate: "%{x} — %{y:.1f}%<extra></extra>" };
  }, [lift]);

  const dailyBars: Partial<Plotly.PlotData>[] = useMemo(() => {
    const rows = daily?.rows ?? [];
    if (!rows.length) return [];
    const x = rows.map((r) => r.date);
    const m = rows.map((r) => (Number.isFinite(Number(r.mae_model)) ? Number(r.mae_model) : null));
    const b = rows.map((r) => (Number.isFinite(Number(r.mae_baseline)) ? Number(r.mae_baseline) : null));
    return [
      { x, y: b, type: "bar" as const, name: "Baseline — MAE" },
      { x, y: m, type: "bar" as const, name: "Modèle — MAE" },
    ];
  }, [daily]);

  const byHourBars: Partial<Plotly.PlotData>[] = useMemo(() => {
    const rows = byHour?.rows ?? [];
    if (!rows.length) return [];
    const x = rows.map((r) => r.hour);
    const m = rows.map((r) => (Number.isFinite(Number(r.mae_model)) ? Number(r.mae_model) : null));
    const b = rows.map((r) => (Number.isFinite(Number(r.mae_baseline)) ? Number(r.mae_baseline) : null));
    return [
      { x, y: b, type: "bar" as const, name: "Baseline — MAE" },
      { x, y: m, type: "bar" as const, name: "Modèle — MAE" },
    ];
  }, [byHour]);

  const byDowBars: Partial<Plotly.PlotData>[] = useMemo(() => {
    const rows = byDow?.rows ?? [];
    if (!rows.length) return [];
    const label = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"];
    const x = rows.map((r) => label[r.dow] ?? String(r.dow));
    const m = rows.map((r) => (Number.isFinite(Number(r.mae_model)) ? Number(r.mae_model) : null));
    const b = rows.map((r) => (Number.isFinite(Number(r.mae_baseline)) ? Number(r.mae_baseline) : null));
    return [
      { x, y: b, type: "bar" as const, name: "Baseline — MAE" },
      { x, y: m, type: "bar" as const, name: "Modèle — MAE" },
    ];
  }, [byDow]);

  const histData: Partial<Plotly.PlotData>[] = useMemo(() => {
    const bins = hist?.bins ?? [];
    const counts = hist?.counts ?? [];
    if (!bins.length || !counts.length || bins.length !== counts.length + 1) return [];
    const centers = bins.slice(0, -1).map((b, i) => (b + bins[i + 1]) / 2);
    return [{ x: centers, y: counts, type: "bar" as const, name: "Résidus (y_true - y_pred)" }];
  }, [hist]);

  // ── Stations Top/Bottom lift ─────────────────────────
  const stationsTopBottom = useMemo(() => {
    const rows = byStation?.rows ?? [];
    const wLift = (x: StationRow) =>
      Number.isFinite(Number(x.lift_vs_baseline)) ? Number(x.lift_vs_baseline) : -Infinity;
    const minN = 30;
    const filtered = rows.filter((r) => Number(r.n) >= minN);
    const tops = [...filtered].sort((a, b) => wLift(b) - wLift(a)).slice(0, 20);
    const bots = [...filtered].sort((a, b) => wLift(a) - wLift(b)).slice(0, 20);
    return { tops, bots };
  }, [byStation]);

  return (
    <div className="monitoring">
      <Head>
        <title>Monitoring — Model / Performance</title>
        <meta
          name="description"
          content="Comparatif modèle vs baseline, lift, histogrammes, découpes et stations."
        />
        {/* Mini styles internes pour les barres de lift station (local à la page) */}
        <style
          dangerouslySetInnerHTML={{
            __html: `
              .monitoring .bar{height:6px;border-radius:999px;background:#111827;position:relative;overflow:hidden}
              .monitoring .bar__fill{height:100%;background:var(--ok)}
              .monitoring .bar__fill--danger{background:#ef4444}
            `,
          }}
        />
      </Head>

      <main className="page" style={{ paddingTop: "calc(var(--header-h, 70px) + 12px)" }}>
        <MonitoringNav
          title="Model — Performance"
          subtitle={`MAE/RMSE vs baseline, lift, histogrammes & découpes (h=${h} min)`}
          generatedAt={generatedAt ?? undefined}
          extraActions={[{ label: "Explainability", href: "/monitoring/model/explainability" }]}
        />

        {/* ✅ LoadingBar uniforme */}
        <LoadingBar status={barStatus} />

        {/* Toolbar / Controls */}
        <section className="mt-3">
          <div className="card" style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
            <div className="small">Horizon</div>
            <HorizonSelect value={h} options={manifest?.horizons ?? [15]} onChange={(v) => setH(v)} />
            <div className="small" style={{ marginLeft: 8 }}>
              fenêtre: {kpis?.window_days ?? manifest?.window_days ?? "—"} j
            </div>
          </div>
        </section>

        {/* KPIs via KpiBar */}
        <section className="mt-4">
          <h2>Résumé — KPIs (h={h} min)</h2>
          <KpiBar items={kpiItems} dense />
          {metaLine && <div className="kpi-bar-meta">{metaLine}</div>}
        </section>

        {/* Lift over time */}
        <section className="mt-6">
          <h2>Lift quotidien</h2>
          <div className="card plot-card">
            {liftCurve ? (
              <Plot
                data={[liftCurve as Plotly.Data]}
                layout={chartLayout({
                  height: 320,
                  margin: { l: 54, r: 10, t: 10, b: 40 },
                  yaxis: { title: { text: "Lift (%)" } },
                  xaxis: { title: { text: "Date (locale)" } },
                  hovermode: "x unified",
                })}
                config={chartConfig}
                className="plot plot--lg"
              />
            ) : (
              <div className="empty">Pas de courbe de lift.</div>
            )}
          </div>
          <div className="figure-note small">
            Lecture : lift = (MAE_baseline − MAE_modèle) / MAE_baseline, agrégé par jour.
          </div>
        </section>

        {/* Daily bars MAE */}
        <section className="mt-6">
          <h2>MAE par jour (Modèle vs Baseline)</h2>
          <div className="card plot-card">
            {dailyBars.length ? (
              <Plot
                data={dailyBars as Plotly.Data[]}
                layout={chartLayout({
                  barmode: "group",
                  height: 360,
                  margin: { l: 54, r: 10, t: 10, b: 60 },
                  yaxis: { title: { text: "MAE" } },
                  xaxis: { title: { text: "Date (locale)" }, tickangle: -30 },
                })}
                config={chartConfig}
                className="plot plot--lg"
              />
            ) : (
              <div className="empty">Pas de métriques quotidiennes.</div>
            )}
          </div>
          <div className="figure-note small">
            Deux barres par date : baseline vs modèle. MAE = moyenne des erreurs absolues.
          </div>
        </section>

        {/* By hour / By dow */}
        <section className="mt-6">
          <h2>Découpes — heure & jour de semaine</h2>
          <div className="grid-2">
            <div className="card plot-card">
              <h3>Par heure (locale)</h3>
              {byHourBars.length ? (
                <Plot
                  data={byHourBars as Plotly.Data[]}
                  layout={chartLayout({
                    barmode: "group",
                    height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 36 },
                    yaxis: { title: { text: "MAE" } },
                    xaxis: { title: { text: "Heure" } },
                  })}
                  config={chartConfig}
                  className="plot plot--sm"
                />
              ) : (
                <div className="empty">—</div>
              )}
            </div>

            <div className="card plot-card">
              <h3>Par jour de semaine</h3>
              {byDowBars.length ? (
                <Plot
                  data={byDowBars as Plotly.Data[]}
                  layout={chartLayout({
                    barmode: "group",
                    height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 36 },
                    yaxis: { title: { text: "MAE" } },
                    xaxis: { title: { text: "Jour" } },
                  })}
                  config={chartConfig}
                  className="plot plot--sm"
                />
              ) : (
                <div className="empty">—</div>
              )}
            </div>
          </div>
          <div className="figure-note small mt-2">
            Découpes agrégées : comparaison des MAE par heure locale et par jour (Lun–Dim).
          </div>
        </section>

        {/* Histogramme des résidus */}
        <section className="mt-6">
          <h2>Histogramme des résidus (y_true - y_pred)</h2>
          <div className="card plot-card">
            {histData.length ? (
              <Plot
                data={histData as Plotly.Data[]}
                layout={chartLayout({
                  height: 320,
                  margin: { l: 54, r: 10, t: 10, b: 40 },
                  xaxis: { title: { text: "Erreur" } },
                  yaxis: { title: { text: "Comptes" } },
                })}
                config={chartConfig}
                className="plot plot--sm"
              />
            ) : (
              <div className="empty">Pas d’histogramme disponible.</div>
            )}
          </div>
          {hist?.n != null && <div className="small mt-2">n = {fmtInt(hist.n)} points</div>}
        </section>

        {/* Top / Bottom stations */}
        <section className="mt-6" style={{ marginBottom: 40 }}>
          <h2>Stations — meilleurs / moins bons lifts</h2>
          <div className="grid-2">
            <div className="card">
              <h3>Top 20 (lift)</h3>
              {stationsTopBottom.tops.length ? (
                <StationList rows={stationsTopBottom.tops} kind="top" />
              ) : (
                <div className="empty">Pas de stations (échantillon trop faible).</div>
              )}
            </div>
            <div className="card">
              <h3>Bottom 20 (lift)</h3>
              {stationsTopBottom.bots.length ? (
                <StationList rows={stationsTopBottom.bots} kind="bottom" />
              ) : (
                <div className="empty">Pas de stations (échantillon trop faible).</div>
              )}
            </div>
          </div>
          <div className="small mt-2">
            Filtre stations avec n ≥ 30 pour éviter les faux positifs. Le <b>lift</b> = (MAE_baseline − MAE_modèle) / MAE_baseline.
          </div>
        </section>
      </main>
    </div>
  );
}

/* ───────────────────────── UI widgets ───────────────────────── */
function HorizonSelect({
  value,
  options,
  onChange,
}: {
  value: number;
  options: number[];
  onChange: (v: number) => void;
}) {
  const opts = Array.isArray(options) && options.length ? options : [15];
  return (
    <div className="row" style={{ gap: 8 }}>
      <select
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="select"
      >
        {opts.map((o) => (
          <option key={o} value={o}>
            {o} min
          </option>
        ))}
      </select>
      <span className="small">({opts.length} option{opts.length > 1 ? "s" : ""})</span>
    </div>
  );
}

function StationList({
  rows,
  kind,
  nameIndex,
}: {
  rows: StationRow[];
  kind: "top" | "bottom";
  nameIndex?: Record<string, string>;
}) {
  return (
    <div className="table-scroll" style={{ overflowX: "hidden" }}>
      <div
        className="table-grid"
        style={{
          // Station | MAE base | MAE modèle | Lift (plus collées)
          ["--cols" as any]: "minmax(0,1fr) 78px 78px 66px",
          minWidth: 0,
          gap: "4px", // ⬅️ très serré
        }}
      >
        <div className="table-head table-head--sticky">Station</div>
        <div className="table-head table-head--sticky">MAE base</div>
        <div className="table-head table-head--sticky">MAE modèle</div>
        <div className="table-head table-head--sticky">Lift</div>

        {rows.map((r) => {
          const displayName =
            (nameIndex && nameIndex[r.station_id]) ||
            ((r as any).name as string) ||
            r.station_id;

          const lift = Number.isFinite(Number(r.lift_vs_baseline))
            ? Number(r.lift_vs_baseline)
            : null;
          const liftPct = lift != null ? lift * 100 : null;
          const maeM = Number(r.mae_model);
          const maeB = Number(r.mae_baseline);
          const better = lift != null && lift > 0;

          return (
            <div key={r.station_id} className="table-row">
              {/* Bloc station + barre + % discret */}
              <div className="table-cell">
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "minmax(0,1fr) auto",
                    alignItems: "baseline",
                    gap: 4,
                  }}
                >
                  <div style={{ minWidth: 0 }}>
                    <div
                      style={{
                        fontWeight: 600,
                        whiteSpace: "nowrap",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                      }}
                      title={displayName}
                    >
                      {displayName}
                    </div>
                    <div className="mono" style={{ fontSize: 12, opacity: 0.7 }}>
                      {r.station_id} · n={fmtInt(r.n)}
                    </div>
                  </div>
                  <div
                    className="small"
                    style={{
                      fontWeight: 700,
                      whiteSpace: "nowrap",
                      marginLeft: 2,
                    }}
                  >
                    {fmtPct(liftPct, 1)}
                  </div>
                </div>

                {/* Barre allongée */}
                <div
                  className="bar"
                  style={{
                    height: 5,
                    marginTop: 5,
                    width: "min(75%, 320px)", // ⬅️ barre bien plus longue
                  }}
                >
                  <div
                    className={`bar__fill ${better ? "" : "bar__fill--danger"}`}
                    style={{
                      width: `${Math.max(0, Math.min(100, liftPct ?? 0))}%`,
                    }}
                    aria-hidden
                  />
                </div>
              </div>

              {/* Colonnes chiffrées bien collées */}
              <div className="table-cell table-cell--right" style={{ paddingRight: 2 }}>
                <div style={{ fontWeight: 700 }}>{fmtNum(maeB, 2)}</div>
              </div>
              <div className="table-cell table-cell--right" style={{ paddingRight: 2 }}>
                <div style={{ fontWeight: 700 }}>{fmtNum(maeM, 2)}</div>
              </div>
              <div className="table-cell table-cell--right" style={{ paddingRight: 0, fontWeight: 700 }}>
                {fmtPct(liftPct, 1)}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
