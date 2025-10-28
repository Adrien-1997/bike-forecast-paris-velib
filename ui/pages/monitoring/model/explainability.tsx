// ui/pages/monitoring/model/explainability.tsx
import Head from "next/head";
import { useCallback, useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import type { ScatterData } from "plotly.js";
import type * as Plotly from "plotly.js";
import MonitoringNav from "@/components/monitoring/MonitoringNav";
import LoadingBar, { type LoadingBarStatus } from "@/components/common/LoadingBar";
import KpiBar, { type KpiItem } from "@/components/monitoring/KpiBar";
import { chartConfig, chartLayout } from "@/lib/plotlyTheme";
import HorizonToggle from "@/components/common/HorizonToggle";

import {
  getExplainOverview,
  getExplainResiduals,
  getExplainCalibration,
  getExplainUncertainty,
  type Overview,
  type ResidualsDoc,
  type CalibrationDoc,
  type UncertaintyDoc,
} from "@/lib/services/monitoring/model_explainability";

/* ───────────────── Plotly (client only) ───────────────── */
const Plot = dynamic(() => import("react-plotly.js").then((m) => m.default), {
  ssr: false,
  loading: () => (
    <div style={{ height: 320, display: "grid", placeItems: "center", opacity: 0.7 }}>
      Chargement du graphique…
    </div>
  ),
});

/* ───────────────── Helpers ───────────────── */
function ok<T>(r: PromiseSettledResult<T>): T | null {
  return r.status === "fulfilled" ? r.value : null;
}
const fmtNum = (x?: number | null, d = 2) => (Number.isFinite(Number(x)) ? Number(x).toFixed(d) : "—");
const fmtInt = (x?: number | null) => (Number.isFinite(Number(x)) ? Number(x).toLocaleString("fr-FR") : "—");
const fmtDate = (s?: string | null) => (s ? s : "—");

function safeMinMax(arrs: number[][]): { min: number; max: number } {
  let min = Number.POSITIVE_INFINITY, max = Number.NEGATIVE_INFINITY;
  for (const arr of arrs) for (const v of arr) if (Number.isFinite(v)) { if (v < min) min = v; if (v > max) max = v; }
  if (!Number.isFinite(min)) min = 0;
  if (!Number.isFinite(max)) max = 0;
  return { min, max };
}

/* ───────────────── Horizon (SSR-safe) ───────────────── */
function useQueryParamH(defaultH = 15): [number, (h: number) => void, boolean] {
  const [h, setH] = useState<number>(defaultH);
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(true);
    try {
      const u = new URL(window.location.href);
      const qh = Number(u.searchParams.get("h"));
      if (Number.isFinite(qh) && qh > 0) setH(qh);
    } catch {}
  }, []);
  useEffect(() => {
    if (!mounted) return;
    try {
      const u = new URL(window.location.href);
      if (u.searchParams.get("h") !== String(h)) {
        u.searchParams.set("h", String(h));
        window.history.replaceState({}, "", u.toString());
      }
    } catch {}
  }, [h, mounted]);
  const setter = useCallback((next: number) => setH(next), []);
  return [h, setter, mounted];
}

/* ───────────────── Page ───────────────── */
export default function ModelExplainabilityPage() {
  const [h, setH, mounted] = useQueryParamH(15);

  const [overview, setOverview] = useState<Overview | null>(null);
  const [residuals, setResiduals] = useState<ResidualsDoc | null>(null);
  const [calib, setCalib] = useState<CalibrationDoc | null>(null);
  const [unc, setUnc] = useState<UncertaintyDoc | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setLoading(true);
        const [rOverview, rResiduals, rCalib, rUnc] = await Promise.allSettled([
          getExplainOverview(h),
          getExplainResiduals(h),
          getExplainCalibration(h),
          getExplainUncertainty(h),
        ]);
        if (!alive) return;

        setOverview(ok(rOverview));
        setResiduals(ok(rResiduals));
        setCalib(ok(rCalib));
        setUnc(ok(rUnc));

        const failures = [rOverview, rResiduals, rCalib, rUnc].filter(
          (r): r is PromiseRejectedResult => r.status === "rejected",
        );
        setError(failures.length ? "Une ou plusieurs requêtes ont échoué." : null);
      } catch (e: any) {
        if (alive) setError(String(e?.message ?? e));
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, [h]);

  const generatedAt =
    overview?.generated_at || residuals?.generated_at || calib?.generated_at || unc?.generated_at || undefined;

  const barStatus: LoadingBarStatus = loading ? "loading" : error ? "error" : "success";

  /* ───────── KPI BAR ───────── */
  const fmtIntSafe = (v: KpiItem["value"]) => (Number.isFinite(Number(v)) ? Number(v).toLocaleString("fr-FR") : "—");
  const fmtFixed = (d = 3) => (v: KpiItem["value"]) => (Number.isFinite(Number(v)) ? Number(v).toFixed(d) : "—");
  const fmtYesNo = (v: KpiItem["value"]) => (Number(v) ? "Oui" : "Non");

  const kpiItems: KpiItem[] = useMemo(() => [
    { label: "Stations (perf)", value: overview?.perf_stations ?? null, fmt: fmtIntSafe },
    { label: "Lignes (n)",      value: overview?.perf_rows ?? null,      fmt: fmtIntSafe },
    { label: "Prédictions",     value: overview?.has_y_pred ? 1 : 0,     fmt: fmtYesNo },
    { label: "Incertitude",     value: overview?.has_uncertainty ? 1 : 0,fmt: fmtYesNo },
    { label: "β global",        value: calib?.fit?.beta ?? null,         fmt: fmtFixed(3) },
    { label: "α global",        value: calib?.fit?.alpha ?? null,        fmt: fmtFixed(3) },
  ], [overview, calib]);

  const metaParts: string[] = [];
  if (overview?.schema_version != null) metaParts.push(`Schema v${overview.schema_version}`);
  if (overview?.ts_min_perf || overview?.ts_max_perf)
    metaParts.push(`Intervalle: ${fmtDate(overview?.ts_min_perf)} → ${fmtDate(overview?.ts_max_perf)} (UTC)`);
  if (generatedAt) metaParts.push(`generated ${generatedAt}`);
  const metaLine = metaParts.join(" · ");

  /* ───────── Graph data ───────── */
  const histData: Partial<Plotly.PlotData>[] = useMemo(() => {
    const bins = residuals?.hist ?? [];
    if (!bins.length) return [];
    const centers = bins.map((b) => (b.bin_left + b.bin_right) / 2);
    const counts = bins.map((b) => b.count);
    return [{
      x: centers,
      y: counts,
      type: "bar" as const,
      name: "Résidus",
      hovertemplate: "x=%{x:.2f}<br>n=%{y}<extra></extra>",
    }];
  }, [residuals]);

  const qqData = useMemo<Partial<ScatterData>[]>(() => {
    const th = residuals?.qq?.th ?? [];
    const emp = residuals?.qq?.emp ?? [];
    if (!th.length || th.length !== emp.length) return [];
    const { min, max } = safeMinMax([th, emp]);
    const line: Partial<ScatterData> = { x: [min, max], y: [min, max], type: "scatter", mode: "lines", name: "y = x", hoverinfo: "none" };
    const pts: Partial<ScatterData> = { x: th, y: emp, type: "scatter", mode: "markers", name: "Quantiles", hovertemplate: "th=%{x:.2f}<br>emp=%{y:.2f}<extra></extra>" };
    return [line, pts];
  }, [residuals]);

  const acfData: Partial<Plotly.PlotData>[] = useMemo(() => {
    const acf = residuals?.acf ?? [];
    if (!acf.length) return [];
    const x = Array.from({ length: acf.length }, (_, i) => i);
    return [{ x, y: acf, type: "bar" as const, name: "ACF", hovertemplate: "lag=%{x}<br>ρ=%{y:.2f}<extra></extra>" }];
  }, [residuals]);

  const heteroData: Partial<Plotly.PlotData>[] = useMemo(() => {
    const rows = residuals?.hetero ?? [];
    if (!rows.length) return [];
    return [{
      x: rows.map((r) => r.quantile),
      y: rows.map((r) => r.mae),
      type: "bar" as const,
      name: "MAE",
      hovertemplate: "quantile=%{x}<br>MAE=%{y:.2f}<extra></extra>",
    }];
  }, [residuals]);

  const calibBinning = useMemo<Partial<ScatterData>[]>(() => {
    const rows = calib?.binned ?? [];
    if (!rows.length) return [];
    const xp = rows.map((r) => r.y_pred_mean);
    const yt = rows.map((r) => r.y_true_mean);
    const { min, max } = safeMinMax([xp, yt]);
    const line: Partial<ScatterData> = { x: [min, max], y: [min, max], type: "scatter", mode: "lines", name: "y = x", hoverinfo: "none" };
    const pts: Partial<ScatterData> = { x: xp, y: yt, type: "scatter", mode: "markers", name: "Binned means", hovertemplate: "E[ŷ]=%{x:.2f}<br>E[y]=%{y:.2f}<extra></extra>" };
    return [line, pts];
  }, [calib]);

  const betaByHour = useMemo<Partial<ScatterData>[]>(() => {
    const rows = calib?.by_hour ?? [];
    if (!rows.length) return [];
    return [{
      x: rows.map((r) => r.hour),
      y: rows.map((r) => (Number.isFinite(Number(r.beta)) ? Number(r.beta) : null)),
      type: "scatter",
      mode: "lines+markers",
      name: "β (pente) vs heure",
      hovertemplate: "h=%{x}<br>β=%{y:.3f}<extra></extra>",
    }];
  }, [calib]);

  const relErrBars: Partial<Plotly.PlotData>[] = useMemo(() => {
    const rows = calib?.rel_error_levels ?? [];
    if (!rows.length) return [];
    return [{
      x: rows.map((r) => r.level),
      y: rows.map((r) => r.mape_like * 100),
      type: "bar" as const,
      name: "MAPE-like (%)",
      hovertemplate: "niveau=%{x}<br>% err.=%{y:.1f}%<extra></extra>",
    }];
  }, [calib]);

  const stationBias = useMemo(() => {
    const rows = calib?.bias_by_station ?? [];
    if (!rows.length) return { tops: [], bots: [] as CalibrationDoc["bias_by_station"] };
    const withN = rows.filter((r) => Number(r.n) >= 30);
    const sorted = [...withN].sort((a, b) => Math.abs(Number(b.bias ?? 0)) - Math.abs(Number(a.bias ?? 0)));
    return { tops: sorted.slice(0, 20), bots: sorted.slice(-20).reverse() };
  }, [calib]);

  const subtitleText = useMemo(
    () => `Résidus, QQ, ACF, hétéroscédasticité, calibration & incertitude (h=${mounted ? h : 15} min)`,
    [h, mounted],
  );

  /* ───────────────── Render ───────────────── */
  return (
    <div className="monitoring">
      <Head>
        <title>Monitoring — Model / Explainability</title>
        <meta name="description" content="Résidus, QQ, ACF, hétéroscédasticité, calibration et incertitude." />
        <style
          dangerouslySetInnerHTML={{
            __html: `
            /* Utilitaires visuels partagés */
            .monitoring .bar{height:6px;border-radius:999px;background:#111827;position:relative;overflow:hidden}
            .monitoring .bar__fill{height:100%;background:var(--ok)}
            .monitoring .bar__fill--danger{background:#ef4444}
            .monitoring .figure-note{opacity:.75;margin-top:6px}

            /* Tableaux à la "overview" */
            .monitoring .table-grid{display:grid;grid-template-columns:var(--cols, 1fr);align-items:center}
            .monitoring .table-head{
              padding:8px 6px;font-size:12px;text-transform:uppercase;color:var(--text-dim);
              border-bottom:1px solid color-mix(in srgb, var(--text) 35%, transparent);
              background:color-mix(in srgb, var(--panel) 88%, transparent);
              backdrop-filter:saturate(140%) blur(6px);-webkit-backdrop-filter:saturate(140%) blur(6px)
            }
            .monitoring .table-head--sticky{position:sticky;top:0;z-index:1}
            .monitoring .table-row{display:contents}
            .monitoring .table-cell{padding:10px 6px;border-bottom:1px solid color-mix(in srgb, var(--text) 20%, transparent)}
            .monitoring .table-cell--right{text-align:right}
            `,
          }}
        />
      </Head>

      <main className="page" style={{ paddingTop: "calc(var(--header-h, 70px) + 12px)" }}>
        <MonitoringNav
          title="Model — Explainability"
          subtitle={subtitleText}
          generatedAt={generatedAt}
          extraActions={[{ label: "Performance", href: "/monitoring/model/performance" }]}
        />

        <LoadingBar status={barStatus} />

        {/* Toolbar / Controls */}
        <section className="mt-3">
          <div className="card" style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 10 }}>
            <HorizonToggle
              value={h}
              onChange={(v: number) => setH(v)}
              leftValue={15}
              rightValue={60}
              leftLabel="15 min"
              rightLabel="60 min"
              ariaLabel="Choix de l’horizon de prévision"
            />
          </div>
        </section>

        {/* KPIs + meta */}
        <section className="mt-4">
          <h2>Résumé — KPIs (h={h} min)</h2>
          <KpiBar items={kpiItems} dense />
          {metaLine && <div className="kpi-bar-meta">{metaLine}</div>}
        </section>

        {/* ───────── Bloc 1 : Résidus (graphiques) ───────── */}
        <section className="mt-6">
          <h2>Résidus</h2>
          <p className="muted small" style={{ marginTop: -6 }}>
            Distribution, normalité et mémoire des erreurs. Idéalement : centré, faible ACF, QQ proche de la diagonale.
          </p>

          <div className="grid-2">
            <div className="card plot-card">
              <h3>Histogramme</h3>
              {histData.length ? (
                <Plot
                  data={histData as Plotly.Data[]}
                  layout={chartLayout({
                    height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 40 },
                    xaxis: { title: { text: "Erreur (y_true - y_pred)" } },
                    yaxis: { title: { text: "Comptes" } },
                  })}
                  config={chartConfig}
                  className="plot plot--sm"
                />
              ) : <div className="empty">—</div>}
            </div>

            <div className="card plot-card">
              <h3>QQ-plot (normalisé)</h3>
              {qqData.length ? (
                <Plot
                  data={qqData as Plotly.Data[]}
                  layout={chartLayout({
                    height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 40 },
                    xaxis: { title: { text: "Théorique" } },
                    yaxis: { title: { text: "Empirique" } },
                    hovermode: "closest",
                  })}
                  config={chartConfig}
                  className="plot plot--sm"
                />
              ) : <div className="empty">—</div>}
            </div>
          </div>

          {/* ⬇️ Petites lignes (hors cartes) */}
          <div className="figure-note small">Histogramme : centres de bacs vs fréquence des résidus.</div>
          <div className="figure-note small">QQ-plot : une bonne calibration suit la diagonale y=x ; écarts = queues lourdes / biais.</div>
        </section>

        {/* ───────── Bloc 2 : ACF & Hétéro (graphiques) ───────── */}
        <section className="mt-6">
          <h2>Structure des erreurs</h2>
          <p className="muted small" style={{ marginTop: -6 }}>
            Autocorrélations et variabilité par niveau de charge.
          </p>

          <div className="grid-2">
            <div className="card plot-card">
              <h3>ACF du résidu (lag 5 min)</h3>
              {acfData.length ? (
                <Plot
                  data={acfData as Plotly.Data[]}
                  layout={chartLayout({
                    height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 36 },
                    xaxis: { title: { text: "Lag (5 min)" } },
                    yaxis: { title: { text: "Corrélation" } },
                  })}
                  config={chartConfig}
                  className="plot plot--sm"
                />
              ) : <div className="empty">—</div>}
            </div>

            <div className="card plot-card">
              <h3>Hétéroscédasticité</h3>
              {heteroData.length ? (
                <Plot
                  data={heteroData as Plotly.Data[]}
                  layout={chartLayout({
                    height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 36 },
                    xaxis: { title: { text: "Quantiles(y_true)" }, tickangle: -30 },
                    yaxis: { title: { text: "MAE" } },
                  })}
                  config={chartConfig}
                  className="plot plot--sm"
                />
              ) : <div className="empty">—</div>}
            </div>
          </div>

          {/* ⬇️ Petites lignes */}
          <div className="figure-note small">ACF : les autocorrélations résiduelles devraient être proches de 0.</div>
          <div className="figure-note small">Hétéroscédasticité : variabilité de l’erreur selon le niveau de charge.</div>
        </section>

        {/* ───────── Bloc 3 : Tableau — Épisodes d’erreurs (seul, pas à côté d’un graphe) ───────── */}
        <section className="mt-6">
          <h2>Épisodes d’erreurs (|résidu| ≥ 4)</h2>
          <div className="card plot-card">
            {residuals?.episodes?.length ? (
              <div className="table-scroll" style={{ overflowX: "hidden" }}>
                <div className="table-grid" style={{ ["--cols" as any]: "minmax(0,1fr)", minWidth: 0 }}>
                  <div className="table-head table-head--sticky">Station</div>
                  {residuals.episodes.slice(0, 80).map((r) => {
                    const widthPct = Math.min(100, Number(r.max_run) * 4);
                    return (
                      <div className="table-row" key={r.station_id}>
                        <div className="table-cell">
                          <div
                            style={{
                              display: "grid",
                              gridTemplateColumns: "minmax(0,1fr) auto",
                              alignItems: "baseline",
                              gap: 8,
                            }}
                          >
                            <div>
                              <div
                                style={{
                                  fontWeight: 600,
                                  whiteSpace: "nowrap",
                                  overflow: "hidden",
                                  textOverflow: "ellipsis",
                                }}
                                title={`#${r.station_id}`}
                              >
                                #{r.station_id}
                              </div>
                              <div className="mono" style={{ fontSize: 12, opacity: 0.7 }}>
                                n={fmtInt(r.n)} · max run={fmtInt(r.max_run)}
                              </div>
                            </div>
                            <div className="small" style={{ fontWeight: 700 }}>
                              {fmtInt(r.max_run)}
                            </div>
                          </div>

                          <div className="bar" style={{ height: 5, marginTop: 6, width: "min(52%, 240px)" }}>
                            <div className="bar__fill bar__fill--danger" style={{ width: `${widthPct}%` }} aria-hidden />
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : <div className="empty">—</div>}
          </div>
          {/* ⬇️ Petite ligne hors carte/tableau */}
          <div className="figure-note small">Un épisode = séquence continue d’erreurs élevées pour une station.</div>
        </section>

        {/* ───────── Bloc 4 : Calibration (graphiques) ───────── */}
        <section className="mt-6">
          <h2>Calibration</h2>
          <p className="muted small" style={{ marginTop: -6 }}>
            Conformité moyenne ŷ → y, stabilité par heure et profils d’erreur relatifs.
          </p>

          <div className="card plot-card">
            <h3>Binned means (y_pred → y_true)</h3>
            {calibBinning.length ? (
              <Plot
                data={calibBinning as Plotly.Data[]}
                layout={chartLayout({
                  height: 320,
                  margin: { l: 54, r: 10, t: 10, b: 40 },
                  xaxis: { title: { text: "E[y_pred]" } },
                  yaxis: { title: { text: "E[y_true]" } },
                })}
                config={chartConfig}
                className="plot plot--sm"
              />
            ) : <div className="empty">—</div>}
          </div>
          <div className="figure-note small">
            y ≈ α + β·ŷ — α={fmtNum(calib?.fit?.alpha, 3)}, β={fmtNum(calib?.fit?.beta, 3)} · n={fmtInt(calib?.fit?.n ?? null)}
          </div>

          <div className="grid-2 mt-4">
            <div className="card plot-card">
              <h3>β par heure (locale)</h3>
              {betaByHour.length ? (
                <Plot
                  data={betaByHour as Plotly.Data[]}
                  layout={chartLayout({
                    height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 36 },
                    xaxis: { title: { text: "Heure" } },
                    yaxis: { title: { text: "β" } },
                  })}
                  config={chartConfig}
                  className="plot plot--sm"
                />
              ) : <div className="empty">—</div>}
            </div>

            <div className="card plot-card">
              <h3>Erreur relative par niveau</h3>
              {relErrBars.length ? (
                <Plot
                  data={relErrBars as Plotly.Data[]}
                  layout={chartLayout({
                    height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 36 },
                    xaxis: { title: { text: "Niveau (quantiles y_true)" } },
                    yaxis: { title: { text: "MAPE-like (%)" } },
                  })}
                  config={chartConfig}
                  className="plot plot--sm"
                />
              ) : <div className="empty">—</div>}
            </div>
          </div>

          <div className="figure-note small">Variations de β par heure : indices de biais systématiques horaires.</div>
          <div className="figure-note small">MAPE-like : erreur relative moyenne par tranche de y_true.</div>
        </section>

        {/* ───────── Bloc 5 : Tableau — Biais par station (seul) ───────── */}
        <section className="mt-6">
          <h2>Stations — biais moyen</h2>
          <div className="card plot-card">
            {calib?.bias_by_station?.length ? (
              <div className="table-scroll" style={{ overflowX: "hidden" }}>
                <div className="table-grid" style={{ ["--cols" as any]: "minmax(0,1fr)", minWidth: 0 }}>
                  <div className="table-head table-head--sticky">Station</div>
                  {stationBias.tops.slice(0, 30).map((r) => {
                    const absBias = Math.abs(Number(r.bias ?? 0));
                    const widthPct = Math.min(100, absBias * 5);
                    return (
                      <div className="table-row" key={r.station_id}>
                        <div className="table-cell">
                          <div
                            style={{
                              display: "grid",
                              gridTemplateColumns: "minmax(0,1fr) auto auto",
                              alignItems: "baseline",
                              gap: 8,
                            }}
                          >
                            <div>
                              <div
                                style={{
                                  fontWeight: 600,
                                  whiteSpace: "nowrap",
                                  overflow: "hidden",
                                  textOverflow: "ellipsis",
                                }}
                                title={r.name ?? `#${r.station_id}`}
                              >
                                {r.name ?? `#${r.station_id}`}
                              </div>
                              <div className="mono" style={{ fontSize: 12, opacity: 0.7 }}>
                                #{r.station_id} · n={fmtInt(r.n)}
                              </div>
                            </div>
                            <div className="small" style={{ fontWeight: 700 }}>{fmtNum(r.bias, 2)}</div>
                            <div className="small" style={{ fontWeight: 700, opacity: .85 }}>|b|</div>
                          </div>

                          <div className="bar" style={{ height: 5, marginTop: 6, width: "min(52%, 240px)" }}>
                            <div className="bar__fill bar__fill--danger" style={{ width: `${widthPct}%` }} aria-hidden />
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : <div className="empty">—</div>}
          </div>
          <div className="figure-note small">Tri sur |biais| ; filtre n ≥ 30 pour réduire le bruit.</div>
        </section>

        {/* ───────── Bloc 6 : Incertitude (carte + note) ───────── */}
        <section className="mt-6" style={{ marginBottom: 40 }}>
          <h2>Incertitude</h2>
          <p className="muted small" style={{ marginTop: -6 }}>
            Cohérence des intervalles prédictifs (couverture empirique vs nominal).
          </p>

          <div className="card plot-card">
            {unc?.coverage ? (
              <div className="row" style={{ alignItems: "center", gap: 16, flexWrap: "wrap" }}>
                <div className="kpi">
                  <div className="kpi__label small muted">Coverage empirique</div>
                  <div className="kpi__value" style={{ fontWeight: 700 }}>
                    {Number.isFinite(Number(unc.coverage.empirical)) ? `${(Number(unc.coverage.empirical) * 100).toFixed(1)}%` : "—"}
                  </div>
                </div>
                <div className="kpi">
                  <div className="kpi__label small muted">Échantillon</div>
                  <div className="kpi__value" style={{ fontWeight: 700 }}>{fmtInt(unc.coverage.n)}</div>
                </div>
                {(unc.method || Number.isFinite(Number(unc.nominal))) && (
                  <div className="small muted">
                    {unc.method ? `Méthode: ${unc.method}` : ""}
                    {Number.isFinite(Number(unc.nominal)) ? ` · nominal=${(Number(unc.nominal) * 100).toFixed(0)}%` : ""}
                  </div>
                )}
              </div>
            ) : <div className="empty">Aucune borne d’incertitude détectée (colonnes yhat_lo / yhat_hi).</div>}
          </div>

          <div className="figure-note small">
            Couverture = part(y_true ∈ [yhat_lo, yhat_hi]). Attendue ≈ 90–95% selon l’intervalle.
          </div>
        </section>
      </main>
    </div>
  );
}
