// ui/pages/monitoring/model/explainability.tsx
import Head from "next/head";
import { useEffect, useMemo, useState } from "react";
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
  getExplainFeatureImportance,
  type Overview,
  type ResidualsDoc,
  type CalibrationDoc,
  type UncertaintyDoc,
  type FeatureImportanceDoc,
} from "@/lib/services/monitoring/model_explainability";

// ⬇️ Index stations (JSON local: public/data/stations.index.json)
import { loadStationsIndexFromArrayJson, type StationMeta } from "@/lib/local/stationsIndex";

/* ───────────────── Plotly (client only) ───────────────── */
const Plot = dynamic(() => import("react-plotly.js").then((m) => m.default), {
  ssr: false,
  loading: () => (
    <div className="empty" style={{ minHeight: 320 }}>
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
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const safeMinMax = (arrs: number[][]) => {
  let min = Number.POSITIVE_INFINITY, max = Number.NEGATIVE_INFINITY;
  for (const a of arrs) for (const v of a) if (Number.isFinite(v)) { if (v < min) min = v; if (v > max) max = v; }
  if (!Number.isFinite(min)) min = 0;
  if (!Number.isFinite(max)) max = 0;
  return { min, max };
};

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
      u.searchParams.set("h", String(h));
      window.history.replaceState({}, "", u.toString());
    } catch {}
  }, [h, mounted]);
  return [h, (v: number) => setH(v), mounted];
}

/* ───────────────── Page ───────────────── */
export default function ModelExplainabilityPage() {
  const [h, setH, mounted] = useQueryParamH(15);

  const [overview, setOverview] = useState<Overview | null>(null);
  const [residuals, setResiduals] = useState<ResidualsDoc | null>(null);
  const [calib, setCalib] = useState<CalibrationDoc | null>(null);
  const [unc, setUnc] = useState<UncertaintyDoc | null>(null);
  const [fiDoc, setFiDoc] = useState<FeatureImportanceDoc | null>(null);

  // Index stations: id → meta
  const [stationsIdx, setStationsIdx] = useState<Record<string, StationMeta>>({});

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Charge l’index des stations au montage
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const idx = await loadStationsIndexFromArrayJson("/data/stations.index.json").catch(() => ({}));
        if (!alive) return;
        setStationsIdx(idx as Record<string, StationMeta>);
      } catch {}
    })();
    return () => { alive = false; };
  }, []);

  // Charge les docs explain pour l’horizon actif
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setLoading(true);
        const res = await Promise.allSettled([
          getExplainOverview(h),
          getExplainResiduals(h),
          getExplainCalibration(h),
          getExplainUncertainty(h),
          getExplainFeatureImportance(h),
        ]);
        if (!alive) return;

        setOverview(ok(res[0]));
        setResiduals(ok(res[1]));
        setCalib(ok(res[2]));
        setUnc(ok(res[3]));
        setFiDoc(ok(res[4]));

        const failures = res.filter((r): r is PromiseRejectedResult => r.status === "rejected");
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
    fiDoc?.generated_at ||
    overview?.generated_at ||
    residuals?.generated_at ||
    calib?.generated_at ||
    unc?.generated_at ||
    undefined;

  const barStatus: LoadingBarStatus = loading ? "loading" : error ? "error" : "success";

  // Dictionnaire id → nom (noms non vides, gère ids sans zéros à gauche)
  const nameIndex = useMemo(() => {
    const rec: Record<string, string> = {};
    for (const [id, meta] of Object.entries(stationsIdx)) {
      const nm = (meta?.name ?? "").trim();
      if (!nm) continue;
      rec[id] = nm;
      const noZeros = id.replace(/^0+/, "");
      if (!(noZeros in rec)) rec[noZeros] = nm;
    }
    return rec;
  }, [stationsIdx]);

  /* ───────── KPI BAR ───────── */
  const kpiItems: KpiItem[] = useMemo(() => {
    const yesNo = (v?: boolean | null) => (v ? "Oui" : "Non");
    return [
      { label: "Stations (perf)", value: overview?.perf_stations ?? null, fmt: (v) => fmtInt(Number(v)) },
      { label: "Lignes (n)", value: overview?.perf_rows ?? null, fmt: (v) => fmtInt(Number(v)) },
      { label: "Prédictions", value: overview?.has_y_pred ? 1 : 0, fmt: (v) => yesNo(Boolean(v)) },
      { label: "Incertitude", value: overview?.has_uncertainty ? 1 : 0, fmt: (v) => yesNo(Boolean(v)) },
      { label: "β global", value: calib?.fit?.beta ?? null, fmt: (v) => fmtNum(Number(v), 3) },
      { label: "α global", value: calib?.fit?.alpha ?? null, fmt: (v) => fmtNum(Number(v), 3) },
    ];
  }, [overview, calib]);

  const subtitleText = useMemo(
    () => `Importance des features, résidus, QQ, ACF, hétéroscédasticité, calibration & incertitude (h=${mounted ? h : 15} min)`,
    [h, mounted],
  );

  /* ───────── Feature importance (barres horizontales) ───────── */
  const fiBarData: Partial<Plotly.PlotData>[] = useMemo(() => {
    if (!fiDoc?.rows?.length) return [];
    const rows = [...fiDoc.rows]
      .filter((r) => Number.isFinite(Number(r.importance)))
      .sort((a, b) => Number(b.importance ?? 0) - Number(a.importance ?? 0));

    if (!rows.length) return [];

    const x = rows.map((r) => Number(r.importance));
    const y = rows.map((r) => r.feature);
    const err = rows.map((r) => (Number.isFinite(Number(r.std)) ? Number(r.std) : 0));

    return [
      {
        x: x.reverse(),
        y: y.reverse(),
        type: "bar" as const,
        orientation: "h",
        name: "importance",
        error_x: {
          type: "data",
          array: err.reverse(),
          visible: err.some((v) => v > 0),
        },
        hovertemplate: "%{y}<br>importance=%{x:.4f}" + (err.some((v) => v > 0) ? "<br>±%{error_x.array:.4f}" : "") + "<extra></extra>",
      },
    ];
  }, [fiDoc]);

  /* ───────── Graph data (résidus / calib / incertitude) ───────── */
  const histData: Partial<Plotly.PlotData>[] = useMemo(() => {
    const bins = residuals?.hist ?? [];
    if (!bins.length) return [];
    const x = bins.map((b) => (b.bin_left + b.bin_right) / 2);
    const y = bins.map((b) => b.count);
    return [{ x, y, type: "bar" as const, name: "Résidus", hovertemplate: "x=%{x:.2f}<br>n=%{y}<extra></extra>" }];
  }, [residuals]);

  const qqData = useMemo<Partial<ScatterData>[]>(() => {
    const th = residuals?.qq?.th ?? [];
    const emp = residuals?.qq?.emp ?? [];
    if (!th.length || th.length !== emp.length) return [];
    const { min, max } = safeMinMax([th, emp]);
    return [
      { x: [min, max], y: [min, max], type: "scatter", mode: "lines", name: "y = x", hoverinfo: "none" },
      { x: th, y: emp, type: "scatter", mode: "markers", name: "Quantiles", hovertemplate: "th=%{x:.2f}<br>emp=%{y:.2f}<extra></extra>" },
    ];
  }, [residuals]);

  const acfData: Partial<Plotly.PlotData>[] = useMemo(() => {
    const a = residuals?.acf ?? [];
    if (!a.length) return [];
    const x = Array.from({ length: a.length }, (_, i) => i);
    return [{ x, y: a, type: "bar" as const, name: "ACF", hovertemplate: "lag=%{x}<br>ρ=%{y:.2f}<extra></extra>" }];
  }, [residuals]);

  const heteroData: Partial<Plotly.PlotData>[] = useMemo(() => {
    const rows = residuals?.hetero ?? [];
    if (!rows.length) return [];
    return [{
      x: rows.map((r) => r.quantile),
      y: rows.map((r) => r.mae),
      type: "bar" as const,
      name: "MAE",
      hovertemplate: "q=%{x}<br>MAE=%{y:.2f}<extra></extra>",
    }];
  }, [residuals]);

  const calibBinning = useMemo<Partial<ScatterData>[]>(() => {
    const rows = calib?.binned ?? [];
    if (!rows.length) return [];
    const xp = rows.map((r) => r.y_pred_mean);
    const yt = rows.map((r) => r.y_true_mean);
    const { min, max } = safeMinMax([xp, yt]);
    return [
      { x: [min, max], y: [min, max], type: "scatter", mode: "lines", name: "y = x", hoverinfo: "none" },
      { x: xp, y: yt, type: "scatter", mode: "markers", name: "Binned means", hovertemplate: "E[ŷ]=%{x:.2f}<br>E[y]=%{y:.2f}<extra></extra>" },
    ];
  }, [calib]);

  const betaByHour = useMemo<Partial<ScatterData>[]>(() => {
    const rows = calib?.by_hour ?? [];
    if (!rows.length) return [];
    return [{
      x: rows.map((r) => r.hour),
      y: rows.map((r) => (Number.isFinite(Number(r.beta)) ? Number(r.beta) : null)),
      type: "scatter",
      mode: "lines+markers",
      name: "β vs heure",
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
      hovertemplate: "niveau=%{x}<br>%=%{y:.1f}<extra></extra>",
    }];
  }, [calib]);

  /* ───────── Tri stations |biais| ───────── */
  const biasRows = useMemo(() => {
    const rows = calib?.bias_by_station ?? [];
    if (!rows.length) return [];
    const filtered = rows.filter((r) => Number(r.n) >= 30);
    return [...filtered].sort((a, b) => Math.abs(Number(b.bias ?? 0)) - Math.abs(Number(a.bias ?? 0))).slice(0, 30);
  }, [calib]);

  /* ───────── RENDER ───────── */
  return (
    <div className="monitoring">
      <Head>
        <title>Monitoring — Model / Explainability</title>
        <meta name="description" content="Importance des features, résidus, QQ, ACF, hétéroscédasticité, calibration et incertitude." />
        {/* Styles barre si non globaux */}
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
          title="Model — Explainability"
          subtitle={subtitleText}
          generatedAt={generatedAt}
          extraActions={[{ label: "Performance", href: "/monitoring/model/performance" }]}
        />

        <LoadingBar status={barStatus} />

        {/* Toolbar */}
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

        {/* ───────────────── Feature Importance (en premier) ───────────────── */}
        <section className="mt-4">
          <h2>Importance des features</h2>
          <div className="card plot-card">
            <div className="row" style={{ justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <div className="small muted">
                {fiDoc
                  ? `Méthode: ${fiDoc.method} · h=${fiDoc.horizon_min} min · n_rows=${fiDoc.n_rows} · n_features=${fiDoc.n_features}`
                  : "Aucune donnée disponible"}
              </div>
            </div>

            {fiBarData.length ? (
              <Plot
                data={fiBarData as Plotly.Data[]}
                layout={chartLayout({
                  height: 420,
                  margin: { l: 220, r: 10, t: 10, b: 36 },
                  xaxis: { title: { text: "importance (ΔMAE, + grand = + important)" } },
                  yaxis: { automargin: true },
                  hovermode: "closest",
                })}
                config={chartConfig}
                className="plot plot--lg"
              />
            ) : (
              <div className="empty">—</div>
            )}
          </div>
          {!!fiDoc?.notes?.length && (
            <div className="figure-note small">
              {fiDoc.notes.join(" · ")}
            </div>
          )}
        </section>

        {/* KPIs */}
        <section className="mt-6">
          <h2>Résumé — KPIs</h2>
          <KpiBar items={kpiItems} dense />
          {(() => {
            const parts: string[] = [];
            if (overview?.schema_version != null) parts.push(`Schema v${overview.schema_version}`);
            if (overview?.ts_min_perf || overview?.ts_max_perf)
              parts.push(`Intervalle: ${overview?.ts_min_perf ?? "—"} → ${overview?.ts_max_perf ?? "—"} (UTC)`);
            if (typeof (overview as any)?.horizon_min === "number") parts.push(`h=${(overview as any).horizon_min} min`);
            if ((overview as any)?.window_days) parts.push(`fenêtre=${(overview as any).window_days} j`);
            return parts.length ? <div className="kpi-bar-meta">{parts.join(" · ")}</div> : null;
          })()}
        </section>

        {/* Résidus */}
        <section className="mt-6">
          <h2>Résidus</h2>
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
              ) : (
                <div className="empty">—</div>
              )}
            </div>

            <div className="card plot-card">
              <h3>QQ-plot</h3>
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
              ) : (
                <div className="empty">—</div>
              )}
            </div>
          </div>
          <div className="figure-note small">
            Lecture : histogramme des résidus sur la fenêtre courante ; QQ-plot pour vérifier la normalité (la droite y=x indique une gaussienne idéale).
          </div>
        </section>

        {/* ACF / Hétéroscédasticité */}
        <section className="mt-6">
          <h2>Structure des erreurs</h2>
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
              ) : (
                <div className="empty">—</div>
              )}
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
              ) : (
                <div className="empty">—</div>
              )}
            </div>
          </div>
          <div className="figure-note small">
            Lecture : l’ACF montre la dépendance temporelle résiduelle ; l’hétéroscédasticité trace la MAE selon des tranches de y_true.
          </div>
        </section>

        {/* Tableau 1 — Épisodes d’erreurs */}
        <section className="mt-6">
          <h2>Épisodes d’erreurs (|résidu| ≥ 4)</h2>
          <div className="card">
            {residuals?.episodes?.length ? (
              <>
                <div className="table-scroll">
                  <div className="table-grid" style={{ ["--cols" as any]: "minmax(0,1fr) 1fr 78px" }}>
                    <div className="table-head table-head--sticky">Station</div>
                    <div className="table-head table-head--sticky">Durée max (pas)</div>
                    <div className="table-head table-head--sticky">n épisodes</div>

                    {(() => {
                      const rows = residuals.episodes.slice(0, 80);
                      const maxRun = Math.max(...rows.map((r) => Number(r.max_run ?? 0)), 1);

                      return rows.map((r) => {
                        const val = Number(r.max_run ?? 0);
                        const widthPct = clamp((val / maxRun) * 100, 0, 100);
                        const nm = nameIndex[r.station_id] ?? `#${r.station_id}`;
                        return (
                          <div className="table-row" key={r.station_id}>
                            {/* col 1: Station (nom + id) */}
                            <div className="table-cell">
                              <div className="table-cell--ellipsis" style={{ fontWeight: 600 }} title={nm}>
                                {nm}
                              </div>
                              <div className="mono" style={{ fontSize: 12, opacity: 0.7 }}>#{r.station_id} · max run={fmtInt(r.max_run)}</div>
                            </div>

                            {/* col 2: Durée — barre + valeur alignée à droite */}
                            <div className="table-cell">
                              <div style={{ display: "grid", gridTemplateColumns: "1fr 64px", alignItems: "center", gap: 10 }}>
                                <div className="bar">
                                  <div className="bar__fill bar__fill--danger" style={{ width: `${widthPct}%` }} aria-hidden />
                                </div>
                                <div className="table-cell--right" style={{ fontWeight: 700 }}>{fmtInt(val)}</div>
                              </div>
                            </div>

                            {/* col 3: n épisodes */}
                            <div className="table-cell table-cell--right" style={{ fontWeight: 700 }}>
                              {fmtInt(r.n)}
                            </div>
                          </div>
                        );
                      });
                    })()}
                  </div>
                </div>
                <div className="figure-note small">
                  Lecture : pour chaque station, on affiche la plus longue séquence consécutive où |résidu| ≥ 4, normalisée par le maximum du tableau.
                </div>
              </>
            ) : (
              <div className="empty">Aucun épisode détecté.</div>
            )}
          </div>
        </section>

        {/* Calibration (graphiques) */}
        <section className="mt-6">
          <h2>Calibration</h2>

          <div className="card plot-card">
            <h3>Binned means (y_pred → y_true)</h3>
            {calibBinning.length ? (
              <Plot
                data={calibBinning as Plotly.Data[]}
                layout={chartLayout({
                  height: 320,
                  margin: { l: 54, r: 10, t: 10, b: 40 },
                  xaxis: { title: { text: "E[ŷ]" } },
                  yaxis: { title: { text: "E[y]" } },
                })}
                config={chartConfig}
                className="plot plot--sm"
              />
            ) : (
              <div className="empty">—</div>
            )}
          </div>
          <div className="figure-note small">
            Lecture : comparaison des moyennes binned E[ŷ] vs E[y] ; la diagonale indique une calibration parfaite.
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
              ) : (
                <div className="empty">—</div>
              )}
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
              ) : (
                <div className="empty">—</div>
              )}
            </div>
          </div>
          <div className="figure-note small">
            Lecture : β reflète un éventuel biais multiplicatif selon l’heure ; l’erreur relative est agrégée par niveaux (quantiles) de y_true.
          </div>
        </section>

        {/* Tableau 2 — Stations |biais| */}
        <section className="mt-6" style={{ marginBottom: 40 }}>
          <h2>Stations — biais moyen</h2>
          <div className="card">
            {biasRows.length ? (
              <>
                <div className="table-scroll">
                  <div className="table-grid" style={{ ["--cols" as any]: "minmax(0,1fr) 1fr 78px" }}>
                    <div className="table-head table-head--sticky">Station</div>
                    <div className="table-head table-head--sticky">|biais|</div>
                    <div className="table-head table-head--sticky">n</div>

                    {(() => {
                      const maxAbs = Math.max(...biasRows.map((r) => Math.abs(Number(r.bias ?? 0))), 1);
                      return biasRows.map((r) => {
                        const absBias = Math.abs(Number(r.bias ?? 0));
                        const widthPct = clamp((absBias / maxAbs) * 100, 0, 100);
                        const nm = nameIndex[r.station_id] ?? r.name ?? `#${r.station_id}`;
                        return (
                          <div className="table-row" key={r.station_id}>
                            {/* col 1: Station */}
                            <div className="table-cell">
                              <div className="table-cell--ellipsis" style={{ fontWeight: 600 }} title={nm}>
                                {nm}
                              </div>
                              <div className="mono" style={{ fontSize: 12, opacity: 0.7 }}>#{r.station_id}</div>
                            </div>

                            {/* col 2: |biais| — barre + valeur */}
                            <div className="table-cell">
                              <div style={{ display: "grid", gridTemplateColumns: "1fr 64px", alignItems: "center", gap: 10 }}>
                                <div className="bar">
                                  <div className="bar__fill bar__fill--danger" style={{ width: `${widthPct}%` }} aria-hidden />
                                </div>
                                <div className="table-cell--right" style={{ fontWeight: 700 }}>{fmtNum(absBias, 2)}</div>
                              </div>
                            </div>

                            {/* col 3: n */}
                            <div className="table-cell table-cell--right">{fmtInt(r.n)}</div>
                          </div>
                        );
                      });
                    })()}
                  </div>
                </div>
                <div className="figure-note small">
                  Lecture : |biais| = |E[y_true − y_pred]| par station (filtrées avec n ≥ 30) ; barres normalisées par le |biais| max du tableau.
                </div>
              </>
            ) : (
              <div className="empty">Aucune station à afficher.</div>
            )}
          </div>
        </section>

        {/* Incertitude */}
        <section className="mt-6" style={{ marginBottom: 40 }}>
          <h2>Incertitude</h2>
          <div className="card">
            {unc?.coverage ? (
              <>
                <div className="row" style={{ gap: 16, flexWrap: "wrap" }}>
                  <div className="kpi">
                    <div className="kpi__label small">Coverage empirique</div>
                    <div className="kpi__value" style={{ fontWeight: 700 }}>
                      {Number.isFinite(Number(unc.coverage.empirical))
                        ? `${(Number(unc.coverage.empirical) * 100).toFixed(1)}%`
                        : "—"}
                    </div>
                  </div>
                  <div className="kpi">
                    <div className="kpi__label small">Échantillon</div>
                    <div className="kpi__value" style={{ fontWeight: 700 }}>{fmtInt(unc.coverage.n)}</div>
                  </div>
                  <div className="small muted" style={{ alignSelf: "center" }}>
                    {unc.method ? `Méthode: ${unc.method}` : ""}{Number.isFinite(Number(unc.nominal)) ? ` · nominal=${(Number(unc.nominal) * 100).toFixed(0)}%` : ""}
                  </div>
                </div>
                <div className="figure-note small">
                  Lecture : proportion observée des y_true dans l’intervalle prédictif. La valeur “nominal” est la cible théorique.
                </div>
              </>
            ) : (
              <div className="empty">Aucune borne d’incertitude détectée.</div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
