// ui/pages/monitoring/model/explainability.tsx
import Head from "next/head";
import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import type { ScatterData } from "plotly.js";
import type * as Plotly from "plotly.js";
import MonitoringNav from "@/components/monitoring/MonitoringNav";
import LoadingBar, { type LoadingBarStatus } from "@/components/common/LoadingBar";
import KpiBar from "@/components/monitoring/KpiBar";
import { chartConfig, chartLayout } from "@/lib/plotlyTheme";

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
  loading: () => <div className="empty">Chargement du graphique…</div>,
});

/* ───────────────── Helpers ───────────────── */
function ok<T>(r: PromiseSettledResult<T>): T | null {
  return r.status === "fulfilled" ? r.value : null;
}
const fmtNum = (x?: number | null, d = 2) => (Number.isFinite(Number(x)) ? Number(x).toFixed(d) : "—");
const fmtInt = (x?: number | null) => (Number.isFinite(Number(x)) ? Number(x).toLocaleString("fr-FR") : "—");
const fmtYesNo = (b?: any) => (b ? "Oui" : "Non");
const fmtDate = (s?: string | null) => (s ? s : "—");
/** Avoid `Math.min/max(...bigArray)` call stack blow-ups */
function safeMinMax(arrs: number[][]): { min: number; max: number } {
  let min = Number.POSITIVE_INFINITY,
    max = Number.NEGATIVE_INFINITY;
  for (const arr of arrs) {
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      if (Number.isFinite(v)) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
  }
  if (!Number.isFinite(min)) min = 0;
  if (!Number.isFinite(max)) max = 0;
  return { min, max };
}

/* ───────────────── Page ───────────────── */
export default function ModelExplainabilityPage() {
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
          getExplainOverview(),
          getExplainResiduals(),
          getExplainCalibration(),
          getExplainUncertainty(),
        ]);
        if (!alive) return;

        setOverview(ok(rOverview));
        setResiduals(ok(rResiduals));
        setCalib(ok(rCalib));
        setUnc(ok(rUnc));

        // Erreur si échec partiel (aligné autres pages)
        const results = [rOverview, rResiduals, rCalib, rUnc];
        const failures = results.filter((r): r is PromiseRejectedResult => r.status === "rejected");
        if (failures.length > 0) {
          const msg =
            failures
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
    return () => {
      alive = false;
    };
  }, []);

  const generatedAt =
    overview?.generated_at ||
    residuals?.generated_at ||
    calib?.generated_at ||
    unc?.generated_at ||
    undefined;

  // Barre d’état unifiée
  const barStatus: LoadingBarStatus = loading ? "loading" : error ? "error" : "success";

  /* ---------------- KPI BAR (dense + meta) ---------------- */
  const kpiItems = useMemo(
    () => [
      { label: "Stations (perf)", value: fmtInt(overview?.perf_stations) },
      { label: "Lignes (n)", value: fmtInt(overview?.perf_rows) },
      { label: "Prédictions", value: fmtYesNo(overview?.has_y_pred) },
      { label: "Incertitude", value: fmtYesNo(overview?.has_uncertainty) },
      {
        label: "β global",
        value: calib?.fit?.beta ?? null,
        fmt: (v: any) => (Number.isFinite(Number(v)) ? Number(v).toFixed(3) : "—"),
      },
      {
        label: "α global",
        value: calib?.fit?.alpha ?? null,
        fmt: (v: any) => (Number.isFinite(Number(v)) ? Number(v).toFixed(3) : "—"),
      },
    ],
    [overview, calib]
  );

  const metaParts: string[] = [];
  if (overview?.schema_version != null) metaParts.push(`Schema v${overview.schema_version}`);
  if (overview?.ts_min_perf || overview?.ts_max_perf)
    metaParts.push(`Intervalle: ${fmtDate(overview?.ts_min_perf)} → ${fmtDate(overview?.ts_max_perf)} (UTC)`);
  if (generatedAt) metaParts.push(`generated ${generatedAt}`);
  const metaLine = metaParts.join(" · ");

  /* ---------------- Charts ---------------- */
  const histData: Partial<Plotly.PlotData>[] = useMemo(() => {
    const bins = residuals?.hist ?? [];
    if (!bins.length) return [];
    const centers = bins.map((b) => (b.bin_left + b.bin_right) / 2);
    const counts = bins.map((b) => b.count);
    return [{ x: centers, y: counts, type: "bar" as const, name: "Résidus" }];
  }, [residuals]);

  const qqData = useMemo<Partial<ScatterData>[]>(() => {
    const th = residuals?.qq?.th ?? [];
    const emp = residuals?.qq?.emp ?? [];
    if (!th.length || th.length !== emp.length) return [];
    const { min, max } = safeMinMax([th, emp]);
    const line: Partial<ScatterData> = {
      x: [min, max],
      y: [min, max],
      type: "scatter",
      mode: "lines",
      name: "y = x",
      hoverinfo: "none",
    };
    const pts: Partial<ScatterData> = { x: th, y: emp, type: "scatter", mode: "markers", name: "Quantiles" };
    return [line, pts];
  }, [residuals]);

  const acfData: Partial<Plotly.PlotData>[] = useMemo(() => {
    const acf = residuals?.acf ?? [];
    if (!acf.length) return [];
    const x = Array.from({ length: acf.length }, (_, i) => i);
    return [{ x, y: acf, type: "bar" as const, name: "ACF" }];
  }, [residuals]);

  const heteroData: Partial<Plotly.PlotData>[] = useMemo(() => {
    const rows = residuals?.hetero ?? [];
    if (!rows.length) return [];
    return [{ x: rows.map((r) => r.quantile), y: rows.map((r) => r.mae), type: "bar" as const, name: "MAE" }];
  }, [residuals]);

  const calibBinning = useMemo<Partial<ScatterData>[]>(() => {
    const rows = calib?.binned ?? [];
    if (!rows.length) return [];
    const xp = rows.map((r) => r.y_pred_mean);
    const yt = rows.map((r) => r.y_true_mean);
    const { min, max } = safeMinMax([xp, yt]);
    const line: Partial<ScatterData> = {
      x: [min, max],
      y: [min, max],
      type: "scatter",
      mode: "lines",
      name: "y = x",
      hoverinfo: "none",
    };
    const pts: Partial<ScatterData> = { x: xp, y: yt, type: "scatter", mode: "markers", name: "Binned means" };
    return [line, pts];
  }, [calib]);

  const betaByHour = useMemo<Partial<ScatterData>[]>(() => {
    const rows = calib?.by_hour ?? [];
    if (!rows.length) return [];
    return [
      {
        x: rows.map((r) => r.hour),
        y: rows.map((r) => (Number.isFinite(Number(r.beta)) ? Number(r.beta) : null)),
        type: "scatter",
        mode: "lines+markers",
        name: "β (pente) vs heure",
      },
    ];
  }, [calib]);

  const relErrBars: Partial<Plotly.PlotData>[] = useMemo(() => {
    const rows = calib?.rel_error_levels ?? [];
    if (!rows.length) return [];
    return [
      {
        x: rows.map((r) => r.level),
        y: rows.map((r) => r.mape_like * 100),
        type: "bar" as const,
        name: "MAPE-like (%)",
      },
    ];
  }, [calib]);

  const stationBias = useMemo(() => {
    const rows = calib?.bias_by_station ?? [];
    if (!rows.length) return { tops: [], bots: [] as CalibrationDoc["bias_by_station"] };
    const withN = rows.filter((r) => Number(r.n) >= 30);
    const sorted = [...withN].sort((a, b) => Math.abs(Number(b.bias ?? 0)) - Math.abs(Number(a.bias ?? 0)));
    return { tops: sorted.slice(0, 20), bots: sorted.slice(-20).reverse() };
  }, [calib]);

  /* ───────────────── Render ───────────────── */
  return (
    <div className="monitoring">
      <Head>
        <title>Monitoring — Model / Explainability</title>
        <meta
          name="description"
          content="Résidus, QQ, ACF, hétéroscédasticité, calibration et incertitude."
        />
      </Head>

      <main className="page" style={{ paddingTop: "calc(var(--header-h, 70px) + 12px)" }}>
        <MonitoringNav
          title="Model — Explainability"
          subtitle="Résidus, QQ, ACF, hétéroscédasticité, calibration & incertitude"
          generatedAt={generatedAt}
          extraActions={[{ label: "Performance", href: "/monitoring/model/performance" }]}
        />

        <LoadingBar status={barStatus} />

        {/* KpiBar + meta */}
        <section className="mt-4">
          <KpiBar items={kpiItems} dense />
          {metaLine && <div className="kpi-bar-meta">{metaLine}</div>}
        </section>

        {/* Résidus */}
        <section className="mt-6">
          <h2>Résidus</h2>
          <div className="grid-2">
            <div className="plot-card">
              <h3>Histogramme</h3>
              {histData.length ? (
                <Plot
                  data={histData as Plotly.Data[]}
                  layout={chartLayout({
                    height: 320,
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

            <div className="plot-card">
              <h3>QQ-plot (normalisé)</h3>
              {qqData.length ? (
                <Plot
                  data={qqData as Plotly.Data[]}
                  layout={chartLayout({
                    height: 320,
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

          <div className="grid-2 mt-4">
            <div className="plot-card">
              <h3>ACF du résidu (lag 5 min)</h3>
              {acfData.length ? (
                <Plot
                  data={acfData as Plotly.Data[]}
                  layout={chartLayout({
                    height: 320,
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

            <div className="plot-card">
              <h3>Hétéroscédasticité (MAE par quantile de y_true)</h3>
              {heteroData.length ? (
                <Plot
                  data={heteroData as Plotly.Data[]}
                  layout={chartLayout({
                    height: 320,
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

          <div className="plot-card mt-4">
            <h3>Épisodes d’erreurs (|résidu| ≥ 4)</h3>
            {residuals?.episodes?.length ? (
              <ul className="status-list" style={{ listStyle: "none", paddingLeft: 0 }}>
                {residuals.episodes.slice(0, 30).map((r) => (
                  <li key={r.station_id}>
                    <div className="row" style={{ gap: 10 }}>
                      <div style={{ whiteSpace: "nowrap", textOverflow: "ellipsis", overflow: "hidden" }}>
                        <b>{r.station_id}</b> <span className="muted">· n={fmtInt(r.n)}</span>
                      </div>
                      <div className="small muted">
                        max run:&nbsp;<b>{fmtInt(r.max_run)}</b>
                      </div>
                      <div style={{ flex: 1 }}>
                        <div
                          style={{
                            height: 6,
                            borderRadius: 999,
                            background: "#111827",
                            position: "relative",
                            overflow: "hidden",
                          }}
                        >
                          <div
                            style={{
                              height: "100%",
                              width: `${Math.min(100, Number(r.max_run) * 4)}%`,
                              background: "var(--ok)",
                            }}
                            aria-hidden
                          />
                        </div>
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="empty">—</div>
            )}
          </div>
        </section>

        {/* Calibration */}
        <section className="mt-6">
          <h2>Calibration</h2>

          <div className="grid-2">
            <div className="plot-card">
              <h3>Binned means (y_pred → y_true)</h3>
              {calibBinning.length ? (
                <Plot
                  data={calibBinning as Plotly.Data[]}
                  layout={chartLayout({
                    height: 320,
                    xaxis: { title: { text: "E[y_pred]" } },
                    yaxis: { title: { text: "E[y_true]" } },
                  })}
                  config={chartConfig}
                  className="plot plot--sm"
                />
              ) : (
                <div className="empty">—</div>
              )}
              {calib?.fit && (
                <div className="small mt-2">
                  Fit global: y ≈ α + β·ŷ — α={fmtNum(calib.fit.alpha, 3)}, β={fmtNum(calib.fit.beta, 3)} · n=
                  {fmtInt(calib.fit.n)}
                </div>
              )}
            </div>

            <div className="plot-card">
              <h3>β par heure (locale)</h3>
              {betaByHour.length ? (
                <Plot
                  data={betaByHour as Plotly.Data[]}
                  layout={chartLayout({
                    height: 320,
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
          </div>

          <div className="grid-2 mt-4">
            <div className="plot-card">
              <h3>Erreur relative par niveau</h3>
              {relErrBars.length ? (
                <Plot
                  data={relErrBars as Plotly.Data[]}
                  layout={chartLayout({
                    height: 320,
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

            <div className="plot-card">
              <h3>Stations (biais moyen)</h3>
              {calib?.bias_by_station?.length ? (
                <ul className="status-list" style={{ listStyle: "none", paddingLeft: 0 }}>
                  {stationBias.tops.slice(0, 20).map((r) => (
                    <li key={r.station_id}>
                      <div className="row" style={{ gap: 10 }}>
                        <div style={{ overflow: "hidden" }}>
                          <div style={{ whiteSpace: "nowrap", textOverflow: "ellipsis", overflow: "hidden" }}>
                            <b>{r.name ?? r.station_id}</b>
                            <span className="muted">{r.name ? ` (#${r.station_id})` : ""} · n={fmtInt(r.n)}</span>
                          </div>
                          <div
                            style={{
                              height: 6,
                              borderRadius: 999,
                              background: "#111827",
                              marginTop: 6,
                              position: "relative",
                              overflow: "hidden",
                            }}
                          >
                            <div
                              style={{
                                width: `${Math.min(100, Math.abs(Number(r.bias ?? 0)) * 5)}%`,
                                height: "100%",
                                background: "#ef4444",
                              }}
                              aria-hidden
                            />
                          </div>
                        </div>
                        <div style={{ marginLeft: "auto", textAlign: "right" }}>
                          <div className="small muted">biais</div>
                          <div style={{ fontWeight: 700 }}>{fmtNum(r.bias, 2)}</div>
                        </div>
                        <div style={{ textAlign: "right" }}>
                          <div className="small muted">n</div>
                          <div style={{ fontWeight: 700 }}>{fmtInt(r.n)}</div>
                        </div>
                      </div>
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="empty">—</div>
              )}
              <div className="small mt-2">Tri sur |biais|, filtre n ≥ 30 pour réduire le bruit.</div>
            </div>
          </div>
        </section>

        {/* Incertitude */}
        <section className="mt-6">
          <h2>Incertitude</h2>
          <div className="plot-card">
            {unc?.coverage ? (
              <div className="row">
                <div className="kpi">
                  <div className="kpi__label small muted">Coverage empirique</div>
                  <div className="kpi__value" style={{ fontWeight: 700 }}>
                    {Number.isFinite(Number(unc.coverage.empirical))
                      ? `${(Number(unc.coverage.empirical) * 100).toFixed(1)}%`
                      : "—"}
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
                <div className="small muted">
                  Couverture = part(y_true ∈ [yhat_lo, yhat_hi]). Attendue ~ 90/95% selon l’intervalle utilisé.
                </div>
              </div>
            ) : (
              <div className="empty">Aucune borne d’incertitude détectée (colonnes yhat_lo / yhat_hi).</div>
            )}
          </div>
        </section>

        <div className="mt-6" />
      </main>
    </div>
  );
}
