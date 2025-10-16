// ui/pages/monitoring/model/explainability.tsx
import Head from "next/head";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import type { ScatterData } from "plotly.js"; // for typed scatter traces
import type * as Plotly from "plotly.js";

/* ───────────────────────── Plotly (client only) ───────────────────────── */
export const Plot = dynamic(() => import("react-plotly.js").then((m) => m.default), {
  ssr: false,
  loading: () => (
    <div style={{ height: 320, display: "grid", placeItems: "center", opacity: 0.7 }}>
      Chargement du graphique…
    </div>
  ),
});

/* ───────────────────────── Helpers ───────────────────────── */
async function getJSON<T = unknown>(path: string): Promise<T> {
  const isBrowser = typeof window !== "undefined";
  const envBase = process.env.NEXT_PUBLIC_API_BASE || "";
  const winBase = isBrowser ? (window as any).__API_BASE__ || "" : "";
  const originBase = isBrowser ? window.location.origin : "";
  const base = envBase || winBase || originBase;
  const url = base ? new URL(path.replace(/^\//, "/"), base).toString() : path;

  if (isBrowser) console.debug("[getJSON] →", url);

  const res = await fetch(url, { headers: { accept: "application/json" }, cache: "no-store" });
  const ct = res.headers.get("content-type") || "";
  if (!res.ok) {
    const hint = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status} on ${url} — ${hint.slice(0, 200)}`);
  }
  if (!ct.includes("application/json")) {
    const peek = await res.text().catch(() => "");
    throw new Error(`Non-JSON response from ${url}: ${peek.slice(0, 160)}`);
  }
  return (await res.json()) as T;
}
function ok<T>(r: PromiseSettledResult<T>): T | null {
  return r.status === "fulfilled" ? r.value : null;
}
function fmtNum(x?: number | null, digits = 2) {
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  return v.toFixed(digits);
}
function fmtPct(x?: number | null, digits = 1) {
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}
function fmtInt(x?: number | null) {
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  return v.toLocaleString("fr-FR");
}
/** Avoids call stack blow-ups from Math.min/max(...bigArray) */
function safeMinMax(arrs: number[][]): { min: number; max: number } {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
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

/* ───────────────────────── Types (API Explainability) ───────────────────────── */
type Overview = {
  schema_version: string;
  generated_at: string;
  tz: string;
  anchor_day_perf: string | null;
  perf_rows: number;
  perf_stations: number;
  ts_min_perf: string | null;
  ts_max_perf: string | null;
  has_y_pred: boolean;
  has_uncertainty: boolean;
};

type ResidHistBin = { bin_left: number; bin_right: number; count: number };
type ResidualsDoc = {
  schema_version: string;
  generated_at: string;
  hist: ResidHistBin[];
  qq: { th: number[]; emp: number[] }; // theor vs empirical
  acf: number[];
  hetero: Array<{ quantile: string; mae: number; n: number }>;
  episodes: Array<{ station_id: string; max_run: number; n: number }>;
};

type CalibrationDoc = {
  schema_version: string;
  generated_at: string;
  fit: { alpha: number | null; beta: number | null; n: number };
  binned: Array<{ quantile: string; y_pred_mean: number; y_true_mean: number; n: number }>;
  by_hour: Array<{ hour: number; alpha: number | null; beta: number | null; n: number }>;
  rel_error_levels: Array<{ level: string; mape_like: number; n: number }>;
  bias_by_station: Array<{
    station_id: string;
    name: string | null;          // ⬅️ utilise le nom quand dispo
    bias: number | null;
    lat: number | null;
    lon: number | null;
    n: number;
  }>;
};

type UncertaintyDoc = {
  schema_version: string;
  generated_at: string;
  coverage: { empirical: number; n: number } | null;
  method?: string;   // optionnel (ex: residual_quantiles/by_hour)
  nominal?: number;  // optionnel (ex: 0.90)
};

/* ───────────────────────── Page ───────────────────────── */
export default function ModelExplainabilityPage() {
  const [overview, setOverview] = useState<Overview | null>(null);
  const [residuals, setResiduals] = useState<ResidualsDoc | null>(null);
  const [calib, setCalib] = useState<CalibrationDoc | null>(null);
  const [unc, setUnc] = useState<UncertaintyDoc | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load all docs (latest-only)
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setLoading(true);
        const res = await Promise.allSettled([
          getJSON<Overview>("/monitoring/model/explainability/overview"),
          getJSON<ResidualsDoc>("/monitoring/model/explainability/residuals"),
          getJSON<CalibrationDoc>("/monitoring/model/explainability/calibration"),
          getJSON<UncertaintyDoc>("/monitoring/model/explainability/uncertainty"),
        ]);
        if (!alive) return;
        setOverview(ok(res[0]));
        setResiduals(ok(res[1]));
        setCalib(ok(res[2]));
        setUnc(ok(res[3]));
        setError(null);
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
    overview?.generated_at ??
    residuals?.generated_at ??
    calib?.generated_at ??
    unc?.generated_at ??
    null;

  /* ───────────────────────── Charts ───────────────────────── */

  // Histogramme des résidus
  const histData: Partial<Plotly.PlotData>[] = useMemo(() => {
    const bins = residuals?.hist ?? [];
    if (!bins.length) return [];
    const centers = bins.map((b) => (b.bin_left + b.bin_right) / 2);
    const counts = bins.map((b) => b.count);
    return [{ x: centers, y: counts, type: "bar" as const, name: "Résidus" }];
  }, [residuals]);

  // QQ plot (empirique vs théorique) + y=x (NO spread)
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
    const pts: Partial<ScatterData> = {
      x: th,
      y: emp,
      type: "scatter",
      mode: "markers",
      name: "Quantiles",
    };
    return [line, pts];
  }, [residuals]);

  // ACF résidu moyen par timestamp
  const acfData: Partial<Plotly.PlotData>[] = useMemo(() => {
    const acf = residuals?.acf ?? [];
    if (!acf.length) return [];
    const x = Array.from({ length: acf.length }, (_, i) => i);
    return [{ x, y: acf, type: "bar" as const, name: "ACF" }];
  }, [residuals]);

  // Hétéroscédasticité — MAE par quantile de y_true
  const heteroData: Partial<Plotly.PlotData>[] = useMemo(() => {
    const rows = residuals?.hetero ?? [];
    if (!rows.length) return [];
    const x = rows.map((r) => r.quantile);
    const y = rows.map((r) => r.mae);
    return [{ x, y, type: "bar" as const, name: "MAE par quantile(y_true)" }];
  }, [residuals]);

  // Calibration — binned means (NO spread)
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
    const pts: Partial<ScatterData> = {
      x: xp,
      y: yt,
      type: "scatter",
      mode: "markers",
      name: "Binned means",
    };
    return [line, pts];
  }, [calib]);

  // β (pente) par heure
  const betaByHour = useMemo<Partial<ScatterData>[]>(() => {
    const rows = calib?.by_hour ?? [];
    if (!rows.length) return [];
    const x = rows.map((r) => r.hour);
    const y = rows.map((r) => (Number.isFinite(Number(r.beta)) ? Number(r.beta) : null));
    const trace: Partial<ScatterData> = {
      x,
      y,
      type: "scatter",
      mode: "lines+markers",
      name: "β (pente) vs heure",
    };
    return [trace];
  }, [calib]);

  // Erreur relative par niveaux (Bas / Moyen / Haut)
  const relErrBars: Partial<Plotly.PlotData>[] = useMemo(() => {
    const rows = calib?.rel_error_levels ?? [];
    if (!rows.length) return [];
    const x = rows.map((r) => r.level);
    const y = rows.map((r) => r.mape_like * 100);
    return [{ x, y, type: "bar" as const, name: "MAPE-like (%)" }];
  }, [calib]);

  // Top/bottom biais par station (triés par |bias|)
  const stationBias = useMemo(() => {
    const rows = calib?.bias_by_station ?? [];
    if (!rows.length) return { tops: [], bots: [] as CalibrationDoc["bias_by_station"] };
    const withN = rows.filter((r) => Number(r.n) >= 30);
    const sorted = [...withN].sort((a, b) => Math.abs(Number(b.bias ?? 0)) - Math.abs(Number(a.bias ?? 0)));
    return { tops: sorted.slice(0, 20), bots: sorted.slice(-20).reverse() };
  }, [calib]);

  /* ───────────────────────── Layout ───────────────────────── */
  return (
    <>
      <Head>
        <title>Monitoring — Model / Explainability</title>
        <meta name="description" content="Résidus, QQ, ACF, hétéroscédasticité, calibration et incertitude." />
        <style
          dangerouslySetInnerHTML={{
            __html: `
              html, body, #__next { height: auto !important; min-height: 100% !important; overflow-y: auto !important; }
              body { overflow-y: scroll !important; overflow-x: hidden !important; position: static !important; overscroll-behavior: auto !important; }
              main { overflow: visible !important; display: block !important; min-height: 100vh !important; }
            `,
          }}
        />
      </Head>

      <main style={{ padding: "2rem", maxWidth: 1280, margin: "0 auto" }}>
        {/* Header */}
        <header style={{ display: "flex", justifyContent: "space-between", gap: 16, alignItems: "center" }}>
          <div>
            <h1 style={{ margin: 0 }}>Model — Explainability</h1>
            <div className="small" style={{ opacity: 0.7 }}>
              {generatedAt ? `Généré : ${new Date(generatedAt).toLocaleString("fr-FR")}` : "—"}
            </div>
          </div>
          <nav style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <Link
              href="/monitoring/model/performance"
              style={{ padding: "6px 10px", borderRadius: 8, border: "1px solid #d1d5db", background: "white", color: "#111827", textDecoration: "none" }}
            >
              Model / Performance
            </Link>
            <Link
              href="/monitoring/network/overview"
              style={{ padding: "6px 10px", borderRadius: 8, border: "1px solid #d1d5db", background: "white", color: "#111827", textDecoration: "none" }}
            >
              Network / Overview
            </Link>
          </nav>
        </header>

        {/* Status */}
        <section style={{ marginTop: 16, display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          {loading && <Banner kind="info">Chargement…</Banner>}
          {error && <Banner kind="error">{error}</Banner>}
        </section>

        {/* KPIs */}
        <section style={{ marginTop: 16 }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(6, minmax(0, 1fr))", gap: 12 }}>
            <Kpi label="Stations" value={overview?.perf_stations} fmt={fmtInt as any} />
            <Kpi label="Lignes (n)" value={overview?.perf_rows} fmt={fmtInt as any} />
            <Kpi label="Prédictions" value={overview?.has_y_pred ? 1 : 0} fmt={(v) => (v ? "oui" : "non")} />
            <Kpi label="Incertitude" value={overview?.has_uncertainty ? 1 : 0} fmt={(v) => (v ? "oui" : "non")} />
            <Kpi label="UTC min" value={overview?.ts_min_perf ? 1 : 0} fmt={() => overview?.ts_min_perf ?? "—"} />
            <Kpi label="UTC max" value={overview?.ts_max_perf ? 1 : 0} fmt={() => overview?.ts_max_perf ?? "—"} />
          </div>
          <div style={{ fontSize: 12, opacity: 0.7, marginTop: 6 }}>
            Schema v{overview?.schema_version ?? "—"} · Intervalle: {overview?.ts_min_perf ?? "—"} → {overview?.ts_max_perf ?? "—"} (UTC)
          </div>
        </section>

        {/* Residuals section */}
        <section style={{ marginTop: 24 }}>
          <h2 style={{ margin: "12px 0" }}>Résidus</h2>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 16 }}>
            <Card>
              <h3 style={{ margin: "6px 0 10px 0", fontSize: 16 }}>Histogramme</h3>
              {histData.length ? (
                <Plot
                  data={histData as Plotly.Data[]}
                  layout={{
                    autosize: true,
                    height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 40 },
                    xaxis: { title: { text: "Erreur (y_true - y_pred)" } },
                    yaxis: { title: { text: "Comptes" } },
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: "100%" }}
                />
              ) : (
                <Empty>—</Empty>
              )}
            </Card>

            <Card>
              <h3 style={{ margin: "6px 0 10px 0", fontSize: 16 }}>QQ-plot (normalisé)</h3>
              {qqData.length ? (
                <Plot
                  data={qqData as Plotly.Data[]}
                  layout={{
                    autosize: true,
                    height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 40 },
                    xaxis: { title: { text: "Théorique" } },
                    yaxis: { title: { text: "Empirique" } },
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                    hovermode: "closest",
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: "100%" }}
                />
              ) : (
                <Empty>—</Empty>
              )}
            </Card>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 16, marginTop: 16 }}>
            <Card>
              <h3 style={{ margin: "6px 0 10px 0", fontSize: 16 }}>ACF du résidu (lag 5 min)</h3>
              {acfData.length ? (
                <Plot
                  data={acfData as Plotly.Data[]}
                  layout={{
                    autosize: true,
                    height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 40 },
                    xaxis: { title: { text: "Lag (5 min)" } },
                    yaxis: { title: { text: "Corrélation" } },
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: "100%" }}
                />
              ) : (
                <Empty>—</Empty>
              )}
            </Card>

            <Card>
              <h3 style={{ margin: "6px 0 10px 0", fontSize: 16 }}>Hétéroscédasticité (MAE par quantile de y_true)</h3>
              {heteroData.length ? (
                <Plot
                  data={heteroData as Plotly.Data[]}
                  layout={{
                    autosize: true,
                    height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 80 },
                    xaxis: { title: { text: "Quantiles(y_true)" }, tickangle: -30 },
                    yaxis: { title: { text: "MAE" } },
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: "100%" }}
                />
              ) : (
                <Empty>—</Empty>
              )}
            </Card>
          </div>

          <Card>
            <h3 style={{ margin: "12px 0 10px 0", fontSize: 16 }}>Épisodes d’erreurs (|résidu| ≥ 4)</h3>
            {residuals?.episodes?.length ? (
              <ul style={{ margin: 0, padding: 0, listStyle: "none", display: "grid", gap: 8 }}>
                {residuals.episodes.slice(0, 30).map((r) => (
                  <li key={r.station_id}>
                    <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) 140px 120px", gap: 10, alignItems: "center" }}>
                      <div style={{ whiteSpace: "nowrap", textOverflow: "ellipsis", overflow: "hidden" }}>
                        <b>{r.station_id}</b> <span style={{ opacity: 0.6 }}>· n={fmtInt(r.n)}</span>
                      </div>
                      <div>max run: <b>{fmtInt(r.max_run)}</b></div>
                      <div>
                        <div style={{ height: 6, borderRadius: 999, background: "#111827", position: "relative", overflow: "hidden" }}>
                          <div style={{ width: `${Math.min(100, Number(r.max_run) * 4)}%`, height: "100%", background: "#f59e0b" }} />
                        </div>
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <Empty>—</Empty>
            )}
          </Card>
        </section>

        {/* Calibration section */}
        <section style={{ marginTop: 32 }}>
          <h2 style={{ margin: "12px 0" }}>Calibration</h2>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 16 }}>
            <Card>
              <h3 style={{ margin: "6px 0 10px 0", fontSize: 16 }}>Binned means (y_pred → y_true)</h3>
              {calibBinning.length ? (
                <Plot
                  data={calibBinning as Plotly.Data[]}
                  layout={{
                    autosize: true,
                    height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 40 },
                    xaxis: { title: { text: "E[y_pred]" } },
                    yaxis: { title: { text: "E[y_true]" } },
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: "100%" }}
                />
              ) : (
                <Empty>—</Empty>
              )}
              {calib?.fit && (
                <div style={{ fontSize: 12, opacity: 0.7, marginTop: 6 }}>
                  Fit global: y ≈ α + β·ŷ — α={fmtNum(calib.fit.alpha, 3)}, β={fmtNum(calib.fit.beta, 3)} · n={fmtInt(calib.fit.n)}
                </div>
              )}
            </Card>

            <Card>
              <h3 style={{ margin: "6px 0 10px 0", fontSize: 16 }}>β par heure (locale)</h3>
              {betaByHour.length ? (
                <Plot
                  data={betaByHour as Plotly.Data[]}
                  layout={{
                    autosize: true,
                    height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 36 },
                    xaxis: { title: { text: "Heure" } },
                    yaxis: { title: { text: "β" } },
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                    hovermode: "x unified",
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: "100%" }}
                />
              ) : (
                <Empty>—</Empty>
              )}
            </Card>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 16, marginTop: 16 }}>
            <Card>
              <h3 style={{ margin: "6px 0 10px 0", fontSize: 16 }}>Erreur relative par niveau</h3>
              {relErrBars.length ? (
                <Plot
                  data={relErrBars as Plotly.Data[]}
                  layout={{
                    autosize: true,
                    height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 36 },
                    xaxis: { title: { text: "Niveau (quantiles y_true)" } },
                    yaxis: { title: { text: "MAPE-like (%)" } },
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: "100%" }}
                />
              ) : (
                <Empty>—</Empty>
              )}
            </Card>

            <Card>
              <h3 style={{ margin: "6px 0 10px 0", fontSize: 16 }}>Stations (biais moyen)</h3>
              {calib?.bias_by_station?.length ? (
                <ul style={{ margin: 0, padding: 0, listStyle: "none", display: "grid", gap: 8 }}>
                  {stationBias.tops.slice(0, 20).map((r) => (
                    <li key={r.station_id}>
                      <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) 110px 70px", alignItems: "center", gap: 10 }}>
                        <div style={{ overflow: "hidden" }}>
                          <div style={{ whiteSpace: "nowrap", textOverflow: "ellipsis" }}>
                            <b>{r.name ?? r.station_id}</b>
                            <span style={{ opacity: 0.6 }}>
                              {r.name ? ` (#${r.station_id})` : ""} · n={fmtInt(r.n)}
                            </span>
                          </div>
                          <div style={{ height: 6, borderRadius: 999, background: "#111827", marginTop: 6, position: "relative", overflow: "hidden" }}>
                            <div
                              style={{
                                width: `${Math.min(100, Math.abs(Number(r.bias ?? 0)) * 5)}%`,
                                height: "100%",
                                background: "#f43f5e",
                              }}
                            />
                          </div>
                        </div>
                        <div style={{ textAlign: "right" }}>
                          <small>biais</small>
                          <div style={{ fontWeight: 700 }}>{fmtNum(r.bias, 2)}</div>
                        </div>
                        <div style={{ textAlign: "right" }}>
                          <small>n</small>
                          <div style={{ fontWeight: 700 }}>{fmtInt(r.n)}</div>
                        </div>
                      </div>
                    </li>
                  ))}
                </ul>
              ) : (
                <Empty>—</Empty>
              )}
              <div style={{ fontSize: 12, opacity: 0.7, marginTop: 6 }}>
                Tri sur |biais|, filtre n ≥ 30 pour réduire le bruit.
              </div>
            </Card>
          </div>
        </section>

        {/* Uncertainty */}
        <section style={{ marginTop: 32, marginBottom: 40 }}>
          <h2 style={{ margin: "12px 0" }}>Incertitude</h2>
          <Card>
            {unc?.coverage ? (
              <div style={{ display: "flex", gap: 24, alignItems: "center", flexWrap: "wrap" }}>
                <Kpi label="Coverage empirique" value={unc.coverage.empirical} fmt={(v) => fmtPct(v, 1)} />
                <Kpi label="Échantillon" value={unc.coverage.n} fmt={fmtInt as any} />
                {(unc.method || Number.isFinite(Number(unc.nominal))) && (
                  <div style={{ fontSize: 12, opacity: 0.75 }}>
                    {unc.method ? `Méthode: ${unc.method}` : ""}
                    {Number.isFinite(Number(unc.nominal)) ? ` · nominal=${fmtPct(Number(unc.nominal), 0)}` : ""}
                  </div>
                )}
                <div style={{ fontSize: 12, opacity: 0.75 }}>
                  Couverture = part(y_true ∈ [yhat_lo, yhat_hi]). Attendue ~ 90/95% selon l’intervalle utilisé.
                </div>
              </div>
            ) : (
              <Empty>Aucune borne d’incertitude détectée (colonnes yhat_lo / yhat_hi).</Empty>
            )}
          </Card>
        </section>
      </main>
    </>
  );
}

/* ───────────────────────── UI atoms ───────────────────────── */
function Kpi({
  label,
  value,
  fmt,
}: {
  label: string;
  value: number | null | undefined;
  fmt?: (v: number | null | undefined) => string;
}) {
  const text = fmt ? fmt(value) : Number.isFinite(Number(value)) ? String(value) : "—";
  return (
    <div
      style={{
        border: "1px solid #374151",
        background: "rgba(15, 23, 42, 0.5)",
        borderRadius: 12,
        padding: "10px 12px",
      }}
    >
      <div style={{ fontSize: 12, color: "#9CA3AF", textTransform: "uppercase" }}>{label}</div>
      <div style={{ fontSize: 22, color: "#E5E7EB", fontWeight: 600, marginTop: 2 }}>{text}</div>
    </div>
  );
}

function Card({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        border: "1px solid #374151",
        background: "rgba(15, 23, 42, 0.5)",
        borderRadius: 12,
        padding: 12,
      }}
    >
      {children}
    </div>
  );
}

function Banner({ kind, children }: { kind: "info" | "error"; children: React.ReactNode }) {
  const style =
    kind === "error"
      ? { border: "1px solid #DC2626", background: "rgba(220, 38, 38, 0.08)", color: "#F87171" }
      : { border: "1px solid #374151", background: "rgba(15, 23, 42, 0.5)", color: "inherit", opacity: 0.85 };
  return (
    <div style={{ borderRadius: 10, padding: "8px 10px", fontSize: 13, ...style }}>{children}</div>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        border: "1px solid #374151",
        background: "rgba(15,23,42,0.35)",
        borderRadius: 12,
        padding: "14px",
        color: "#9CA3AF",
        fontSize: 13,
      }}
    >
      {children}
    </div>
  );
}