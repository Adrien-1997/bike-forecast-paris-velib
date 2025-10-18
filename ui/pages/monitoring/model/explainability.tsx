// ui/pages/monitoring/model/explainability.tsx
import Head from "next/head";
import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import type { ScatterData } from "plotly.js";
import type * as Plotly from "plotly.js";
import MonitoringNav from "@/components/monitoring/MonitoringNav";

/* ───────────────────────── Plotly (client only) ───────────────────────── */
const Plot = dynamic(() => import("react-plotly.js").then((m) => m.default), {
  ssr: false,
  loading: () => <div className="empty">Chargement du graphique…</div>,
});

/* ───────────────────────── Helpers ───────────────────────── */
async function getJSON<T = unknown>(path: string): Promise<T> {
  const base =
    (typeof window !== "undefined" ? (window as any).NEXT_PUBLIC_API_BASE : undefined) ||
    process.env.NEXT_PUBLIC_API_BASE ||
    "";
  const url = base ? new URL(path, base).toString() : path;

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
function ok<T>(r: PromiseSettledResult<T>): T | null { return r.status === "fulfilled" ? r.value : null; }
const fmtNum = (x?: number | null, d = 2) => (Number.isFinite(Number(x)) ? Number(x).toFixed(d) : "—");
const fmtInt = (x?: number | null) => (Number.isFinite(Number(x)) ? Number(x).toLocaleString("fr-FR") : "—");
const fmtPct = (x?: number | null, d = 1) => (Number.isFinite(Number(x)) ? `${(Number(x) * 100).toFixed(d)}%` : "—");
/** Avoids call stack blow-ups from Math.min/max(...bigArray) */
function safeMinMax(arrs: number[][]): { min: number; max: number } {
  let min = Number.POSITIVE_INFINITY, max = Number.NEGATIVE_INFINITY;
  for (const arr of arrs) for (let i = 0; i < arr.length; i++) {
    const v = arr[i]; if (Number.isFinite(v)) { if (v < min) min = v; if (v > max) max = v; }
  }
  if (!Number.isFinite(min)) min = 0;
  if (!Number.isFinite(max)) max = 0;
  return { min, max };
}

/* ───────────────────────── Types (API Explainability) ───────────────────────── */
type Overview = {
  schema_version: string; generated_at: string; tz: string;
  anchor_day_perf: string | null; perf_rows: number; perf_stations: number;
  ts_min_perf: string | null; ts_max_perf: string | null;
  has_y_pred: boolean; has_uncertainty: boolean;
};
type ResidHistBin = { bin_left: number; bin_right: number; count: number };
type ResidualsDoc = {
  schema_version: string; generated_at: string;
  hist: ResidHistBin[];
  qq: { th: number[]; emp: number[] };
  acf: number[];
  hetero: Array<{ quantile: string; mae: number; n: number }>;
  episodes: Array<{ station_id: string; max_run: number; n: number }>;
};
type CalibrationDoc = {
  schema_version: string; generated_at: string;
  fit: { alpha: number | null; beta: number | null; n: number };
  binned: Array<{ quantile: string; y_pred_mean: number; y_true_mean: number; n: number }>;
  by_hour: Array<{ hour: number; alpha: number | null; beta: number | null; n: number }>;
  rel_error_levels: Array<{ level: string; mape_like: number; n: number }>;
  bias_by_station: Array<{ station_id: string; name: string | null; bias: number | null; lat: number | null; lon: number | null; n: number }>;
};
type UncertaintyDoc = {
  schema_version: string; generated_at: string;
  coverage: { empirical: number; n: number } | null;
  method?: string; nominal?: number;
};

/* ───────────────────────── Page ───────────────────────── */
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
    return () => { alive = false; };
  }, []);

  const generatedAt =
    overview?.generated_at || residuals?.generated_at || calib?.generated_at || unc?.generated_at || undefined;

  /* ───────────────────────── Charts ───────────────────────── */
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
    const line: Partial<ScatterData> = { x: [min, max], y: [min, max], type: "scatter", mode: "lines", name: "y = x", hoverinfo: "none" };
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
    return [{ x: rows.map(r => r.quantile), y: rows.map(r => r.mae), type: "bar" as const, name: "MAE" }];
  }, [residuals]);

  const calibBinning = useMemo<Partial<ScatterData>[]>(() => {
    const rows = calib?.binned ?? [];
    if (!rows.length) return [];
    const xp = rows.map((r) => r.y_pred_mean);
    const yt = rows.map((r) => r.y_true_mean);
    const { min, max } = safeMinMax([xp, yt]);
    const line: Partial<ScatterData> = { x: [min, max], y: [min, max], type: "scatter", mode: "lines", name: "y = x", hoverinfo: "none" };
    const pts: Partial<ScatterData> = { x: xp, y: yt, type: "scatter", mode: "markers", name: "Binned means" };
    return [line, pts];
  }, [calib]);

  const betaByHour = useMemo<Partial<ScatterData>[]>(() => {
    const rows = calib?.by_hour ?? [];
    if (!rows.length) return [];
    return [{
      x: rows.map(r => r.hour),
      y: rows.map(r => (Number.isFinite(Number(r.beta)) ? Number(r.beta) : null)),
      type: "scatter", mode: "lines+markers", name: "β (pente) vs heure",
    }];
  }, [calib]);

  const relErrBars: Partial<Plotly.PlotData>[] = useMemo(() => {
    const rows = calib?.rel_error_levels ?? [];
    if (!rows.length) return [];
    return [{ x: rows.map(r => r.level), y: rows.map(r => r.mape_like * 100), type: "bar" as const, name: "MAPE-like (%)" }];
  }, [calib]);

  const stationBias = useMemo(() => {
    const rows = calib?.bias_by_station ?? [];
    if (!rows.length) return { tops: [], bots: [] as CalibrationDoc["bias_by_station"] };
    const withN = rows.filter((r) => Number(r.n) >= 30);
    const sorted = [...withN].sort((a, b) => Math.abs(Number(b.bias ?? 0)) - Math.abs(Number(a.bias ?? 0)));
    return { tops: sorted.slice(0, 20), bots: sorted.slice(-20).reverse() };
  }, [calib]);

  /* ───────────────────────── Render ───────────────────────── */
  return (
    <div className="monitoring">
      <Head>
        <title>Monitoring — Model / Explainability</title>
        <meta name="description" content="Résidus, QQ, ACF, hétéroscédasticité, calibration et incertitude." />
        <link rel="stylesheet" href="/css/monitoring.css" />
      </Head>

      <main className="page">
        <MonitoringNav
          title="Model — Explainability"
          subtitle="Résidus, QQ, ACF, hétéroscédasticité, calibration & incertitude"
          generatedAt={generatedAt}
          crumbs={[
            { label: "Accueil", href: "/" },
            { label: "Monitoring", href: "/monitoring" },
            { label: "App", href: "/app" },
          ]}
          extraActions={[
            { label: "Performance", href: "/monitoring/model/performance" },
          ]}
        />

        {/* Status */}
        {loading && <div className="banner">Chargement…</div>}
        {error && <div className="banner banner--error">{error}</div>}

        {/* KPIs */}
        <section className="mt-4">
          <h2>Résumé</h2>
          <div className="kpi-grid" style={{ gridTemplateColumns: "repeat(6, minmax(0, 1fr))" as any }}>
            <Kpi label="Stations" value={overview?.perf_stations} fmt={fmtInt as any} />
            <Kpi label="Lignes (n)" value={overview?.perf_rows} fmt={fmtInt as any} />
            <Kpi label="Prédictions" value={overview?.has_y_pred ? 1 : 0} fmt={(v) => (v ? "oui" : "non")} />
            <Kpi label="Incertitude" value={overview?.has_uncertainty ? 1 : 0} fmt={(v) => (v ? "oui" : "non")} />
            <Kpi label="UTC min" value={overview?.ts_min_perf ? 1 : 0} fmt={() => overview?.ts_min_perf ?? "—"} />
            <Kpi label="UTC max" value={overview?.ts_max_perf ? 1 : 0} fmt={() => overview?.ts_max_perf ?? "—"} />
          </div>
          {overview && (
            <div className="small mt-2">
              Schema v{overview.schema_version} · Intervalle: {overview.ts_min_perf ?? "—"} → {overview.ts_max_perf ?? "—"} (UTC)
            </div>
          )}
        </section>

        {/* Résidus */}
        <section className="mt-6">
          <h2>Résidus</h2>

          <div className="grid-2">
            <div className="card">
              <h3>Histogramme</h3>
              {histData.length ? (
                <Plot
                  data={histData as Plotly.Data[]}
                  layout={{
                    autosize: true, height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 40 },
                    xaxis: { title: { text: "Erreur (y_true - y_pred)" } },
                    yaxis: { title: { text: "Comptes" } },
                    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  className="plot plot--sm"
                />
              ) : <div className="empty">—</div>}
            </div>

            <div className="card">
              <h3>QQ-plot (normalisé)</h3>
              {qqData.length ? (
                <Plot
                  data={qqData as Plotly.Data[]}
                  layout={{
                    autosize: true, height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 40 },
                    xaxis: { title: { text: "Théorique" } },
                    yaxis: { title: { text: "Empirique" } },
                    hovermode: "closest",
                    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  className="plot plot--sm"
                />
              ) : <div className="empty">—</div>}
            </div>
          </div>

          <div className="grid-2 mt-4">
            <div className="card">
              <h3>ACF du résidu (lag 5 min)</h3>
              {acfData.length ? (
                <Plot
                  data={acfData as Plotly.Data[]}
                  layout={{
                    autosize: true, height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 40 },
                    xaxis: { title: { text: "Lag (5 min)" } },
                    yaxis: { title: { text: "Corrélation" } },
                    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  className="plot plot--sm"
                />
              ) : <div className="empty">—</div>}
            </div>

            <div className="card">
              <h3>Hétéroscédasticité (MAE par quantile de y_true)</h3>
              {heteroData.length ? (
                <Plot
                  data={heteroData as Plotly.Data[]}
                  layout={{
                    autosize: true, height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 80 },
                    xaxis: { title: { text: "Quantiles(y_true)" }, tickangle: -30 },
                    yaxis: { title: { text: "MAE" } },
                    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  className="plot plot--sm"
                />
              ) : <div className="empty">—</div>}
            </div>
          </div>

          <div className="card mt-4">
            <h3>Épisodes d’erreurs (|résidu| ≥ 4)</h3>
            {residuals?.episodes?.length ? (
              <ul className="status-list" style={{ listStyle: "none", paddingLeft: 0 }}>
                {residuals.episodes.slice(0, 30).map((r) => (
                  <li key={r.station_id}>
                    <div className="row" style={{ gap: 10 }}>
                      <div style={{ whiteSpace: "nowrap", textOverflow: "ellipsis", overflow: "hidden" }}>
                        <b>{r.station_id}</b> <span className="muted">· n={fmtInt(r.n)}</span>
                      </div>
                      <div className="small muted">max run:&nbsp;<b>{fmtInt(r.max_run)}</b></div>
                      <div style={{ flex: 1 }}>
                        <div className="bar">
                          <div
                            className="bar__fill"
                            style={{ width: `${Math.min(100, Number(r.max_run) * 4)}%` }}
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
            <div className="card">
              <h3>Binned means (y_pred → y_true)</h3>
              {calibBinning.length ? (
                <Plot
                  data={calibBinning as Plotly.Data[]}
                  layout={{
                    autosize: true, height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 40 },
                    xaxis: { title: { text: "E[y_pred]" } },
                    yaxis: { title: { text: "E[y_true]" } },
                    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  className="plot plot--sm"
                />
              ) : <div className="empty">—</div>}
              {calib?.fit && (
                <div className="small mt-2">
                  Fit global: y ≈ α + β·ŷ — α={fmtNum(calib.fit.alpha, 3)}, β={fmtNum(calib.fit.beta, 3)} · n={fmtInt(calib.fit.n)}
                </div>
              )}
            </div>

            <div className="card">
              <h3>β par heure (locale)</h3>
              {betaByHour.length ? (
                <Plot
                  data={betaByHour as Plotly.Data[]}
                  layout={{
                    autosize: true, height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 36 },
                    xaxis: { title: { text: "Heure" } },
                    yaxis: { title: { text: "β" } },
                    hovermode: "x unified",
                    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  className="plot plot--sm"
                />
              ) : <div className="empty">—</div>}
            </div>
          </div>

          <div className="grid-2 mt-4">
            <div className="card">
              <h3>Erreur relative par niveau</h3>
              {relErrBars.length ? (
                <Plot
                  data={relErrBars as Plotly.Data[]}
                  layout={{
                    autosize: true, height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 36 },
                    xaxis: { title: { text: "Niveau (quantiles y_true)" } },
                    yaxis: { title: { text: "MAPE-like (%)" } },
                    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  className="plot plot--sm"
                />
              ) : <div className="empty">—</div>}
            </div>

            <div className="card">
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
                          <div className="bar mt-2">
                            <div
                              className="bar__fill bar__fill--danger"
                              style={{ width: `${Math.min(100, Math.abs(Number(r.bias ?? 0)) * 5)}%` }}
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
          <div className="card">
            {unc?.coverage ? (
              <div className="row">
                <Kpi label="Coverage empirique" value={unc.coverage.empirical} fmt={(v)=>fmtPct(v,1)} />
                <Kpi label="Échantillon" value={unc.coverage.n} fmt={fmtInt as any} />
                {(unc.method || Number.isFinite(Number(unc.nominal))) && (
                  <div className="small muted">
                    {unc.method ? `Méthode: ${unc.method}` : ""}{Number.isFinite(Number(unc.nominal)) ? ` · nominal=${fmtPct(Number(unc.nominal), 0)}` : ""}
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

/* ───────────────────────── UI atoms (compat monitoring.css) ───────────────────────── */
function Kpi({
  label, value, fmt,
}: { label: string; value: number | null | undefined; fmt?: (v: number | null | undefined) => string }) {
  const text = fmt ? fmt(value) : Number.isFinite(Number(value)) ? String(value) : "—";
  return (
    <div className="kpi">
      <div className="kpi__label">{label}</div>
      <div className="kpi__value">{text}</div>
    </div>
  );
}