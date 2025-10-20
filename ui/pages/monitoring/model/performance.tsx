// ui/pages/monitoring/model/performance.tsx
import Head from "next/head";
import { useCallback, useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import type * as Plotly from "plotly.js";
import MonitoringNav from "@/components/monitoring/MonitoringNav";
import GlobalHeader from "@/components/layout/GlobalHeader";
import GlobalFooter from "@/components/layout/GlobalFooter";

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
// HTTP JSON — robuste et verbeux pour debug
async function getJSON<T = unknown>(path: string): Promise<T> {
  const isBrowser = typeof window !== "undefined";
  const envBase = process.env.NEXT_PUBLIC_API_BASE || "";               // build-time
  const winBase = isBrowser ? (window as any).__API_BASE__ || "" : "";  // runtime injectable
  const originBase = isBrowser ? window.location.origin : "";

  // Priorité: env → window.__API_BASE__ → origin courant
  const base = envBase || winBase || originBase;

  // URL absolue propre
  const url = base ? new URL(path.replace(/^\//, "/"), base).toString() : path;

  // Debug minimal
  if (isBrowser) {
    // eslint-disable-next-line no-console
    console.debug("[getJSON] →", url);
  }

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
function fmtPct(x?: number | null, digits = 1) {
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  return `${v.toFixed(digits)}%`;
}
function fmtNum(x?: number | null, digits = 2) {
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  return v.toFixed(digits);
}
function fmtInt(x?: number | null) {
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  return v.toLocaleString("fr-FR");
}
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

/* ───────────────────────── Types (API Model/Performance) ───────────────────────── */
type Manifest = {
  schema_version: string;
  generated_at: string;
  latest_prefix: string;
  window_days: number;
  horizons: number[];
};
type KPIs = {
  schema_version: string;
  generated_at: string;
  window_days: number;
  horizon_min: number | null;
  coverage_pred_pct: number | null;
  mae_model: number | null;
  rmse_model: number | null;
  me_model: number | null;
  mae_baseline: number | null;
  rmse_baseline: number | null;
  me_baseline: number | null;
  lift_vs_baseline: number | null; // 0..1
  n_rows: number;
  n_stations: number;
  ts_min_utc: string | null;
  ts_max_utc: string | null;
};
type DailyRow = {
  date: string;
  mae_model: number | null;
  mae_baseline: number | null;
  rmse_model: number | null;
  rmse_baseline: number | null;
  coverage_pred_pct: number | null;
  lift_vs_baseline: number | null;
  n: number;
};
type DailyMetrics = {
  schema_version: string;
  horizon_min: number;
  rows: DailyRow[];
};
type HourRow = { hour: number; mae_model: number | null; mae_baseline: number | null; coverage_pred_pct: number | null; n: number; };
type ByHour = { schema_version: string; horizon_min: number; rows: HourRow[]; };
type DOWRow = { dow: number; mae_model: number | null; mae_baseline: number | null; coverage_pred_pct: number | null; n: number; };
type ByDow = { schema_version: string; horizon_min: number; rows: DOWRow[]; };
type StationRow = { station_id: string; mae_model: number | null; mae_baseline: number | null; coverage_pred_pct: number | null; n: number; lift_vs_baseline: number | null; };
type ByStation = { schema_version: string; horizon_min: number; rows: StationRow[]; };
type LiftCurve = { schema_version: string; horizon_min?: number; points: Array<{ date: string; lift_vs_baseline: number | null }>; };
type HistResiduals = { schema_version: string; horizon_min?: number; bins: number[]; counts: number[]; n: number; };

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

  // Load manifest once
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const m = await getJSON<Manifest>("/monitoring/model/performance/manifest");
        if (!alive) return;
        setManifest(m);
        // si l'horizon courant n'est pas dispo, basculer sur le premier
        if (Array.isArray(m.horizons) && m.horizons.length && !m.horizons.includes(h)) {
          setH(m.horizons[0]);
        }
        setError(null);
      } catch (e: any) {
        if (alive) setError(String(e?.message ?? e));
      }
    })();
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Load data for current horizon
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setLoading(true);
        const res = await Promise.allSettled([
          getJSON<KPIs>(`/monitoring/model/performance/kpis?h=${h}`),
          getJSON<DailyMetrics>(`/monitoring/model/performance/daily_metrics?h=${h}`),
          getJSON<ByHour>(`/monitoring/model/performance/by_hour?h=${h}`),
          getJSON<ByDow>(`/monitoring/model/performance/by_dow?h=${h}`),
          getJSON<ByStation>(`/monitoring/model/performance/by_station?h=${h}`),
          getJSON<LiftCurve>(`/monitoring/model/performance/lift_curve?h=${h}`),
          getJSON<HistResiduals>(`/monitoring/model/performance/hist_residuals?h=${h}`),
        ]);
        if (!alive) return;
        setKpis(ok(res[0]));
        setDaily(ok(res[1]));
        setByHour(ok(res[2]));
        setByDow(ok(res[3]));
        setByStation(ok(res[4]));
        setLift(ok(res[5]));
        setHist(ok(res[6]));
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
  }, [h]);

  const generatedAt =
    kpis?.generated_at ??
    manifest?.generated_at ??
    null;

  // ── Charts ────────────────────────────────────────────────────────────
  const liftCurve: Partial<Plotly.PlotData> | null = useMemo(() => {
    const pts = lift?.points ?? [];
    if (!pts.length) return null;
    const x = pts.map((p) => p.date);
    const y = pts.map((p) => (Number.isFinite(Number(p.lift_vs_baseline)) ? Number(p.lift_vs_baseline) * 100 : null));
    return { x, y, type: "scatter", mode: "lines+markers", name: "Lift (%)", connectgaps: false };
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
    // centers des bins pour barres
    const centers = bins.slice(0, -1).map((b, i) => (b + bins[i + 1]) / 2);
    return [{ x: centers, y: counts, type: "bar" as const, name: "Résidus (y_true - y_pred)" }];
  }, [hist]);

  // ── Stations Top/Bottom lift ──────────────────────────────────────────
  const stationsTopBottom = useMemo(() => {
    const rows = byStation?.rows ?? [];
    const wLift = (x: StationRow) => (Number.isFinite(Number(x.lift_vs_baseline)) ? Number(x.lift_vs_baseline) : -Infinity);
    const minN = 30; // filtre anti-bruit
    const filtered = rows.filter((r) => Number(r.n) >= minN);
    const tops = [...filtered].sort((a, b) => wLift(b) - wLift(a)).slice(0, 20);
    const bots = [...filtered].sort((a, b) => wLift(a) - wLift(b)).slice(0, 20);
    return { tops, bots };
  }, [byStation]);

  return (
    <div className="monitoring">
      <Head>
        <title>Monitoring — Model / Performance</title>
        <meta name="description" content="Comparatif modèle vs baseline, lift, histogrammes, découpes et stations." />
        {/* Ajout minime: barre de progression réutilisable */}
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

      {/* Header global sticky */}
      <GlobalHeader />

      {/* Contenu principal */}
      <main className="page" style={{ paddingTop: "calc(var(--header-h, 70px) + 12px)" }}>
        <MonitoringNav
          title="Model — Performance"
          subtitle={`MAE/RMSE vs baseline, lift, histogrammes & découpes (h=${h} min)`}
          generatedAt={generatedAt ?? undefined}
          extraActions={[
            { label: "Explainability", href: "/monitoring/model/explainability" },
          ]}
        />

        {/* Toolbar / Controls */}
        <section className="mt-3">
          <div className="card" style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
            <div className="small">Horizon</div>
            <HorizonSelect value={h} options={manifest?.horizons ?? [15]} onChange={(v) => setH(v)} />
            <div className="small" style={{ marginLeft: 8 }}>
              fenêtre: {kpis?.window_days ?? manifest?.window_days ?? "—"} j
            </div>
          </div>

          {loading && <div className="banner mt-2">Chargement…</div>}
          {error && <div className="banner banner--error mt-2">{error}</div>}
        </section>

        {/* KPIs */}
        <section className="mt-4">
          <h2>Résumé — KPIs (h={h} min)</h2>
          <div className="kpi-grid" style={{ gridTemplateColumns: "repeat(6,minmax(0,1fr))" }}>
            <Kpi label="Stations" value={kpis?.n_stations} fmt={fmtInt as any} />
            <Kpi label="Lignes (n)" value={kpis?.n_rows} fmt={fmtInt as any} />
            <Kpi label="MAE — Modèle" value={kpis?.mae_model} fmt={(v) => fmtNum(v, 2)} />
            <Kpi label="MAE — Baseline" value={kpis?.mae_baseline} fmt={(v) => fmtNum(v, 2)} />
            <Kpi
              label="Lift vs baseline"
              value={Number.isFinite(Number(kpis?.lift_vs_baseline)) ? Number(kpis?.lift_vs_baseline) * 100 : null}
              fmt={(v) => fmtPct(v, 1)}
            />
            <Kpi label="Coverage préd." value={kpis?.coverage_pred_pct} fmt={(v) => fmtPct(v, 1)} />
          </div>
          <div className="small mt-2">
            Schema v{kpis?.schema_version ?? manifest?.schema_version ?? "—"} · Intervalle: {kpis?.ts_min_utc ?? "—"} → {kpis?.ts_max_utc ?? "—"} (UTC)
          </div>
        </section>

        {/* Lift over time */}
        <section className="mt-6">
          <h2>Lift quotidien</h2>
          {liftCurve ? (
            <Plot
              data={[liftCurve as Plotly.Data]}
              layout={{
                autosize: true,
                height: 320,
                margin: { l: 54, r: 10, t: 10, b: 40 },
                yaxis: { title: { text: "Lift (%)" } },
                xaxis: { title: { text: "Date (locale)" } },
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                hovermode: "x unified",
              }}
              config={{ displayModeBar: false, responsive: true }}
              className="plot plot--lg"
            />
          ) : (
            <div className="empty">Pas de courbe de lift.</div>
          )}
        </section>

        {/* Daily bars MAE (baseline vs modèle) */}
        <section className="mt-6">
          <h2>MAE par jour (Modèle vs Baseline)</h2>
          {dailyBars.length ? (
            <Plot
              data={dailyBars as Plotly.Data[]}
              layout={{
                barmode: "group",
                autosize: true,
                height: 360,
                margin: { l: 54, r: 10, t: 10, b: 60 },
                yaxis: { title: { text: "MAE" } },
                xaxis: { title: { text: "Date (locale)" }, tickangle: -30 },
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
              }}
              config={{ displayModeBar: false, responsive: true }}
              className="plot plot--lg"
            />
          ) : (
            <div className="empty">Pas de métriques quotidiennes.</div>
          )}
        </section>

        {/* By hour / By dow */}
        <section className="mt-6">
          <h2>Découpes — heure & jour de semaine</h2>
          <div className="grid-2">
            <div className="card">
              <h3>Par heure (locale)</h3>
              {byHourBars.length ? (
                <Plot
                  data={byHourBars as Plotly.Data[]}
                  layout={{
                    barmode: "group",
                    autosize: true,
                    height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 36 },
                    yaxis: { title: { text: "MAE" } },
                    xaxis: { title: { text: "Heure" } },
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  className="plot plot--sm"
                />
              ) : (
                <div className="empty">—</div>
              )}
            </div>

            <div className="card">
              <h3>Par jour de semaine</h3>
              {byDowBars.length ? (
                <Plot
                  data={byDowBars as Plotly.Data[]}
                  layout={{
                    barmode: "group",
                    autosize: true,
                    height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 36 },
                    yaxis: { title: { text: "MAE" } },
                    xaxis: { title: { text: "Jour" } },
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  className="plot plot--sm"
                />
              ) : (
                <div className="empty">—</div>
              )}
            </div>
          </div>
        </section>

        {/* Histogramme des résidus */}
        <section className="mt-6">
          <h2>Histogramme des résidus (y_true - y_pred)</h2>
          {histData.length ? (
            <Plot
              data={histData as Plotly.Data[]}
              layout={{
                autosize: true,
                height: 320,
                margin: { l: 54, r: 10, t: 10, b: 40 },
                xaxis: { title: { text: "Erreur" } },
                yaxis: { title: { text: "Comptes" } },
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
              }}
              config={{ displayModeBar: false, responsive: true }}
              className="plot plot--sm"
            />
          ) : (
            <div className="empty">Pas d’histogramme disponible.</div>
          )}
          {hist?.n != null && <div className="small mt-2">n = {fmtInt(hist.n)} points</div>}
        </section>

        {/* Top / Bottom stations by lift */}
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
            Filtre stations avec n ≥ 30 pour éviter les faux positifs. Le <b>lift</b> = (MAE_baseline - MAE_modèle)/MAE_baseline.
          </div>
        </section>
      </main>

      {/* Footer global */}
      <GlobalFooter />
    </div>
  );
}

/* ───────────────────────── UI atoms & widgets (CSS-first) ───────────────────────── */
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
    <div className="kpi">
      <div className="kpi__label">{label}</div>
      <div className="kpi__value">{text}</div>
    </div>
  );
}

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
        style={{ background: "#0b1220" }}
      >
        {opts.map((o) => (
          <option key={o} value={o}>
            {o} min
          </option>
        ))}
      </select>
      <span className="small">
        ({opts.length} option{opts.length > 1 ? "s" : ""})
      </span>
    </div>
  );
}

function StationList({ rows }: { rows: StationRow[]; kind: "top" | "bottom" }) {
  return (
    <ul className="status-list" style={{ listStyle: "none", paddingLeft: 0 }}>
      {rows.map((r) => {
        const lift = Number.isFinite(Number(r.lift_vs_baseline)) ? Number(r.lift_vs_baseline) : null;
        const liftPct = lift != null ? lift * 100 : null;
        const maeM = Number(r.mae_model);
        const maeB = Number(r.mae_baseline);
        const better = lift != null && lift > 0;
        return (
          <li key={r.station_id}>
            <div className="row" style={{ gap: 10, alignItems: "center" }}>
              <div style={{ flex: 1, overflow: "hidden" }}>
                <div style={{ whiteSpace: "nowrap", textOverflow: "ellipsis", overflow: "hidden" }}>
                  <b>{r.station_id}</b>
                  <span className="muted"> · n={fmtInt(r.n)}</span>
                </div>
                <div className="bar mt-2">
                  <div
                    className={`bar__fill ${better ? "" : "bar__fill--danger"}`}
                    style={{ width: `${Math.max(0, Math.min(100, liftPct ?? 0))}%` }}
                    aria-hidden
                  />
                </div>
              </div>
              <div style={{ textAlign: "right", minWidth: 90 }}>
                <div className="small muted">MAE base</div>
                <div style={{ fontWeight: 700 }}>{fmtNum(maeB, 2)}</div>
              </div>
              <div style={{ textAlign: "right", minWidth: 90 }}>
                <div className="small muted">MAE modèle</div>
                <div style={{ fontWeight: 700 }}>{fmtNum(maeM, 2)}</div>
              </div>
              <div style={{ textAlign: "right", fontWeight: 700, minWidth: 70 }}>{fmtPct(liftPct, 1)}</div>
            </div>
          </li>
        );
      })}
    </ul>
  );
}
