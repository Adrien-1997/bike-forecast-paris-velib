// ui/pages/monitoring/model/performance.tsx
import Head from "next/head";
import { useCallback, useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import type * as Plotly from "plotly.js";
import MonitoringNav from "@/components/monitoring/MonitoringNav";
import LoadingBar, { type LoadingBarStatus } from "@/components/common/LoadingBar";
import KpiBar, { type KpiItem } from "@/components/monitoring/KpiBar";
import { chartConfig, chartLayout } from "@/lib/plotlyTheme";
import HorizonToggle from "@/components/common/HorizonToggle";

import {
  getPerformanceManifest,
  getPerformanceKpis,
  getPerformanceDailyMetrics,
  getPerformanceByHour,
  getPerformanceByDow,
  getPerformanceByStation,
  getPerformanceLiftCurve,
  // getPerformanceHistResiduals, // dispo si besoin
  getPerformanceStationTimeseries, // ⬅️ NEW (API unique)
  type Manifest,
  type KPIs,
  type DailyMetrics,
  type ByHour,
  type ByDow,
  type ByStation,
  type StationRow,
  type LiftCurve,
  type StationTimeseries, // ⬅️ NEW
} from "@/lib/services/monitoring/model_performance";

// ⬇️ Index stations (JSON local)
import { loadStationsIndexFromArrayJson, type StationMeta } from "@/lib/local/stationsIndex";

/* ───────────────────────── Plotly (client only) ───────────────────────── */
const Plot = dynamic(() => import("react-plotly.js").then((m) => m.default), {
  ssr: false,
  loading: () => (
    <div style={{ height: 320, display: "grid", placeItems: "center", opacity: 0.7 }}>
      Chargement du graphique…
    </div>
  ),
});

/* ───────────────────────── Map (strictement comme overview) ───────────────────────── */
type MapPoint = {
  station_id: string;
  name?: string;
  lat: number;
  lon: number;
  color: string;
  mae_model?: number | null;
  mae_baseline?: number | null;
  lift_pct?: number | null;
  n?: number | null;
};

const SnapshotMap = dynamic(async () => {
  const RL = await import("react-leaflet");
  const { MapContainer, TileLayer, CircleMarker, Tooltip, useMap } = RL as any;
  const React = await import("react");
  const { useEffect, useMemo, useState } = React;

  function FitBounds({ rows }: { rows: Array<{ lat: number; lon: number }> }) {
    const map = useMap();
    useEffect(() => {
      const pts = rows.filter(
        (r) => Number.isFinite(Number(r.lat)) && Number.isFinite(Number(r.lon))
      );
      if (!pts.length) return;
      let minLat = 90, maxLat = -90, minLon = 180, maxLon = -180;
      for (const r of pts) {
        const la = Number(r.lat);
        const lo = Number(r.lon);
        if (la < minLat) minLat = la;
        if (la > maxLat) maxLat = la;
        if (lo < minLon) minLon = lo;
        if (lo > maxLon) maxLon = lo;
      }
      if (minLat <= maxLat && minLon <= maxLon) {
        map.fitBounds(
          [
            [minLat, minLon],
            [maxLat, maxLon],
          ],
          { padding: [20, 20] }
        );
      }
    }, [rows, map]);
    return null;
  }

  function MapInner({ rows }: { rows: MapPoint[] }) {
    const valid = useMemo(
      () => rows.filter((r) => Number.isFinite(Number(r.lat)) && Number.isFinite(Number(r.lon))),
      [rows]
    );

    const latMed = valid.length
      ? valid.map((r) => Number(r.lat)).sort((a, b) => a - b)[Math.floor(valid.length / 2)]
      : 48.8566;
    const lonMed = valid.length
      ? valid.map((r) => Number(r.lon)).sort((a, b) => a - b)[Math.floor(valid.length / 2)]
      : 2.3522;

    // fond Carto Light no-labels avec fallback OSM
    const [tileUrl, setTileUrl] = useState(
      "https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png"
    );
    useEffect(() => {
      const img = new Image();
      img.onerror = () => setTileUrl("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png");
      img.src = "https://a.basemaps.cartocdn.com/light_nolabels/3/4/2.png";
    }, []);

    return (
      <div className="map-wrap h-360">
        <MapContainer
          center={[latMed, lonMed]}
          zoom={12}
          className="tile-bg"
          style={{ width: "100%", height: "100%" }}
        >
          <TileLayer
            url={tileUrl}
            attribution='&copy; OpenStreetMap, &copy; <a href="https://carto.com/">CARTO</a>'
            detectRetina
          />
          <FitBounds rows={valid} />
          {valid.map((r) => (
            <CircleMarker
              key={r.station_id}
              center={[Number(r.lat), Number(r.lon)]}
              radius={6}
              pathOptions={{
                color: r.color,
                weight: 0.8,
                fillColor: r.color,
                fillOpacity: 0.85,
              }}
            >
              <Tooltip className="tooltip-dark">
                <div style={{ display: "grid", gap: 4 }}>
                  <div>
                    <b>{r.name ?? r.station_id}</b>
                  </div>
                  {r.n != null && (
                    <div className="mono" style={{ fontSize: 12, opacity: 0.85 }}>
                      {r.station_id} · n={r.n.toLocaleString("fr-FR")}
                    </div>
                  )}
                  {Number.isFinite(Number(r.mae_model)) && (
                    <div>
                      MAE modèle: <b>{Number(r.mae_model).toFixed(2)}</b>
                    </div>
                  )}
                  {Number.isFinite(Number(r.mae_baseline)) && (
                    <div>
                      MAE base: <b>{Number(r.mae_baseline).toFixed(2)}</b>
                    </div>
                  )}
                  {Number.isFinite(Number(r.lift_pct)) && (
                    <div>
                      Lift: <b>{Number(r.lift_pct).toFixed(1)}%</b>
                    </div>
                  )}
                </div>
              </Tooltip>
            </CircleMarker>
          ))}
        </MapContainer>

        {/* Légende harmonisée CSS */}
        <div className="cluster-legend">
          <div className="cluster-legend__title">Lift par station</div>
          <div className="cluster-legend__row">
            <span className="cluster-legend__dot" style={{ background: "#10b981" }} />
            <span>Top (lift positif)</span>
          </div>
          <div className="cluster-legend__row">
            <span className="cluster-legend__dot" style={{ background: "#ef4444" }} />
            <span>Bottom (lift négatif)</span>
          </div>
        </div>
      </div>
    );
  }

  return MapInner;
}, { ssr: false });

/* ───────────────────────── Helpers ───────────────────────── */
function ok<T>(r: PromiseSettledResult<T>): T | null {
  return r.status === "fulfilled" ? r.value : null;
}
function fmtPct(x?: number | string | null, digits = 1) {
  const v = Number(x);
  return Number.isFinite(v) ? `${v.toFixed(digits)}%` : "—";
}
function fmtNum(x?: number | string | null, digits = 2) {
  const v = Number(x);
  return Number.isFinite(v) ? v.toFixed(digits) : "—";
}
function fmtInt(x?: number | string | null) {
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

function getLatLng(meta?: StationMeta | null): [number, number] | null {
  if (!meta) return null;
  const lat =
    (meta as any).lat ?? (meta as any).latitude ?? (meta as any).Lat ?? (meta as any).Latitude;
  const lon =
    (meta as any).lon ?? (meta as any).lng ?? (meta as any).longitude ?? (meta as any).Lon ?? (meta as any).Longitude;
  const la = Number(lat);
  const lo = Number(lon);
  if (!Number.isFinite(la) || !Number.isFinite(lo)) return null;
  return [la, lo];
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
  const [st24h, setSt24h] = useState<StationTimeseries | null>(null);

  // Index stations: id → meta
  const [stationsIdx, setStationsIdx] = useState<Record<string, StationMeta>>({});

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Charge manifest + index stations au montage
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const [m, idx] = await Promise.all([
          getPerformanceManifest(),
          loadStationsIndexFromArrayJson("/data/stations.index.json").catch(() => ({})),
        ]);
        if (!alive) return;
        setManifest(m);
        setStationsIdx(idx as Record<string, StationMeta>);

        const hs = Array.isArray(m.horizons) && m.horizons.length ? m.horizons : [15, 60];
        if (!hs.includes(h)) setH(hs[0]);
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

  // Données pour l’horizon courant (inclut la série 24 h via route API unique)
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
          getPerformanceStationTimeseries(h), // ⬅️ unique call
        ]);
        if (!alive) return;

        setKpis(ok(res[0]));
        setDaily(ok(res[1]));
        setByHour(ok(res[2]));
        setByDow(ok(res[3]));
        setByStation(ok(res[4]));
        setLift(ok(res[5]));
        setSt24h(ok(res[6]));

        const failures = res.filter((r): r is PromiseRejectedResult => r.status === "rejected");
        setError(failures.length ? "Une ou plusieurs requêtes ont échoué." : null);
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

  const generatedAt = kpis?.generated_at ?? manifest?.generated_at ?? null;
  const barStatus: LoadingBarStatus = loading ? "loading" : error ? "error" : "success";

  /* ───────── KPI BAR ───────── */
  const kpiItems: KpiItem[] = useMemo((): KpiItem[] => {
    const liftPct =
      Number.isFinite(Number(kpis?.lift_vs_baseline)) ? Number(kpis!.lift_vs_baseline) * 100 : null;

    return [
      { label: "Stations",       value: kpis?.n_stations,          fmt: (v) => fmtInt(v) },
      { label: "Lignes (n)",     value: kpis?.n_rows,              fmt: (v) => fmtInt(v) },
      { label: "MAE — Modèle",   value: kpis?.mae_model,           fmt: (v) => fmtNum(v, 2) },
      { label: "MAE — Baseline", value: kpis?.mae_baseline,        fmt: (v) => fmtNum(v, 2) },
      { label: "Lift vs baseline", value: liftPct,                 fmt: (v) => fmtPct(v, 1) },
      { label: "Coverage préd.",   value: kpis?.coverage_pred_pct, fmt: (v) => fmtPct(v, 1) },
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

  /* ───────── Courbes (line plots) ───────── */
  // 1) Lift quotidien (line)
  const liftCurve: Partial<Plotly.PlotData> | null = useMemo(() => {
    const pts = lift?.points ?? [];
    if (!pts.length) return null;
    const x = pts.map((p) => p.date);
    const y = pts.map((p) =>
      Number.isFinite(Number(p.lift_vs_baseline)) ? Number(p.lift_vs_baseline) * 100 : null
    );
    return {
      x, y,
      type: "scatter",
      mode: "lines",
      name: `Lift (%) h=${h}`,
      connectgaps: false,
      hovertemplate: "%{x} — %{y:.1f}%<extra></extra>",
    };
  }, [lift, h]);

  // 2) MAE par jour → line plot
  const dailyLines: Partial<Plotly.PlotData>[] = useMemo(() => {
    const rows = daily?.rows ?? [];
    if (!rows.length) return [];
    const x = rows.map((r) => r.date);
    const m = rows.map((r) => (Number.isFinite(Number(r.mae_model)) ? Number(r.mae_model) : null));
    const b = rows.map((r) => (Number.isFinite(Number(r.mae_baseline)) ? Number(r.mae_baseline) : null));
    return [
      { x, y: b, type: "scatter", mode: "lines+markers", name: "Baseline — MAE", connectgaps: false },
      { x, y: m, type: "scatter", mode: "lines+markers", name: "Modèle — MAE",   connectgaps: false },
    ];
  }, [daily]);

  // 3) Par heure → line plot
  const byHourLines: Partial<Plotly.PlotData>[] = useMemo(() => {
    const rows = byHour?.rows ?? [];
    if (!rows.length) return [];
    const x = rows.map((r) => r.hour);
    const m = rows.map((r) => (Number.isFinite(Number(r.mae_model)) ? Number(r.mae_model) : null));
    const b = rows.map((r) => (Number.isFinite(Number(r.mae_baseline)) ? Number(r.mae_baseline) : null));
    return [
      { x, y: b, type: "scatter", mode: "lines+markers", name: "Baseline — MAE" },
      { x, y: m, type: "scatter", mode: "lines+markers", name: "Modèle — MAE" },
    ];
  }, [byHour]);

  // 4) Par jour de semaine → line plot
  const byDowLines: Partial<Plotly.PlotData>[] = useMemo(() => {
    const rows = byDow?.rows ?? [];
    if (!rows.length) return [];
    const label = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"];
    const x = rows.map((r) => label[r.dow] ?? String(r.dow));
    const m = rows.map((r) => (Number.isFinite(Number(r.mae_model)) ? Number(r.mae_model) : null));
    const b = rows.map((r) => (Number.isFinite(Number(r.mae_baseline)) ? Number(r.mae_baseline) : null));
    return [
      { x, y: b, type: "scatter", mode: "lines+markers", name: "Baseline — MAE" },
      { x, y: m, type: "scatter", mode: "lines+markers", name: "Modèle — MAE" },
    ];
  }, [byDow]);

  // 5) Série 24 h — Observé / Modèle / Baseline (depuis StationTimeseries)
  const series24h: Partial<Plotly.PlotData>[] = useMemo(() => {
    if (!st24h || !st24h.ts?.length) return [];
    // Convertir les timestamps UTC → labels HH:MM locaux (n’affecte pas l’axe)
    const t = st24h.ts.map((s) => {
      const d = new Date(s);
      return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
    });
    const yt = (st24h.y_true ?? []).map((x) => (Number.isFinite(Number(x)) ? Number(x) : null));
    const yp = (st24h.y_pred ?? []).map((x) => (Number.isFinite(Number(x)) ? Number(x) : null));
    const ybArr = (st24h as any).y_base ?? []; // champ défini comme y_base côté API
    const yb = (ybArr as Array<number | null>).map((x) =>
      Number.isFinite(Number(x)) ? Number(x) : null
    );
    return [
      { x: t, y: yt, type: "scatter", mode: "lines+markers", name: "Observé" },
      { x: t, y: yp, type: "scatter", mode: "lines+markers", name: "Modèle" },
      { x: t, y: yb, type: "scatter", mode: "lines+markers", name: "Baseline" },
    ];
  }, [st24h]);

  // ── Stations Top/Bottom lift ─────────────────────────
  const stationsTopBottom = useMemo(() => {
    const rows = byStation?.rows ?? [];
    const wLift = (x: StationRow) =>
      Number.isFinite(Number(x.lift_vs_baseline)) ? Number(x.lift_vs_baseline) : -Infinity;
    const minN = 30;
    const filtered = rows.filter((r: StationRow) => Number(r.n) >= minN);
    const tops = [...filtered].sort((a: StationRow, b: StationRow) => wLift(b) - wLift(a)).slice(0, 20);
    const bots = [...filtered].sort((a: StationRow, b: StationRow) => wLift(a) - wLift(b)).slice(0, 20);
    return { tops, bots };
  }, [byStation]);

  // Dictionnaire id → nom
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

  // Données carte (couleurs identiques à overview)
  const mapRows = useMemo<MapPoint[]>(() => {
    const rows: MapPoint[] = [];
    const pushRow = (r: StationRow, color: string) => {
      const meta = stationsIdx[r.station_id] as StationMeta | undefined;
      if (!meta) return;
      const ll = getLatLng(meta);
      if (!ll) return;
      rows.push({
        station_id: r.station_id,
        name: (meta as any).name ?? r.station_id,
        lat: ll[0],
        lon: ll[1],
        color,
        mae_model: Number.isFinite(Number(r.mae_model)) ? Number(r.mae_model) : null,
        mae_baseline: Number.isFinite(Number(r.mae_baseline)) ? Number(r.mae_baseline) : null,
        lift_pct: Number.isFinite(Number(r.lift_vs_baseline)) ? Number(r.lift_vs_baseline) * 100 : null,
        n: Number.isFinite(Number(r.n)) ? Number(r.n) : null,
      });
    };
    (stationsTopBottom.tops || []).forEach((r: StationRow) => pushRow(r, "#10b981")); // vert
    (stationsTopBottom.bots || []).forEach((r: StationRow) => pushRow(r, "#ef4444")); // rouge
    return rows;
  }, [stationsTopBottom, stationsIdx]);

  return (
    <div className="monitoring">
      <Head>
        <title>Monitoring — Model / Performance</title>
        <meta
          name="description"
          content="Comparatif modèle vs baseline, lift, séries MAE, découpes, série 24 h et carte des stations."
        />
      </Head>

      <main className="page" style={{ paddingTop: "calc(var(--header-h, 70px) + 12px)" }}>
        <MonitoringNav
          title="Model — Performance"
          subtitle={`MAE/RMSE vs baseline, lift, séries & stations (h=${h} min)`}
          generatedAt={generatedAt ?? undefined}
          extraActions={[{ label: "Explainability", href: "/monitoring/model/explainability" }]}
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

        {/* KPIs */}
        <section className="mt-4">
          <h2>Résumé — KPIs (h={h} min)</h2>
          <KpiBar items={kpiItems} dense />
          {metaLine && <div className="kpi-bar-meta">{metaLine}</div>}
        </section>

        {/* Lift */}
        <section className="mt-6">
          <h2>Lift quotidien</h2>
          <div className="plot-card">
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
        </section>

        {/* MAE par jour — LINES */}
        <section className="mt-6">
          <h2>Séries temporelles — MAE (Modèle vs Baseline)</h2>
          <div className="plot-card">
            {dailyLines.length ? (
              <Plot
                data={dailyLines as Plotly.Data[]}
                layout={chartLayout({
                  height: 360,
                  margin: { l: 54, r: 10, t: 10, b: 60 },
                  yaxis: { title: { text: "MAE" } },
                  xaxis: { title: { text: "Date (locale)" }, tickangle: -30 },
                  hovermode: "x unified",
                })}
                config={chartConfig}
                className="plot plot--line"
              />
            ) : (
              <div className="empty">Pas de séries quotidiennes.</div>
            )}
          </div>
        </section>

        {/* Découpes — LINES */}
        <section className="mt-6">
          <h2>Découpes — heure & jour de semaine</h2>
          <div className="grid-2">
            <div className="plot-card">
              <h3>Par heure (locale)</h3>
              {byHourLines.length ? (
                <Plot
                  data={byHourLines as Plotly.Data[]}
                  layout={chartLayout({
                    height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 36 },
                    yaxis: { title: { text: "MAE" } },
                    xaxis: { title: { text: "Heure" } },
                    hovermode: "x unified",
                  })}
                  config={chartConfig}
                  className="plot plot--sm"
                />
              ) : (
                <div className="empty">—</div>
              )}
            </div>

            <div className="plot-card">
              <h3>Par jour de semaine</h3>
              {byDowLines.length ? (
                <Plot
                  data={byDowLines as Plotly.Data[]}
                  layout={chartLayout({
                    height: 320,
                    margin: { l: 54, r: 10, t: 10, b: 36 },
                    yaxis: { title: { text: "MAE" } },
                    xaxis: { title: { text: "Jour" } },
                    hovermode: "x unified",
                  })}
                  config={chartConfig}
                  className="plot plot--sm"
                />
              ) : (
                <div className="empty">—</div>
              )}
            </div>
          </div>
        </section>

        {/* Série 24 h — Observé / Modèle / Baseline */}
        <section className="mt-6">
          <h2>Série 24 h — Observé / Modèle / Baseline</h2>
          <div className="plot-card">
            {series24h.length ? (
              <Plot
                data={series24h as Plotly.Data[]}
                layout={chartLayout({
                  height: 340,
                  margin: { l: 54, r: 10, t: 10, b: 40 },
                  yaxis: { title: { text: "Nb vélos" } },
                  xaxis: { title: { text: `Heure locale` } },
                  hovermode: "x unified",
                })}
                config={chartConfig}
                className="plot plot--sm"
              />
            ) : (
              <div className="empty">Pas de série 24 h disponible.</div>
            )}
          </div>
          {st24h?.station_id && (
            <div className="small mt-2">
              Station échantillon : <b>{st24h.station_id}</b> · h={st24h.h} min
            </div>
          )}
        </section>

        {/* Stations — listes */}
        <section className="mt-6">
          <h2>Stations — meilleurs / moins bons lifts</h2>
          <div className="grid-2">
            <div className="card">
              <h3>Top 20 (lift)</h3>
              {stationsTopBottom.tops.length ? (
                <StationList rows={stationsTopBottom.tops} kind="top" nameIndex={nameIndex} />
              ) : (
                <div className="empty">Pas de stations (échantillon trop faible).</div>
              )}
            </div>
            <div className="card">
              <h3>Bottom 20 (lift)</h3>
              {stationsTopBottom.bots.length ? (
                <StationList rows={stationsTopBottom.bots} kind="bottom" nameIndex={nameIndex} />
              ) : (
                <div className="empty">Pas de stations (échantillon trop faible).</div>
              )}
            </div>
          </div>
          <div className="small mt-2">
            Filtre n ≥ 30. Lift = (MAE_baseline − MAE_modèle) / MAE_baseline.
          </div>
        </section>

        {/* Carte — intégration identique overview (map-block/map-wrap du CSS fourni) */}
        <section className="mt-6" style={{ marginBottom: 40 }}>
          <h2>Carte — Top vs Worst stations (lift)</h2>
          <div className="map-block">
            {mapRows.length ? (
              <SnapshotMap rows={mapRows} />
            ) : (
              <div className="empty">Aucune donnée carte.</div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

/* ───────────────────────── UI widget ───────────────────────── */
function StationList({
  rows,
  kind,
  nameIndex,
}: {
  rows: StationRow[];
  kind: "top" | "bottom";
  nameIndex?: Record<string, string>;
}) {
  const abs = (x: number | null) => (x == null ? 0 : Math.abs(x));
  const lifts = rows.map((r: StationRow) =>
    Number.isFinite(Number(r.lift_vs_baseline)) ? Number(r.lift_vs_baseline) : null
  );
  const maxAbsLift = Math.max(1e-9, ...lifts.map((v) => abs(v as number | null))); // évite 0

  return (
    <div className="table-scroll" style={{ overflowX: "hidden" }}>
      <div
        className="table-grid"
        style={{
          ["--cols" as any]: "minmax(0,1fr) 78px 78px 66px",
          minWidth: 0,
          gap: "4px",
        }}
      >
        <div className="table-head table-head--sticky">Station</div>
        <div className="table-head table-head--sticky">MAE base</div>
        <div className="table-head table-head--sticky">MAE modèle</div>
        <div className="table-head table-head--sticky">Lift</div>

        {rows.map((r: StationRow) => {
          const displayName =
            (nameIndex && nameIndex[r.station_id]) ||
            ((r as any).name as string | undefined) ||
            r.station_id;

          const lift = Number.isFinite(Number(r.lift_vs_baseline)) ? Number(r.lift_vs_baseline) : null;
          const liftPct = lift != null ? lift * 100 : null;

          const widthPct =
            lift != null ? Math.max(0, Math.min(100, (abs(lift) / maxAbsLift) * 100)) : 0;

          const maeM = Number(r.mae_model);
          const maeB = Number(r.mae_baseline);
          const better = lift != null && lift > 0;

          const showIdBelow = displayName !== r.station_id;

          return (
            <div key={r.station_id} className="table-row">
              {/* Col 1: Station */}
              <div className="table-cell">
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
                    {showIdBelow ? `${r.station_id} · ` : ""}n={fmtInt(r.n)}
                  </div>
                </div>

                {/* Barre normalisée sur max(|lift|) */}
                <div className="bar" style={{ height: 5, marginTop: 5, width: "min(75%, 320px)" }}>
                  <div
                    className={`bar__fill ${better ? "" : "bar__fill--danger"}`}
                    style={{ width: `${widthPct}%` }}
                    aria-hidden
                  />
                </div>
              </div>

              {/* Col 2-3-4 */}
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
