// ui/pages/monitoring/overview.tsx
import Head from "next/head";
import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import type * as Plotly from "plotly.js";
import MonitoringNav from "@/components/monitoring/MonitoringNav";
import LoadingBar, { type LoadingBarStatus } from "@/components/common/LoadingBar";
import KpiBar, { fmtPct, fmtInt } from "@/components/monitoring/KpiBar";

import {
  getOverviewKpis,
  getOverviewSnapshotDistribution,
  getOverviewTodayCurve,
  getOverviewRefMedianCurve,
  getOverviewKpisTodayVsLags,
  getOverviewSnapshotMap,
  getOverviewStationsTension,
  fetchStationsIndex,
  type OverviewKpis,
  type OverviewSnapshotDistribution,
  type OverviewTodayCurve,
  type OverviewRefMedianCurve,
  type OverviewKpisTodayVsLags,
  type OverviewSnapshotMap,
  type OverviewStationsTension,
  type StationMeta,
} from "@/lib/services/monitoring/network_overview";

/* ───────────────────────── Plotly (client only) ───────────────────────── */
const Plot = dynamic(() => import("react-plotly.js").then((m) => m.default), {
  ssr: false,
  loading: () => (
    <div style={{ height: 320, display: "grid", placeItems: "center", opacity: 0.7 }}>
      Loading chart…
    </div>
  ),
});

/* ───────────────────────── Utils ───────────────────────── */
function ok<T>(r: PromiseSettledResult<T>): T | null {
  return r.status === "fulfilled" ? r.value : null;
}

/* ───────────────────────── Mini Map (snapshot) ───────────────────────── */
type MapRow = OverviewSnapshotMap["rows"][number];

const SnapshotMap = dynamic(async () => {
  const RL = await import("react-leaflet");
  const { MapContainer, TileLayer, CircleMarker, Tooltip, useMap } = RL as any;
  const { useEffect, useMemo, useState } = await import("react");

  function FitBounds({ rows }: { rows: MapRow[] }) {
    const map = useMap();
    useEffect(() => {
      const pts = rows.filter(
        (r) => Number.isFinite(Number(r.lat)) && Number.isFinite(Number(r.lon))
      );
      if (!pts.length) return;
      let minLat = 90,
        maxLat = -90,
        minLon = 180,
        maxLon = -180;
      for (const r of pts) {
        const la = Number(r.lat),
          lo = Number(r.lon);
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

  function MapInner({ rows }: { rows: MapRow[] }) {
    const valid = useMemo(
      () =>
        rows.filter(
          (r) => Number.isFinite(Number(r.lat)) && Number.isFinite(Number(r.lon))
        ),
      [rows]
    );
    const latMed = valid.length
      ? valid.map((r) => Number(r.lat)).sort((a, b) => a - b)[Math.floor(valid.length / 2)]
      : 48.8566;
    const lonMed = valid.length
      ? valid.map((r) => Number(r.lon)).sort((a, b) => a - b)[Math.floor(valid.length / 2)]
      : 2.3522;

    const [tileUrl, setTileUrl] = useState(
      "https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png"
    );
    useEffect(() => {
      const img = new Image();
      img.onerror = () => setTileUrl("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png");
      img.src = "https://a.basemaps.cartocdn.com/light_nolabels/3/4/2.png";
    }, []);

    return (
      <div style={{ position: "relative", width: "100%", height: "100%" }}>
        <MapContainer
          center={[latMed, lonMed]}
          zoom={12}
          style={{ height: "100%", width: "100%", background: "#fff" }}
        >
          <TileLayer
            url={tileUrl}
            attribution='&copy; OpenStreetMap, &copy; <a href="https://carto.com/">CARTO</a>'
            detectRetina
          />
          <FitBounds rows={valid} />
          {valid.map((r) => {
            const pen = r.is_penury === 1;
            const sat = r.is_saturation === 1;
            const col = pen ? "#ef4444" : sat ? "#3b82f6" : "#10b981";
            const rad = Math.max(
              3,
              Math.min(9, Math.sqrt(Math.max(0, Number(r.bikes ?? 0))) + (sat ? 2 : 0))
            );
            return (
              <CircleMarker
                key={r.station_id}
                center={[Number(r.lat), Number(r.lon)]}
                radius={rad}
                pathOptions={{ color: col, weight: 0.8, fillColor: col, fillOpacity: 0.85 }}
              >
                <Tooltip>
                  <div style={{ display: "grid", gap: 4 }}>
                    <div>
                      <b>{r.name}</b>
                    </div>
                    <div>
                      bikes: {Number.isFinite(Number(r.bikes)) ? Number(r.bikes) : "?"}
                    </div>
                    <div>
                      docks:{" "}
                      {Number.isFinite(Number(r.docks_avail)) ? Number(r.docks_avail) : "?"}
                    </div>
                    {pen && <div style={{ color: "#ef4444" }}>penury</div>}
                    {sat && <div style={{ color: "#3b82f6" }}>saturation</div>}
                    <a
                      href={`/monitoring/network/dynamics?station_id=${encodeURIComponent(
                        r.station_id
                      )}`}
                      style={{ textDecoration: "underline" }}
                    >
                      View dynamics →
                    </a>
                  </div>
                </Tooltip>
              </CircleMarker>
            );
          })}
        </MapContainer>

        {/* Légende snapshot */}
        <div
          style={{
            position: "absolute",
            right: 8,
            bottom: 8,
            zIndex: 1000,
            background: "rgba(255,255,255,0.92)",
            color: "#111",
            borderRadius: 10,
            padding: "8px 10px",
            fontSize: 12,
            boxShadow: "0 2px 10px rgba(0,0,0,0.15)",
            border: "1px solid #0001",
          }}
        >
          <div style={{ fontWeight: 700, marginBottom: 6 }}>Snapshot</div>
          <div style={{ display: "grid", gap: 4 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span
                style={{
                  width: 12,
                  height: 12,
                  borderRadius: 999,
                  background: "#ef4444",
                  border: "1px solid #0002",
                }}
              />
              <span>Pénurie</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span
                style={{
                  width: 12,
                  height: 12,
                  borderRadius: 999,
                  background: "#3b82f6",
                  border: "1px solid #0002",
                }}
              />
              <span>Saturation</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span
                style={{
                  width: 12,
                  height: 12,
                  borderRadius: 999,
                  background: "#10b981",
                  border: "1px solid #0002",
                }}
              />
              <span>OK</span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return MapInner;
}, { ssr: false });

/* ───────────────────────── Page ───────────────────────── */
export default function OverviewPage() {
  // State
  const [kpis, setKpis] = useState<OverviewKpis | null>(null);
  const [dist, setDist] = useState<OverviewSnapshotDistribution | null>(null);
  const [today, setToday] = useState<OverviewTodayCurve | null>(null);
  const [refMedian, setRefMedian] = useState<OverviewRefMedianCurve | null>(null);
  const [lags, setLags] = useState<OverviewKpisTodayVsLags | null>(null);
  const [snapMap, setSnapMap] = useState<OverviewSnapshotMap | null>(null);
  const [tension, setTension] = useState<OverviewStationsTension | null>(null);
  const [stationsIdx, setStationsIdx] = useState<Record<string, StationMeta>>({});

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load all (corrigé pour refléter correctement l'erreur)
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setLoading(true);

        const calls: [
          Promise<OverviewKpis>,
          Promise<OverviewSnapshotDistribution>,
          Promise<OverviewTodayCurve>,
          Promise<OverviewRefMedianCurve>,
          Promise<OverviewKpisTodayVsLags>,
          Promise<OverviewSnapshotMap>,
          Promise<OverviewStationsTension>
        ] = [
          getOverviewKpis(),
          getOverviewSnapshotDistribution(),
          getOverviewTodayCurve(),
          getOverviewRefMedianCurve(),
          getOverviewKpisTodayVsLags(),
          getOverviewSnapshotMap(),
          getOverviewStationsTension(),
        ];

        const [
          kpisRes,
          distRes,
          todayRes,
          refMedianRes,
          lagsRes,
          snapMapRes,
          tensionRes,
        ] = await Promise.allSettled(calls);

        setKpis(ok(kpisRes));
        setDist(ok(distRes));
        setToday(ok(todayRes));
        setRefMedian(ok(refMedianRes));
        setLags(ok(lagsRes));
        setSnapMap(ok(snapMapRes));
        setTension(ok(tensionRes));

        const allResults = [
          kpisRes,
          distRes,
          todayRes,
          refMedianRes,
          lagsRes,
          snapMapRes,
          tensionRes,
        ];
        const failures = allResults.filter(
          (r): r is PromiseRejectedResult => r.status === "rejected"
        );
        if (failures.length > 0) {
          const msg =
            failures
              .map((f) => String((f.reason && (f.reason.message ?? f.reason)) || "request failed"))
              .join(" | ") || "API error";
          setError(msg);
        } else {
          setError(null);
        }

        fetchStationsIndex()
          .then((idx) => alive && setStationsIdx(idx))
          .catch(() => alive && setStationsIdx({}));
      } catch (e: any) {
        if (alive) setError(String(e?.message ?? e ?? "Unknown error"));
      } finally {
        if (alive) setLoading(false);
      }
    })();

    return () => {
      alive = false;
    };
  }, []);

  const generatedAt =
    kpis?.generated_at ?? today?.generated_at ?? refMedian?.generated_at ?? snapMap?.generated_at;

  const barStatus: LoadingBarStatus = loading ? "loading" : error ? "error" : "success";

  return (
    <div className="monitoring">
      <Head>
        <title>Monitoring — Network / Overview</title>
        <meta name="description" content="KPIs réseau, snapshot, courbes et carte." />
        {/* ⚠️ Place les CSS globaux (leaflet, monitoring) dans _app.tsx / globals.css. Pas d’injection locale ici. */}
      </Head>

      {/* Contenu principal (header/footer injectés par _app.tsx) */}
      <main className="page" style={{ paddingTop: "calc(var(--header-h, 70px) + 12px)" }}>
        <MonitoringNav
          title="Network — Overview"
          subtitle="KPIs, snapshot distribution, daily curves & map"
          generatedAt={generatedAt}
          extraActions={[
            { label: "Stations", href: "/monitoring/network/stations" },
            { label: "Dynamics", href: "/monitoring/network/dynamics" },
          ]}
        />

        {/* Loading / Error → LoadingBar uniforme */}
        <LoadingBar status={barStatus} />

        {/* KPIs Snapshot — via KpiBar */}
        <section className="mt-4">
          <h2>Résumé — snapshot</h2>

          <KpiBar
            items={[
              { label: "Stations actives", value: fmtInt(kpis?.stations_active) },
              { label: "Hors ligne", value: fmtInt(kpis?.stations_offline) },
              { label: "≥1 vélo", value: fmtPct(kpis?.availability_bike_pct, 1) },
              { label: "≥1 place", value: fmtPct(kpis?.availability_dock_pct, 1) },
              {
                label: "Couverture (récente)",
                value: fmtPct(Number(kpis?.coverage_pct ?? NaN), 1),
              },
            ]}
            dense
          />

          {/* 🆕 Métadonnées en dehors des cartes */}
          <div className="kpi-bar-meta">
            Window: {kpis?.last_days ?? "—"} days · Ref: {kpis?.ref_days ?? "—"} days · Schema v
            {kpis?.schema_version ?? "—"}
            <span style={{ marginLeft: 8, opacity: 0.8 }}>
              ({kpis?.snapshot_ts_local ?? "—"})
            </span>
          </div>
        </section>


        {/* === Carte snapshot === */}
        <section className="mt-6">
          <h2>Stations map — Snapshot</h2>

          <div className="map-block">
            <div className="map-wrap" style={{ width: "100%", height: 520 }}>
              {snapMap?.rows?.length ? (
                <SnapshotMap rows={snapMap.rows} />
              ) : (
                <div className="empty">No snapshot map data.</div>
              )}
            </div>
          </div>

          <div className="small mt-2">
            Basemap: Carto Light (no labels) · Red=pénurie · Blue=saturation · Green=OK (size ~ √bikes).
          </div>
        </section>

        {/* === Distribution snapshot === */}
        <section className="mt-6">
          <h2>Snapshot — distribution</h2>
          <div className="card plot-card">
            <h3 style={{ margin: "6px 0 10px 0", fontSize: 16 }}>Distribution (%)</h3>
            {Array.isArray(dist) && dist.length ? (
              <Plot
                data={
                  ([
                    {
                      x: dist.map(
                        (x) =>
                          (
                            {
                              bike_avail: "≥1 vélo",
                              dock_avail: "≥1 place",
                              penury: "pénurie",
                              saturation: "saturation",
                            } as Record<string, string>
                          )[x.metric] ?? x.metric
                      ),
                      y: dist.map((x) => Number(x.pct)),
                      type: "bar",
                      name: "Snapshot (%)",
                    },
                  ] as unknown) as Plotly.Data[]
                }
                layout={{
                  autosize: true,
                  height: 280,
                  margin: { l: 48, r: 10, t: 10, b: 36 },
                  yaxis: { title: { text: "%" }, range: [0, 100] },
                  xaxis: { title: { text: "State" } },
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                }}
                config={{ displayModeBar: false, responsive: true }}
                className="plot plot--sm"
              />
            ) : (
              <div className="empty">No snapshot distribution.</div>
            )}
          </div>
        </section>

        {/* Courbes du jour vs référence */}
        <section className="mt-6">
          <h2>Hier (UTC) vs Référence</h2>
          {(() => {
            const curveToday =
              today?.points?.length
                ? ({
                    x: today.points.map((p) => p.hhmm),
                    y: today.points.map((p) =>
                      Number.isFinite(Number(p.pct)) ? Number(p.pct) : null
                    ),
                    type: "scatter",
                    mode: "lines",
                    name: "Hier (UTC)",
                    connectgaps: false,
                  } as Partial<Plotly.PlotData>)
                : null;

            const curveRef =
              refMedian?.median?.length
                ? ({
                    x: refMedian.median.map((p) => p.hhmm),
                    y: refMedian.median.map((p) =>
                      Number.isFinite(Number(p.pct_median)) ? Number(p.pct_median) : null
                    ),
                    type: "scatter",
                    mode: "lines",
                    name: "Référence (UTC)",
                    connectgaps: false,
                  } as Partial<Plotly.PlotData>)
                : null;

            return curveToday || curveRef ? (
              <>
                <Plot
                  data={[curveRef, curveToday].filter(Boolean) as Plotly.Data[]}
                  layout={{
                    autosize: true,
                    height: 380,
                    margin: { l: 52, r: 10, t: 30, b: 40 },
                    xaxis: { title: { text: "Heure (UTC)" } },
                    yaxis: { title: { text: "Stations avec ≥1 vélo (%)" }, range: [0, 100] },
                    legend: { orientation: "h" },
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                    hovermode: "x unified",
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  className="plot plot--lg"
                />
                <div className="small mt-2">
                  Les courbes sont en UTC. « Hier (UTC) » correspond au jour UTC complet (00:00–23:55).
                </div>
              </>
            ) : (
              <div className="empty">Curves unavailable.</div>
            );
          })()}
        </section>

        {/* KPIs Today vs J-7/J-14/J-21 */}
        <section className="mt-6">
          <h2>Aujourd’hui vs J-7 / J-14 / J-21</h2>
          {lags ? (
            <div className="grid-4">
              <CardSmall
                title="≥1 vélo (temps %)"
                values={[
                  { label: "J", v: lags.today.avail_bike },
                  { label: "J-7", v: lags.lags["J-7"].avail_bike },
                  { label: "J-14", v: lags.lags["J-14"].avail_bike },
                  { label: "J-21", v: lags.lags["J-21"].avail_bike },
                ]}
              />
              <CardSmall
                title="≥1 place (temps %)"
                values={[
                  { label: "J", v: lags.today.avail_dock },
                  { label: "J-7", v: lags.lags["J-7"].avail_dock },
                  { label: "J-14", v: lags.lags["J-14"].avail_dock },
                  { label: "J-21", v: lags.lags["J-21"].avail_dock },
                ]}
              />
              <CardSmall
                title="Pénurie (temps %)"
                values={[
                  { label: "J", v: lags.today.pen },
                  { label: "J-7", v: lags.lags["J-7"].pen },
                  { label: "J-14", v: lags.lags["J-14"].pen },
                  { label: "J-21", v: lags.lags["J-21"].pen },
                ]}
              />
              <CardSmall
                title="Saturation (temps %)"
                values={[
                  { label: "J", v: lags.today.sat },
                  { label: "J-7", v: lags.lags["J-7"].sat },
                  { label: "J-14", v: lags.lags["J-14"].sat },
                  { label: "J-21", v: lags.lags["J-21"].sat },
                ]}
              />
            </div>
          ) : (
            <div className="empty">Comparisons unavailable.</div>
          )}
        </section>

        {/* Stations en tension */}
        <section className="mt-6">
          <h2>Stations en tension (fenêtre récente)</h2>
          <div className="grid-2">
            <div className="card">
              <h3 style={{ margin: "6px 0 10px 0", fontSize: 16 }}>Top pénurie</h3>
              {(() => {
                const rows = tension?.rows ?? [];
                const stations = rows
                  .filter((r) => Number.isFinite(Number(r.penury_rate)))
                  .sort((a, b) => Number(b.penury_rate) - Number(a.penury_rate))
                  .slice(0, 20)
                  .map((r) => {
                    const id = String(r.station_id);
                    const meta = stationsIdx[id];
                    return {
                      station_id: id,
                      name: meta?.name ?? id,
                      penury_pct: Number(r.penury_rate) * 100,
                      href: `/monitoring/network/dynamics?station_id=${encodeURIComponent(id)}`,
                    };
                  });

                return stations.length ? (
                  <ul
                    style={{
                      margin: 0,
                      padding: 0,
                      listStyle: "none",
                      display: "grid",
                      gap: 8,
                    }}
                  >
                    {stations.map((r) => (
                      <li key={r.station_id}>
                        <a href={r.href} style={{ textDecoration: "none", color: "inherit" }}>
                          <div
                            style={{
                              display: "grid",
                              gridTemplateColumns: "minmax(0,1fr) 64px",
                              alignItems: "center",
                              gap: 10,
                            }}
                          >
                            <div style={{ overflow: "hidden" }}>
                              <div style={{ whiteSpace: "nowrap", textOverflow: "ellipsis" }}>
                                <b>{r.name}</b>{" "}
                                <span style={{ opacity: 0.6 }}>({r.station_id})</span>
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
                                    width: `${Math.max(0, Math.min(100, r.penury_pct ?? 0))}%`,
                                    height: "100%",
                                    background: "#ef4444",
                                  }}
                                />
                              </div>
                            </div>
                            <div style={{ textAlign: "right", fontWeight: 700 }}>
                              {fmtPct(r.penury_pct ?? undefined, 1)}
                            </div>
                          </div>
                        </a>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <div className="empty">—</div>
                );
              })()}
            </div>

            <div className="card">
              <h3 style={{ margin: "6px 0 10px 0", fontSize: 16 }}>Top saturation</h3>
              {(() => {
                const rows = tension?.rows ?? [];
                const stations = rows
                  .filter((r) => Number.isFinite(Number(r.saturation_rate)))
                  .sort((a, b) => Number(b.saturation_rate) - Number(a.saturation_rate))
                  .slice(0, 20)
                  .map((r) => {
                    const id = String(r.station_id);
                    const meta = stationsIdx[id];
                    return {
                      station_id: id,
                      name: meta?.name ?? id,
                      saturation_pct: Number(r.saturation_rate) * 100,
                      href: `/monitoring/network/dynamics?station_id=${encodeURIComponent(id)}`,
                    };
                  });

                return stations.length ? (
                  <ul
                    style={{
                      margin: 0,
                      padding: 0,
                      listStyle: "none",
                      display: "grid",
                      gap: 8,
                    }}
                  >
                    {stations.map((r) => (
                      <li key={r.station_id}>
                        <a href={r.href} style={{ textDecoration: "none", color: "inherit" }}>
                          <div
                            style={{
                              display: "grid",
                              gridTemplateColumns: "minmax(0,1fr) 64px",
                              alignItems: "center",
                              gap: 10,
                            }}
                          >
                            <div style={{ overflow: "hidden" }}>
                              <div style={{ whiteSpace: "nowrap", textOverflow: "ellipsis" }}>
                                <b>{r.name}</b>{" "}
                                <span style={{ opacity: 0.6 }}>({r.station_id})</span>
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
                                    width: `${Math.max(
                                      0,
                                      Math.min(100, r.saturation_pct ?? 0)
                                    )}%`,
                                    height: "100%",
                                    background: "#3b82f6",
                                  }}
                                />
                              </div>
                            </div>
                            <div style={{ textAlign: "right", fontWeight: 700 }}>
                              {fmtPct(r.saturation_pct ?? undefined, 1)}
                            </div>
                          </div>
                        </a>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <div className="empty">—</div>
                );
              })()}
            </div>
          </div>

          {tension && (
            <div className="small mt-2">
              Schema v{tension.schema_version} — generated {tension.generated_at}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

/* ───────────────────────── UI atoms ───────────────────────── */
function CardSmall({
  title,
  values,
}: {
  title: string;
  values: { label: string; v: number | null }[];
}) {
  return (
    <div className="card">
      <h3 style={{ margin: "6px 0 10px 0", fontSize: 16 }}>{title}</h3>
      <div className="grid-4">
        {values.map((x) => (
          <div key={x.label} style={{ textAlign: "center" }}>
            <div className="small" style={{ opacity: 0.75 }}>
              {x.label}
            </div>
            <div style={{ fontSize: 18, fontWeight: 700 }}>{fmtPct(x.v ?? undefined, 1)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
