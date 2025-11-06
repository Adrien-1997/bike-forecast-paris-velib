// ui/pages/monitoring/overview.tsx
import Head from "next/head";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import dynamic from "next/dynamic";
import type * as Plotly from "plotly.js";
import MonitoringNav from "@/components/monitoring/MonitoringNav";
import LoadingBar, { type LoadingBarStatus } from "@/components/common/LoadingBar";
import KpiBar, { fmtPct, fmtInt } from "@/components/monitoring/KpiBar";
import { chartConfig, chartLayout } from "@/lib/plotlyTheme";

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
      Chargement du graphique…
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
        const la = Number(r.lat), lo = Number(r.lon);
        if (la < minLat) minLat = la;
        if (la > maxLat) maxLat = la;
        if (lo < minLon) minLon = lo;
        if (lo > maxLon) maxLon = lo;
      }
      if (minLat <= maxLat && minLon <= maxLon) {
        map.fitBounds([[minLat, minLon], [maxLat, maxLon]], { padding: [20, 20] });
      }
    }, [rows, map]);
    return null;
  }

  function MapInner({ rows }: { rows: MapRow[] }) {
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

    const [tileUrl, setTileUrl] = useState(
      "https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png"
    );
    useEffect(() => {
      const img = new Image();
      img.onerror = () => setTileUrl("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png");
      img.src = "https://a.basemaps.cartocdn.com/light_nolabels/3/4/2.png";
    }, []);

    return (
      <div className="map-wrap h-520" style={{ position: "relative", width: "100%", height: "100%" }}>
        <MapContainer
          center={[latMed, lonMed]}
          zoom={12}
          className="tile-bg"
          style={{ height: "100%", width: "100%" }}
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
            const rad = Math.max(3, Math.min(9, Math.sqrt(Math.max(0, Number(r.bikes ?? 0))) + (sat ? 2 : 0)));
            return (
              <CircleMarker
                key={r.station_id}
                center={[Number(r.lat), Number(r.lon)]}
                radius={rad}
                pathOptions={{ color: col, weight: 0.8, fillColor: col, fillOpacity: 0.85 }}
              >
                <Tooltip className="tooltip-dark">
                  <div style={{ display: "grid", gap: 4 }}>
                    <div><b>{r.name}</b></div>
                    <div>vélos: {Number.isFinite(Number(r.bikes)) ? Number(r.bikes) : "?"}</div>
                    <div>places: {Number.isFinite(Number(r.docks_avail)) ? Number(r.docks_avail) : "?"}</div>
                    {pen && <div style={{ color: "#ef4444" }}>pénurie</div>}
                    {sat && <div style={{ color: "#3b82f6" }}>saturation</div>}
                    {/* Lien désactivé (alignement avec la règle "non cliquable") */}
                    <div className="small" style={{ opacity: 0.7 }}>Voir dynamique (lien désactivé)</div>
                  </div>
                </Tooltip>
              </CircleMarker>
            );
          })}
        </MapContainer>

        {/* Légende snapshot (bulle) */}
        <div
          className="cluster-legend"
          style={{ right: 8, bottom: 8 }}
        >
          <div className="cluster-legend__title">Snapshot</div>
          <div className="cluster-legend__row">
            <span className="cluster-legend__dot" style={{ background: "#ef4444" }} />
            <span>Pénurie</span>
          </div>
          <div className="cluster-legend__row">
            <span className="cluster-legend__dot" style={{ background: "#3b82f6" }} />
            <span>Saturation</span>
          </div>
          <div className="cluster-legend__row">
            <span className="cluster-legend__dot" style={{ background: "#10b981" }} />
            <span>OK</span>
          </div>
        </div>
      </div>
    );
  }

  return MapInner;
}, { ssr: false });

/* ───────────────────────── Page ───────────────────────── */
export default function OverviewPage() {
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

        const [kpisRes, distRes, todayRes, refMedianRes, lagsRes, snapMapRes, tensionRes] =
          await Promise.allSettled(calls);

        setKpis(ok(kpisRes));
        setDist(ok(distRes));
        setToday(ok(todayRes));
        setRefMedian(ok(refMedianRes));
        setLags(ok(lagsRes));
        setSnapMap(ok(snapMapRes));
        setTension(ok(tensionRes));

        const allResults = [kpisRes, distRes, todayRes, refMedianRes, lagsRes, snapMapRes, tensionRes];
        const failures = allResults.filter((r): r is PromiseRejectedResult => r.status === "rejected");
        if (failures.length > 0) {
          const msg =
            failures.map((f) => String((f.reason && (f.reason.message ?? f.reason)) || "request failed")).join(" | ") ||
            "API error";
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
    return () => { alive = false; };
  }, []);

  const generatedAt =
    kpis?.generated_at ?? today?.generated_at ?? refMedian?.generated_at ?? snapMap?.generated_at;

  const barStatus: LoadingBarStatus = loading ? "loading" : error ? "error" : "success";

  return (
    <div className="monitoring">
      <Head>
        <title>Monitoring — Réseau / Aperçu</title>
        <meta name="description" content="KPIs réseau, snapshot, courbes et carte." />
      </Head>

      <main className="page" style={{ paddingTop: "calc(var(--header-h, 70px) + 12px)" }}>
        <MonitoringNav
          title="Réseau — Aperçu"
          subtitle="KPIs, distribution snapshot, courbes journalières & carte"
          generatedAt={generatedAt}
          extraActions={[
            { label: "Stations", href: "/monitoring/network/stations" },
            { label: "Dynamiques", href: "/monitoring/network/dynamics" },
          ]}
        />

        <LoadingBar status={barStatus} />
        {error && <div className="alert error" style={{ marginTop: 8 }}>{error}</div>}

        {/* KPIs Snapshot — via KpiBar */}
        <section className="mt-4">
          <h2>Résumé — instantané</h2>
          <KpiBar
            items={[
              { label: "Stations actives", value: fmtInt(kpis?.stations_active) },
              { label: "Hors ligne", value: fmtInt(kpis?.stations_offline) },
              { label: "≥1 vélo", value: fmtPct(kpis?.availability_bike_pct, 1) },
              { label: "≥1 place", value: fmtPct(kpis?.availability_dock_pct, 1) },
              { label: "Couverture (récente)", value: fmtPct(Number(kpis?.coverage_pct ?? NaN), 1) },
            ]}
            dense
          />
          <div className="kpi-bar-meta">
            Fenêtre : {kpis?.last_days ?? "—"} j · Réf : {kpis?.ref_days ?? "—"} j · Schéma v{ kpis?.schema_version ?? "—" }
            <span style={{ marginLeft: 8, opacity: 0.8 }}>({kpis?.snapshot_ts_local ?? "—"})</span>
          </div>
        </section>

        {/* === Carte snapshot === */}
        <section className="mt-6">
          <h2>Carte des stations — instantané</h2>
          <div className="map-block" style={{ height: 520 }}>
            <SnapshotMap rows={snapMap?.rows ?? []} />
          </div>
          <div className="figure-note small">
            Basemap : Carto Light (no labels). Rouge = pénurie ; Bleu = saturation ; Vert = OK. La taille des points est proportionnelle à √(vélos).
          </div>
        </section>

        {/* === Distribution snapshot === */}
        <section className="mt-6">
          <h2>Instantané — distribution</h2>
          {Array.isArray(dist) && dist.length ? (
            <>
              <div className="card plot-card">
                <h3>Distribution des états du snapshot — proportion des stations (%)</h3>
                <Plot
                  data={
                    ([{
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
                      hovertemplate: "%{x} — %{y:.1f}%<extra></extra>",
                    }] as unknown) as Plotly.Data[]
                  }
                  layout={chartLayout({
                    height: 280,
                    yaxis: {
                      title: { text: "Proportion des stations (%)" },
                      range: [0, 100],
                      ticksuffix: "%",
                    },
                    xaxis: { title: { text: "État (catégories)" } },
                  })}
                  config={chartConfig}
                  className="plot plot--sm"
                />
              </div>
              <div className="figure-note small">
                Lecture : parts des stations par état instantané (pénurie, saturation, etc.). La somme approche 100 % (arrondis).
              </div>
            </>
          ) : (
            <div className="card plot-card">
              <div className="empty">Distribution indisponible.</div>
            </div>
          )}
        </section>

        {/* Courbes du jour vs référence */}
        <section className="mt-6">
          <h2>Hier (UTC) vs Référence</h2>
          {(() => {
            const curveToday =
              today?.points?.length
                ? ({
                    x: today.points.map((p) => p.hhmm),
                    y: today.points.map((p) => (Number.isFinite(Number(p.pct)) ? Number(p.pct) : null)),
                    type: "scatter",
                    mode: "lines",
                    name: "J−1 (UTC)",
                    connectgaps: false,
                    hovertemplate: "%{x} — %{y:.1f}%<extra>J−1</extra>",
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
                    name: "Référence (médiane)",
                    connectgaps: false,
                    hovertemplate: "%{x} — %{y:.1f}%<extra>Référence</extra>",
                  } as Partial<Plotly.PlotData>)
                : null;

            return curveToday || curveRef ? (
              <>
                <div className="card plot-card">
                  <h3>Disponibilité ≥1 vélo — courbe journalière (J−1 UTC) vs médiane de référence</h3>
                  <Plot
                    data={[curveRef, curveToday].filter(Boolean) as Plotly.Data[]}
                    layout={chartLayout({
                      height: 380,
                      xaxis: { title: { text: "Heure (UTC, HH:MM)" } },
                      yaxis: {
                        title: { text: "Stations avec ≥1 vélo (%)" },
                        range: [0, 100],
                        ticksuffix: "%",
                      },
                    })}
                    config={chartConfig}
                    className="plot plot--lg"
                  />
                </div>
                <div className="figure-note small">
                  Séries agrégées par pas de 5 minutes. J−1 couvre 00:00–23:55 (UTC). La référence est la médiane historique sur la fenêtre indiquée.
                </div>
              </>
            ) : (
              <div className="card plot-card">
                <div className="empty">Courbes indisponibles.</div>
              </div>
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
            <div className="empty">Comparaison indisponible.</div>
          )}
        </section>

        {/* Stations en tension — Top pénurie / Top saturation en tableaux (compact, 1 colonne fluide) */}
        <section className="mt-6">
          <h2>Stations en tension (fenêtre récente)</h2>

          {(() => {
            const rows = tension?.rows ?? [];

            const topPenRows = rows
              .filter((r) => Number.isFinite(Number(r.penury_rate)))
              .sort((a, b) => Number(b.penury_rate) - Number(a.penury_rate))
              .slice(0, 20)
              .map((r) => {
                const id = String(r.station_id);
                const name = stationsIdx[id]?.name ?? (r as any).name ?? id;
                const pct = Number(r.penury_rate) * 100;
                return { station_id: id, name, pct };
              });

            const topSatRows = rows
              .filter((r) => Number.isFinite(Number(r.saturation_rate)))
              .sort((a, b) => Number(b.saturation_rate) - Number(a.saturation_rate))
              .slice(0, 20)
              .map((r) => {
                const id = String(r.station_id);
                const name = stationsIdx[id]?.name ?? (r as any).name ?? id;
                const pct = Number(r.saturation_rate) * 100;
                return { station_id: id, name, pct };
              });

            return (
              <div className="grid-2">
                {/* Top pénurie — compact */}
                <div className="card">
                  <h3 style={{ margin: "6px 0 10px 0", fontSize: 16 }}>Top pénurie</h3>
                  {topPenRows.length ? (
                    <div className="table-scroll" style={{ overflowX: "hidden" }}>
                      <div
                        className="table-grid"
                        style={{
                          ["--cols" as any]: "minmax(0,1fr)",
                          minWidth: 0,
                        }}
                      >
                        <div className="table-head table-head--sticky">Station</div>

                        {topPenRows.map((r) => (
                          <div key={`pen-${r.station_id}`} className="table-row">
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
                                    title={r.name}
                                  >
                                    {r.name}
                                  </div>
                                  <div className="mono" style={{ fontSize: 12, opacity: 0.7 }}>
                                    {r.station_id}
                                  </div>
                                </div>
                                <div style={{ fontWeight: 700 }}>{fmtPct(r.pct, 1)}</div>
                              </div>

                              <div
                                className="bar"
                                style={{
                                  height: 5,
                                  marginTop: 6,
                                  width: "min(52%, 240px)",
                                }}
                              >
                                <div
                                  className="bar__fill"
                                  style={{
                                    width: `${Math.max(0, Math.min(100, r.pct))}%`,
                                    background: "#ef4444",
                                  }}
                                  aria-hidden
                                />
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="empty">—</div>
                  )}
                </div>

                {/* Top saturation — compact */}
                <div className="card">
                  <h3 style={{ margin: "6px 0 10px 0", fontSize: 16 }}>Top saturation</h3>
                  {topSatRows.length ? (
                    <div className="table-scroll" style={{ overflowX: "hidden" }}>
                      <div
                        className="table-grid"
                        style={{
                          ["--cols" as any]: "minmax(0,1fr)",
                          minWidth: 0,
                        }}
                      >
                        <div className="table-head table-head--sticky">Station</div>

                        {topSatRows.map((r) => (
                          <div key={`sat-${r.station_id}`} className="table-row">
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
                                    title={r.name}
                                  >
                                    {r.name}
                                  </div>
                                  <div className="mono" style={{ fontSize: 12, opacity: 0.7 }}>
                                    {r.station_id}
                                  </div>
                                </div>
                                <div style={{ fontWeight: 700 }}>{fmtPct(r.pct, 1)}</div>
                              </div>

                              <div
                                className="bar"
                                style={{
                                  height: 5,
                                  marginTop: 6,
                                  width: "min(52%, 240px)",
                                }}
                              >
                                <div
                                  className="bar__fill"
                                  style={{
                                    width: `${Math.max(0, Math.min(100, r.pct))}%`,
                                    background: "#3b82f6",
                                  }}
                                  aria-hidden
                                />
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="empty">—</div>
                  )}
                </div>
              </div>
            );
          })()}

          {tension && (
            <div className="small mt-2">
              Schéma v{tension.schema_version} — généré {tension.generated_at}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

/* ───────────────────────── UI atoms (branchés sur monitoring.css) ───────────────────────── */
function Row({ children }: { children: ReactNode }) {
  const items = Array.isArray(children) ? children : [children];
  // aligne à droite la dernière colonne (%)
  return (
    <div className="table-row">
      {items.map((child, i) => {
        const isLast = i === items.length - 1;
        return (
          <div key={i} className={`table-cell ${isLast ? "table-cell--right" : ""}`}>
            {child}
          </div>
        );
      })}
    </div>
  );
}

function HeaderCell({ children }: { children: ReactNode }) {
  return <div className="table-head table-head--sticky">{children}</div>;
}

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
            <div className="small" style={{ opacity: 0.75 }}>{x.label}</div>
            <div style={{ fontSize: 18, fontWeight: 700 }}>{fmtPct(x.v ?? undefined, 1)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
