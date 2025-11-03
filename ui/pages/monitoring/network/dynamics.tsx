// ui/pages/monitoring/network/dynamics.tsx
import Head from "next/head";
import dynamic from "next/dynamic";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useRouter } from "next/router";
import type * as Plotly from "plotly.js";
import MonitoringNav from "@/components/monitoring/MonitoringNav";
import LoadingBar, { type LoadingBarStatus } from "@/components/common/LoadingBar";
import KpiBar, { fmtPct, fmtInt } from "@/components/monitoring/KpiBar";
import { chartConfig, chartLayout } from "@/lib/plotlyTheme";

import {
  getDynamicsHeatmapsProfiles,
  getDynamicsHourlyPenSat,
  getDynamicsEpisodes,
  getDynamicsTensionByStation,
  fetchStationsIndex,
  type HeatmapsProfilesDoc,
  type HourlyDoc,
  type EpisodesDoc,
  type TensionByStationDoc,
  type StationMeta,
} from "@/lib/services/monitoring/network_dynamics";

/* ───────────────── Plotly (client only) ───────────────── */
const Plot = dynamic(() => import("react-plotly.js").then((m) => m.default), {
  ssr: false,
  loading: () => <div className="empty">Chargement du graphique…</div>,
});

/* ───────────────── Utils ───────────────── */
function ok<T>(r: PromiseSettledResult<T>): T | null {
  return r.status === "fulfilled" ? r.value : null;
}
function clamp01(x: number | null | undefined): number | null {
  if (!Number.isFinite(Number(x))) return null;
  return Math.max(0, Math.min(1, Number(x)));
}
function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

/* ───────────────── Episodes Map ───────────────── */
type EpisodePoint = {
  station_id: string;
  name: string;
  lat: number;
  lon: number;
  type: "penury" | "saturation";
  start_utc: string;
  end_utc: string;
  duration_min: number | null;
};

const EpisodesMap = dynamic(async () => {
  const RL = await import("react-leaflet");
  const { MapContainer, TileLayer, CircleMarker, Tooltip, useMap } = RL as any;
  const { useEffect, useMemo, useState } = await import("react");

  function FitBounds({ rows }: { rows: EpisodePoint[] }) {
    const map = useMap();
    useEffect(() => {
      if (!rows.length) return;
      let minLat = 90, maxLat = -90, minLon = 180, maxLon = -180;
      for (const r of rows) {
        if (r.lat < minLat) minLat = r.lat;
        if (r.lat > maxLat) maxLat = r.lat;
        if (r.lon < minLon) minLon = r.lon;
        if (r.lon > maxLon) maxLon = r.lon;
      }
      if (minLat <= maxLat && minLon <= maxLon) {
        map.fitBounds([[minLat, minLon], [maxLat, maxLon]], { padding: [20, 20] });
      }
    }, [rows, map]);
    return null;
  }

  function MapInner({ rows }: { rows: EpisodePoint[] }) {
    const valid = useMemo(
      () => rows.filter((r) => Number.isFinite(r.lat) && Number.isFinite(r.lon)),
      [rows]
    );
    const latMed = valid.length ? [...valid].map(r => r.lat).sort((a,b)=>a-b)[Math.floor(valid.length/2)] : 48.8566;
    const lonMed = valid.length ? [...valid].map(r => r.lon).sort((a,b)=>a-b)[Math.floor(valid.length/2)] : 2.3522;

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
          className="leaflet-container"
          style={{ width: "100%", height: "100%" }}
        >
          <TileLayer
            url={tileUrl}
            attribution='&copy; OpenStreetMap, &copy; <a href="https://carto.com/">CARTO</a>'
            detectRetina
          />
          <FitBounds rows={valid} />
          {valid.map((r, i) => {
            const col = r.type === "penury" ? "#ef4444" : "#3b82f6";
            return (
              <CircleMarker
                key={`${r.station_id}-${i}`}
                center={[r.lat, r.lon]}
                radius={5}
                pathOptions={{ color: col, weight: 0.8, fillColor: col, fillOpacity: 0.85 }}
              >
                <Tooltip className="tooltip-dark">
                  <div style={{ display: "grid", gap: 4 }}>
                    <div>
                      <b>{r.name}</b>{" "}
                      <span style={{ opacity: 0.6 }}>({r.station_id})</span>
                    </div>
                    <div>
                      Type : <b style={{ color: col }}>{r.type}</b>
                    </div>
                    <div>Début : {new Date(r.start_utc).toLocaleString("fr-FR")}</div>
                    <div>Fin : {new Date(r.end_utc).toLocaleString("fr-FR")}</div>
                    <div>Durée : {fmtInt(r.duration_min)}</div>
                    {/* lien retiré volontairement */}
                    <div className="small" style={{ opacity: 0.7 }}>Voir épisodes (lien désactivé)</div>
                  </div>
                </Tooltip>
              </CircleMarker>
            );
          })}
        </MapContainer>

        {/* Légende */}
        <div className="cluster-legend">
          <div className="cluster-legend__title">Épisodes</div>
          <div className="cluster-legend__row">
            <span className="cluster-legend__dot" style={{ background: "#ef4444" }} />
            <span>Pénurie</span>
          </div>
          <div className="cluster-legend__row">
            <span className="cluster-legend__dot" style={{ background: "#3b82f6" }} />
            <span>Saturation</span>
          </div>
          <div className="cluster-legend__meta">{valid.length} points</div>
        </div>
      </div>
    );
  }

  return MapInner;
}, { ssr: false });

/* ───────────────── Page ───────────────── */
export default function NetworkDynamicsPage() {
  const router = useRouter();
  const qStation = typeof router.query.station_id === "string" ? router.query.station_id : "";
  const [stationId, setStationId] = useState<string>(qStation);

  const [heat, setHeat] = useState<HeatmapsProfilesDoc | null>(null);
  const [hourly, setHourly] = useState<HourlyDoc | null>(null);
  const [episodes, setEpisodes] = useState<EpisodesDoc | null>(null);
  const [tension, setTension] = useState<TensionByStationDoc | null>(null);
  const [stationsIdx, setStationsIdx] = useState<Record<string, StationMeta>>({});

  const [dowSel, setDowSel] = useState<number>(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const barStatus: LoadingBarStatus = loading ? "loading" : error ? "error" : "success";

  useEffect(() => { setStationId(qStation); }, [qStation]);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setLoading(true);
        const [rHeat, rHourly, rEpisodes, rTension] = await Promise.allSettled([
          getDynamicsHeatmapsProfiles(),
          getDynamicsHourlyPenSat(),
          getDynamicsEpisodes(),
          getDynamicsTensionByStation(),
        ]);
        if (!alive) return;

        setHeat(ok(rHeat));
        setHourly(ok(rHourly));
        setEpisodes(ok(rEpisodes));
        setTension(ok(rTension));

        const results = [rHeat, rHourly, rEpisodes, rTension];
        const failures = results.filter((r): r is PromiseRejectedResult => r.status === "rejected");
        setError(
          failures.length
            ? failures.map((f) => String((f.reason && (f.reason.message ?? f.reason)) || "request failed")).join(" | ")
            : null
        );

        fetchStationsIndex()
          .then((idx) => { if (alive) setStationsIdx(idx); })
          .catch(() => {});
      } catch (e: any) {
        if (alive) setError(String(e?.message ?? e));
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, []);

  const generatedAt =
    heat?.generated_at ?? hourly?.generated_at ?? episodes?.generated_at ?? tension?.generated_at;

  // KPIs
  const stationsCount = useMemo(() => {
    const n = tension?.rows?.length;
    return Number.isFinite(Number(n)) ? Number(n) : NaN;
  }, [tension]);

  const episodesCount = useMemo(() => {
    const n = episodes?.rows?.length;
    return Number.isFinite(Number(n)) ? Number(n) : NaN;
  }, [episodes]);

  const peakPenury = useMemo(() => {
    const rows = hourly?.rows ?? [];
    let m = 0;
    for (const r of rows) {
      const v = clamp01(r?.penury_rate);
      if (v != null) m = Math.max(m, v * 100);
    }
    return m || NaN;
  }, [hourly]);

  const peakSaturation = useMemo(() => {
    const rows = hourly?.rows ?? [];
    let m = 0;
    for (const r of rows) {
      const v = clamp01(r?.saturation_rate);
      if (v != null) m = Math.max(m, v * 100);
    }
    return m || NaN;
  }, [hourly]);

  const windowDays =
    episodes?.last_days ?? tension?.last_days ?? (heat as any)?.last_days ?? (heat as any)?.window_days ?? undefined;

  const schemaVersion =
    episodes?.schema_version ?? tension?.schema_version ?? (hourly as any)?.schema_version ?? (heat as any)?.schema_version ?? undefined;

  /* ───────────────── Heatmaps 7×24 — compacts ───────────────── */
  const compactPlotCardStyle = { padding: "8px 8px 6px 8px" } as const;
  const compactH3Style = { margin: "0 0 6px 0", fontSize: 16, lineHeight: 1.25 } as const;

  const heatmap = (title: string, matrix: (number | null)[][], isPct01 = false): JSX.Element => {
    const z =
      matrix?.map((row) =>
        row?.map((v) => (Number.isFinite(Number(v)) ? (isPct01 ? Number(v) * 100 : Number(v)) : null))
      ) ?? [];
    const y = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"];
    const x = [...Array(24)].map((_, i) => `${String(i).padStart(2, "0")}:00`);
    return (
      <div className="plot-card" style={compactPlotCardStyle}>
        <h3 style={compactH3Style}>{title}</h3>
        <Plot
          data={
            ([{
              z, x, y, type: "heatmap", hoverongaps: false,
              colorbar: { title: isPct01 ? "%" : "Occ" },
            }] as unknown) as Plotly.Data[]
          }
          layout={chartLayout({
            height: 300,
            margin: { l: 56, r: 8, t: 4, b: 36 },
            xaxis: { title: { text: "Heure (locale)" } },
            yaxis: { title: { text: "Jour" } },
          })}
          config={chartConfig}
          className="plot plot--lg"
        />
      </div>
    );
  };

  /* ───────────────── Profils par jour ───────────────── */
  const selectedProfile = useMemo(() => {
    const key = String(dowSel ?? 1);
    const arr = heat?.profiles_occ_by_dow?.[key] ?? [];
    return arr.map((v) => (Number.isFinite(Number(v)) ? Number(v) * 100 : null));
  }, [heat, dowSel]);

  const profileYMax = useMemo(() => {
    const vals = (selectedProfile || []).filter((v) => Number.isFinite(Number(v))) as number[];
    const maxv = vals.length ? Math.max(...vals) : 100;
    return clamp(Math.ceil(maxv + 5), 20, 100);
  }, [selectedProfile]);

  /* ───────────────── Barres horaires ───────────────── */
  const hourlyBars: Partial<Plotly.PlotData>[] = useMemo(() => {
    const rows = hourly?.rows ?? [];
    const hours = [...Array(24)].map((_, i) => i);
    return [
      {
        x: hours,
        y: hours.map((h) => {
          const r = rows.find((q) => q.hour === h);
          return clamp01(r?.penury_rate) != null ? Number(r!.penury_rate) * 100 : null;
        }),
        type: "bar",
        name: "Pénurie (%)",
      },
      {
        x: hours,
        y: hours.map((h) => {
          const r = rows.find((q) => q.hour === h);
          return clamp01(r?.saturation_rate) != null ? Number(r!.saturation_rate) * 100 : null;
        }),
        type: "bar",
        name: "Saturation (%)",
      },
    ];
  }, [hourly]);

  const hourlyYMax = useMemo(() => {
    const vals: number[] = [];
    for (const s of hourlyBars) {
      for (const v of ((s.y as (number | null)[]) ?? [])) {
        if (Number.isFinite(Number(v))) vals.push(Number(v));
      }
    }
    const maxv = vals.length ? Math.max(...vals) : 100;
    return clamp(Math.ceil(maxv + 5), 10, 100);
  }, [hourlyBars]);

  /* ───────────────── Episodes (filtre station) ───────────────── */
  const episodesFiltered = useMemo(() => {
    const rows = episodes?.rows ?? [];
    const sid = (stationId || "").trim();
    return sid ? rows.filter((r) => r.station_id === sid) : rows;
  }, [episodes, stationId]);

  /* ───────────────── Tension par station (recherche) ───────────────── */
  const [search, setSearch] = useState("");
  const tensionRows = useMemo(() => {
    const rows = tension?.rows ?? [];
    const q = (search || "").toLowerCase();
    const filtered = q
      ? rows.filter(
          (r) =>
            r.station_id.toLowerCase().includes(q) ||
            (r.name ?? "").toLowerCase().includes(q)
        )
      : rows;
    return filtered
      .map((r) => ({ ...r, _name: stationsIdx[r.station_id]?.name ?? r.name ?? r.station_id }))
      .sort((a, b) => Number(b.tension_index ?? 0) - Number(a.tension_index ?? 0))
      .slice(0, 200);
  }, [tension, search, stationsIdx]);

  /* ───────────────── Rows pour EpisodesMap ───────────────── */
  const episodePoints: EpisodePoint[] = useMemo(() => {
    const rows = episodesFiltered ?? [];
    const out: EpisodePoint[] = [];
    for (const r of rows) {
      const meta = stationsIdx[r.station_id];
      if (meta?.lat != null && meta?.lon != null) {
        out.push({
          station_id: r.station_id,
          name: meta.name ?? r.station_id,
          lat: meta.lat!,
          lon: meta.lon!,
          type: r.type,
          start_utc: r.start_utc,
          end_utc: r.end_utc,
          duration_min: r.duration_min ?? null,
        });
      }
    }
    return out;
  }, [episodesFiltered, stationsIdx]);

  return (
    <div className="monitoring">
      <Head>
        <title>Monitoring — Réseau / Dynamiques</title>
        <meta name="description" content="Dynamiques réseau : heatmaps, profils, épisodes et tension par station." />
      </Head>

      <main className="page" style={{ paddingTop: "calc(var(--header-h, 70px) + 12px)" }}>
        <MonitoringNav
          title="Réseau — Dynamiques"
          subtitle="Heatmaps 7×24, profils par jour, épisodes et tension"
          generatedAt={generatedAt}
          extraActions={[
            { label: "Aperçu", href: "/monitoring/network/overview" },
            { label: "Stations", href: "/monitoring/network/stations" },
          ]}
        />

        <LoadingBar status={barStatus} />
        {error && <div className="alert error" style={{ marginTop: 8 }}>{error}</div>}

        {/* KPIs */}
        <section className="mt-4">
          <h2>Résumé réseau</h2>
          <KpiBar
            dense
            items={[
              { label: "Stations couvertes", value: fmtInt(stationsCount) },
              { label: "Épisodes récents", value: fmtInt(episodesCount) },
              { label: "Pic pénurie (heure)", value: fmtPct(peakPenury, 1) },
              { label: "Pic saturation (heure)", value: fmtPct(peakSaturation, 1) },
            ]}
          />
          <div className="kpi-bar-meta">
            Fenêtre : {windowDays ?? "—"} j · Schéma v{schemaVersion ?? "—"} · Généré {generatedAt ?? "—"}
          </div>
        </section>

        {/* Heatmaps */}
        <section className="mt-6">
          <h2>Heatmaps 7×24</h2>
          {heat ? (
            <>
              {heatmap("Occupation moyenne (0..1)", heat.heatmap?.occ_mean ?? [], false)}
              <div className="figure-note small">
                Lecture : occupation moyenne par pas de 1 h, sur 7 jours (lignes) × 24 h (colonnes).
              </div>

              <div className="mt-3">
                {heatmap("Pénurie (%)", heat.heatmap?.penury_rate ?? [], true)}
              </div>
              <div className="figure-note small">
                Part horaire des stations en pénurie (0 % – 100 %).
              </div>

              <div className="mt-3">
                {heatmap("Saturation (%)", heat.heatmap?.saturation_rate ?? [], true)}
              </div>
              <div className="figure-note small">
                Part horaire des stations en saturation (0 % – 100 %).
              </div>
            </>
          ) : (
            <div className="plot-card" style={{ padding: "8px 8px 6px 8px" }}>
              <div className="empty">—</div>
            </div>
          )}
        </section>

        {/* Profils par jour */}
        <section className="mt-6">
          <h2>Profils d’occupation par jour</h2>
          <div className="filters" style={{ marginBottom: 8 }}>
            <label className="small" style={{ opacity: 0.8 }}>
              Jour :
            </label>
            {[1, 2, 3, 4, 5, 6, 0].map((d) => {
              const lbl = ["Dim", "Lun", "Mar", "Mer", "Jeu", "Ven", "Sam"][d];
              const val = d === 0 ? 0 : d;
              const active = val === (dowSel ?? 1);
              return (
                <button
                  key={d}
                  onClick={() => setDowSel(val)}
                  className={`btn ${active ? "btn--primary" : "btn--ghost"}`}
                >
                  {lbl}
                </button>
              );
            })}
          </div>

          <div className="plot-card" style={{ padding: "8px 8px 6px 8px" }}>
            {selectedProfile.length ? (
              <Plot
                data={
                  ([{
                    x: [...Array(24)].map((_, i) => `${String(i).padStart(2, "0")}:00`),
                    y: selectedProfile,
                    type: "scatter",
                    mode: "lines",
                    name: "Occupation (%)",
                    connectgaps: false,
                    hovertemplate: "%{x} — %{y:.1f}%<extra></extra>",
                  }] as unknown) as Plotly.Data[]
                }
                layout={chartLayout({
                  height: 330,
                  margin: { l: 56, r: 8, t: 4, b: 36 },
                  xaxis: { title: { text: "Heure (locale — jour sélectionné)" } },
                  yaxis: { title: { text: "%" }, range: [0, profileYMax], ticksuffix: "%" },
                })}
                config={chartConfig}
                className="plot plot--lg"
              />
            ) : (
              <div className="empty">Profil indisponible.</div>
            )}
          </div>
          <div className="figure-note small">
            Série horaire agrégée (médiane) par jour de semaine sélectionné.
          </div>
        </section>

        {/* Barres horaires pen/sat */}
        <section className="mt-6">
          <h2>Pénurie & Saturation par heure</h2>
          <div className="plot-card" style={{ padding: "8px 8px 6px 8px" }}>
            {hourlyBars?.length ? (
              <Plot
                data={hourlyBars as Plotly.Data[]}
                layout={chartLayout({
                  height: 312,
                  barmode: "group",
                  margin: { l: 56, r: 8, t: 4, b: 36 },
                  xaxis: { title: { text: "Heure (locale)" } },
                  yaxis: { title: { text: "%" }, range: [0, hourlyYMax], ticksuffix: "%" },
                  legend: { orientation: "h" },
                })}
                config={chartConfig}
                className="plot plot--sm"
              />
            ) : (
              <div className="empty">—</div>
            )}
          </div>
          <div className="figure-note small">
            Lecture : pour chaque heure locale, part des stations en état de pénurie ou de saturation.
          </div>
        </section>

        {/* Épisodes */}
        <section className="mt-6">
          <h2>Épisodes (fenêtre récente)</h2>
          <div className="filters">
            <label className="small" style={{ opacity: 0.8 }}>
              Filtrer station_id :
            </label>
            <input
              value={stationId}
              onChange={(e) => setStationId(e.target.value)}
              placeholder="ex : 12123"
              className="input"
            />
            <button
              onClick={() =>
                router.push(
                  { pathname: "/monitoring/network/dynamics", query: stationId ? { station_id: stationId } : {} },
                  undefined,
                  { shallow: true }
                )
              }
              className="btn btn--primary"
            >
              Appliquer
            </button>
            {episodes && (
              <span className="small" style={{ opacity: 0.7 }}>
                Fenêtre : {episodes.last_days} j
              </span>
            )}
          </div>

          {/* Carte */}
          <div className="map-block">
            <div className="map-wrap h-360">
              {episodePoints.length ? <EpisodesMap rows={episodePoints} /> : <div className="empty">Aucun point à afficher.</div>}
            </div>
          </div>
          <div className="figure-note small">
            Basemap : Carto Light (no labels). Rouge = pénurie ; Bleu = saturation. Un point par épisode détecté.
          </div>

          {/* Liste des épisodes (stations non cliquables) */}
          {episodesFiltered?.length ? (
            <div className="card mt-4">
              <h3 style={{ margin: "6px 0 10px 0", fontSize: 16 }}>Liste des épisodes détectés</h3>
              <div className="table-scroll">
                <div
                  className="table-grid"
                  style={{ ["--cols" as any]: "minmax(260px,1.2fr) 120px 1fr 1fr 120px 100px" }}
                >
                  <HeaderCell>Station</HeaderCell>
                  <HeaderCell>Type</HeaderCell>
                  <HeaderCell>Début (UTC)</HeaderCell>
                  <HeaderCell>Fin (UTC)</HeaderCell>
                  <HeaderCell>Durée (min)</HeaderCell>
                  <HeaderCell>Pas (#)</HeaderCell>

                  {episodesFiltered.slice(0, 1000).map((r, i) => {
                    const name = stationsIdx[r.station_id]?.name ?? r.station_id;
                    return (
                      <Row key={`${r.station_id}-${r.start_utc}-${i}`}>
                        {/* Colonne gauche style "performance", sans lien */}
                        <div style={{ minWidth: 0 }}>
                          <div
                            style={{
                              fontWeight: 600,
                              whiteSpace: "nowrap",
                              overflow: "hidden",
                              textOverflow: "ellipsis",
                              maxWidth: 380,
                            }}
                            title={name}
                          >
                            {name}
                          </div>
                          <div className="mono" style={{ fontSize: 12, opacity: 0.7 }}>
                            {r.station_id}
                          </div>
                        </div>

                        <div style={{ color: r.type === "penury" ? "#ef4444" : "#3b82f6" }}>{r.type}</div>
                        <div>{new Date(r.start_utc).toLocaleString("fr-FR")}</div>
                        <div>{new Date(r.end_utc).toLocaleString("fr-FR")}</div>
                        <div className="table-cell--right" style={{ fontVariantNumeric: "tabular-nums" }}>
                          {fmtInt(r.duration_min)}
                        </div>
                        <div className="table-cell--right" style={{ fontVariantNumeric: "tabular-nums" }}>
                          {fmtInt(r.steps)}
                        </div>
                      </Row>
                    );
                  })}
                </div>
              </div>
            </div>
          ) : (
            <div className="empty mt-4">Aucun épisode.</div>
          )}
        </section>

        {/* Tension par station (stations non cliquables) */}
        <section className="mt-6" style={{ marginBottom: 40 }}>
          <h2>Tension par station</h2>
          <div className="card">
            <div className="filters">
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Recherche station_id ou nom…"
                className="input"
                style={{ minWidth: 280 }}
              />
              {tension && (
                <span className="small" style={{ opacity: 0.7 }}>
                  Fenêtre : {tension.last_days} j
                </span>
              )}
            </div>

            {tensionRows?.length ? (
              <div className="table-scroll">
                <div
                  className="table-grid"
                  style={{ ["--cols" as any]: "minmax(260px,1.2fr) 120px 120px 120px 140px 100px" }}
                >
                  <HeaderCell>Station</HeaderCell>
                  <HeaderCell>Pénurie</HeaderCell>
                  <HeaderCell>Saturation</HeaderCell>
                  <HeaderCell>Occupation</HeaderCell>
                  <HeaderCell>Tension idx</HeaderCell>
                  <HeaderCell>Obs</HeaderCell>

                  {tensionRows.slice(0, 400).map((r, i) => {
                    const displayName = (r as any)._name;
                    return (
                      <Row key={`${r.station_id}-${i}`}>
                        {/* Colonne gauche style "performance", sans lien */}
                        <div style={{ minWidth: 0 }}>
                          <div
                            style={{
                              fontWeight: 600,
                              whiteSpace: "nowrap",
                              overflow: "hidden",
                              textOverflow: "ellipsis",
                              maxWidth: 380,
                            }}
                            title={displayName}
                          >
                            {displayName}
                          </div>
                          <div className="mono" style={{ fontSize: 12, opacity: 0.7 }}>
                            {r.station_id}
                          </div>
                        </div>

                        <div style={{ fontVariantNumeric: "tabular-nums" }}>
                          {fmtPct(Number(r.penury_rate ?? NaN) * 100, 1)}
                        </div>
                        <div style={{ fontVariantNumeric: "tabular-nums" }}>
                          {fmtPct(Number(r.saturation_rate ?? NaN) * 100, 1)}
                        </div>
                        <div style={{ fontVariantNumeric: "tabular-nums" }}>
                          {fmtPct(Number(r.occ_mean ?? NaN) * 100, 1)}
                        </div>
                        <div style={{ fontVariantNumeric: "tabular-nums" }}>
                          {fmtPct(Number(r.tension_index ?? NaN) * 100, 1)}
                        </div>
                        <div className="table-cell--right" style={{ fontVariantNumeric: "tabular-nums" }}>
                          {fmtInt(r.n_obs)}
                        </div>
                      </Row>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div className="empty">—</div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

/* ───────────────────────── UI atoms (alignés monitoring.css) ───────────────────────── */
function Row({ children }: { children: ReactNode }) {
  const items = Array.isArray(children) ? children : [children];
  return (
    <div className="table-row">
      {items.map((child, i) => {
        const alignRight = items.length >= 6 && (i === items.length - 1 || i === items.length - 2);
        return (
          <div key={i} className={`table-cell ${alignRight ? "table-cell--right" : ""}`}>
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
