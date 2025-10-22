// ui/pages/monitoring/network/dynamics.tsx
import Head from "next/head";
import Link from "next/link";
import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/router";
import type * as Plotly from "plotly.js";
import MonitoringNav from "@/components/monitoring/MonitoringNav";
import LoadingBar, { type LoadingBarStatus } from "@/components/common/LoadingBar";
import KpiBar from "@/components/monitoring/KpiBar";

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
  loading: () => (
    <div style={{ height: 320, display: "grid", placeItems: "center", opacity: 0.7 }}>
      Loading chart…
    </div>
  ),
});

/* ───────────────── Utils ───────────────── */
function ok<T>(r: PromiseSettledResult<T>): T | null {
  return r.status === "fulfilled" ? r.value : null;
}
function fmtPct(x?: number | null, digits = 1) {
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  return `${v.toFixed(digits)}%`;
}
function fmtInt(x?: number | null) {
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  return v.toLocaleString("fr-FR");
}
function clamp01(x: number | null | undefined): number | null {
  if (!Number.isFinite(Number(x))) return null;
  return Math.max(0, Math.min(1, Number(x)));
}
function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

/* ───────────────── Episodes Map (identique Stations) ───────────────── */
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
    const valid = useMemo(() => rows.filter(r => Number.isFinite(r.lat) && Number.isFinite(r.lon)), [rows]);
    const latMed = valid.length ? [...valid].map(r=>r.lat).sort((a,b)=>a-b)[Math.floor(valid.length/2)] : 48.8566;
    const lonMed = valid.length ? [...valid].map(r=>r.lon).sort((a,b)=>a-b)[Math.floor(valid.length/2)] : 2.3522;

    const [tileUrl, setTileUrl] = useState("https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png");
    useEffect(() => {
      const img = new Image();
      img.onerror = () => setTileUrl("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png");
      img.src = "https://a.basemaps.cartocdn.com/light_nolabels/3/4/2.png";
    }, []);

    return (
      <div className="map-wrap" style={{ width: "100%", height: 520 }}>
        <MapContainer center={[latMed, lonMed]} zoom={12} className="leaflet-container">
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
                <Tooltip>
                  <div style={{display:"grid", gap:4}}>
                    <div><b>{r.name}</b> <span style={{opacity:.6}}>({r.station_id})</span></div>
                    <div>Type : <b style={{ color: col }}>{r.type}</b></div>
                    <div>Début : {new Date(r.start_utc).toLocaleString("fr-FR")}</div>
                    <div>Fin : {new Date(r.end_utc).toLocaleString("fr-FR")}</div>
                    <div>Durée : {fmtInt(r.duration_min)}</div>
                    <a href={`/monitoring/network/dynamics?station_id=${encodeURIComponent(r.station_id)}`} style={{textDecoration:"underline"}}>
                      Voir épisodes →
                    </a>
                  </div>
                </Tooltip>
              </CircleMarker>
            );
          })}
        </MapContainer>

        {/* Légende identique Stations */}
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

  // ✅ barre de chargement uniforme (comme overview.tsx)
  const barStatus: LoadingBarStatus = loading ? "loading" : error ? "error" : "success";

  useEffect(() => { setStationId(qStation); }, [qStation]);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setLoading(true);
        const res = await Promise.allSettled([
          getDynamicsHeatmapsProfiles(),
          getDynamicsHourlyPenSat(),
          getDynamicsEpisodes(),
          getDynamicsTensionByStation(),
        ]);
        if (!alive) return;

        setHeat(ok(res[0]));
        setHourly(ok(res[1]));
        setEpisodes(ok(res[2]));
        setTension(ok(res[3]));

        fetchStationsIndex().then((idx) => alive && setStationsIdx(idx)).catch(()=>{});
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
    heat?.generated_at ?? hourly?.generated_at ?? episodes?.generated_at ?? tension?.generated_at;

  // ── KPIs pour KpiBar (même logique que Stations)
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
    episodes?.last_days ??
    tension?.last_days ??
    (heat as any)?.last_days ??
    (heat as any)?.window_days ??
    undefined;

  const schemaVersion =
    episodes?.schema_version ??
    tension?.schema_version ??
    (hourly as any)?.schema_version ??
    (heat as any)?.schema_version ??
    undefined;

  /* ───────────────── Heatmaps 7×24 — empilées ───────────────── */
  const heatmap = (title: string, matrix: (number | null)[][], isPct01 = false): JSX.Element => {
    const z = matrix?.map((row) => row?.map((v) => (Number.isFinite(Number(v)) ? (isPct01 ? Number(v) * 100 : Number(v)) : null))) ?? [];
    const y = ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"];
    const x = [...Array(24)].map((_, i) => `${String(i).padStart(2, "0")}:00`);
    return (
      <Plot
        data={[{ z, x, y, type: "heatmap", hoverongaps: false, colorbar: { title: isPct01 ? "%" : "Occ" } } as any]}
        layout={{
          autosize: true,
          height: 300,
          margin: { l: 60, r: 10, t: 30, b: 40 },
          title: { text: title, x: 0, y: 0.98, xanchor: "left", yanchor: "top" },
          xaxis: { side: "bottom" },
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
        }}
        config={{ displayModeBar: false, responsive: true }}
        className="plot plot--lg"
      />
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
      ? rows.filter((r) => r.station_id.toLowerCase().includes(q) || (r.name ?? "").toLowerCase().includes(q))
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
        <title>Monitoring — Network / Dynamics</title>
        <meta name="description" content="Dynamiques réseau: heatmaps, profils, épisodes, tension par station." />
      </Head>

      {/* Contenu principal (header/footer injectés par _app.tsx) */}
      <main className="page" style={{ paddingTop: "calc(var(--header-h, 70px) + 12px)" }}>
        <MonitoringNav
          title="Network — Dynamics"
          subtitle="Heatmaps 7×24, profils par jour, épisodes et tension"
          generatedAt={generatedAt}
          extraActions={[
            { label: "Overview", href: "/monitoring/network/overview" },
            { label: "Stations", href: "/monitoring/network/stations" },
          ]}
        />

        {/* ✅ LoadingBar uniforme, juste sous le MonitoringNav */}
        <LoadingBar status={barStatus} />

        {/* ───────────────── KPIs (KpiBar) ───────────────── */}
        <section className="mt-4">
          <h2>Network summary</h2>
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
            Window: {windowDays ?? "—"} days · Schema v{schemaVersion ?? "—"} · Generated {generatedAt ?? "—"}
          </div>
        </section>

        {/* Heatmaps — EMPILÉES */}
        <section className="mt-6">
          <h2>Heatmaps 7×24</h2>
          <div className="card">{heat ? heatmap("Occupation moyenne (0..1)", heat.heatmap?.occ_mean ?? [], false) : <div className="empty">—</div>}</div>
          <div className="card mt-4">{heat ? heatmap("Pénurie (%)", heat.heatmap?.penury_rate ?? [], true) : <div className="empty">—</div>}</div>
          <div className="card mt-4">{heat ? heatmap("Saturation (%)", heat.heatmap?.saturation_rate ?? [], true) : <div className="empty">—</div>}</div>
        </section>

        {/* Profils par jour */}
        <section className="mt-6">
          <h2>Profils d’occupation par jour</h2>
          <div className="filters" style={{ marginBottom: 8 }}>
            <label className="small" style={{ opacity: .8 }}>Jour :</label>
            {[1,2,3,4,5,6,0].map((d) => {
              const lbl = ["Dim","Lun","Mar","Mer","Jeu","Ven","Sam"][d];
              const val = d === 0 ? 0 : d;
              const active = val === (dowSel ?? 1);
              return (
                <button key={d} onClick={() => setDowSel(val)} className={`btn ${active ? "btn-primary" : ""}`}>
                  {lbl}
                </button>
              );
            })}
          </div>
          <div className="card">
            {selectedProfile.length ? (
              <Plot
                data={[{
                  x: [...Array(24)].map((_, i) => `${String(i).padStart(2, "0")}:00`),
                  y: selectedProfile, type: "scatter", mode: "lines", name: "Occupation (%)"
                } as any]}
                layout={{
                  autosize: true,
                  height: 340,
                  margin: { l: 52, r: 10, t: 20, b: 40 },
                  yaxis: { title: { text: "%" }, range: [0, profileYMax] },
                  xaxis: { title: { text: "Heure (locale — jour sélectionné)" } },
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                }}
                config={{ displayModeBar: false, responsive: true }}
                className="plot plot--lg"
              />
            ) : <div className="empty">Profil indisponible.</div>}
          </div>
        </section>

        {/* Barres horaires pen/sat */}
        <section className="mt-6">
          <h2>Pénurie & Saturation par heure</h2>
          <div className="card">
            {hourlyBars?.length ? (
              <Plot
                data={hourlyBars as Plotly.Data[]}
                layout={{
                  barmode: "group",
                  autosize: true,
                  height: 320,
                  margin: { l: 52, r: 10, t: 20, b: 40 },
                  xaxis: { title: { text: "Heure (locale)" } },
                  yaxis: { title: { text: "%" }, range: [0, hourlyYMax] },
                  legend: { orientation: "h" },
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                }}
                config={{ displayModeBar: false, responsive: true }}
                className="plot plot--sm"
              />
            ) : <div className="empty">—</div>}
          </div>
        </section>

        {/* Épisodes — même intégration carte que Stations */}
        <section className="mt-6">
          <h2>Épisodes (fenêtre récente)</h2>
          <div className="filters">
            <label className="small" style={{ opacity: .8 }}>Filtrer station_id :</label>
            <input
              value={stationId}
              onChange={(e)=>setStationId(e.target.value)}
              placeholder="ex: 12123"
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
              className="btn btn-primary"
            >
              Appliquer
            </button>
            {episodes && <span className="small" style={{ opacity: .7 }}>Fenêtre: {episodes.last_days} j</span>}
          </div>

          <div className="map-block">
            <div className="map-wrap" style={{ width: "100%", height: 520 }}>
              {episodePoints.length ? <EpisodesMap rows={episodePoints} /> : <div className="empty">Aucun point à afficher.</div>}
            </div>
          </div>

          {episodesFiltered?.length ? (
            <div className="card mt-4">
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                  <thead>
                    <tr style={{ textAlign: "left" }}>
                      <th>Station</th>
                      <th>Type</th>
                      <th>Début (UTC)</th>
                      <th>Fin (UTC)</th>
                      <th>Durée (min)</th>
                      <th>Pas (#)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {episodesFiltered.slice(0, 1000).map((r, i) => (
                      <tr key={`${r.station_id}-${r.start_utc}-${i}`} style={{ borderTop: "1px solid #374151" }}>
                        <td>
                          <Link href={`/monitoring/network/stations?station_id=${encodeURIComponent(r.station_id)}`} style={{ textDecoration: "underline" }}>
                            {stationsIdx[r.station_id]?.name ?? r.station_id}
                          </Link>
                        </td>
                        <td style={{ color: r.type === "penury" ? "#ef4444" : "#3b82f6" }}>{r.type}</td>
                        <td>{new Date(r.start_utc).toLocaleString("fr-FR")}</td>
                        <td>{new Date(r.end_utc).toLocaleString("fr-FR")}</td>
                        <td>{fmtInt(r.duration_min)}</td>
                        <td>{fmtInt(r.steps)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : <div className="empty mt-4">Aucun épisode.</div>}
        </section>

        {/* Tension par station */}
        <section className="mt-6">
          <h2>Tension par station</h2>
          <div className="card">
            <div className="filters">
              <input
                value={search}
                onChange={(e)=>setSearch(e.target.value)}
                placeholder="Recherche station_id ou nom…"
                className="input"
                style={{ minWidth: 280 }}
              />
              {tension && <span className="small" style={{ opacity: 0.7 }}>Fenêtre: {tension.last_days} j</span>}
            </div>
            {tensionRows?.length ? (
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                  <thead>
                    <tr style={{ textAlign: "left" }}>
                      <th>Station</th>
                      <th>Pénurie</th>
                      <th>Saturation</th>
                      <th>Occupation</th>
                      <th>Tension idx</th>
                      <th>Obs</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tensionRows.slice(0, 400).map((r, i) => (
                      <tr key={`${r.station_id}-${i}`} style={{ borderTop: "1px solid #374151" }}>
                        <td>
                          <div style={{ whiteSpace: "nowrap", textOverflow: "ellipsis", overflow: "hidden", maxWidth: 240 }}>
                            <b>{(r as any)._name}</b> <span style={{ opacity: 0.6 }}>({r.station_id})</span>
                          </div>
                        </td>
                        <td>{fmtPct(Number(r.penury_rate ?? NaN) * 100, 1)}</td>
                        <td>{fmtPct(Number(r.saturation_rate ?? NaN) * 100, 1)}</td>
                        <td>{fmtPct(Number(r.occ_mean ?? NaN) * 100, 1)}</td>
                        <td>{fmtPct(Number(r.tension_index ?? NaN) * 100, 1)}</td>
                        <td>{fmtInt(r.n_obs)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div className="empty">—</div>}
          </div>
        </section>
      </main>
    </div>
  );
}
