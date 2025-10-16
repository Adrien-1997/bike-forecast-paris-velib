// ui/pages/monitoring/network/dynamics.tsx
import Head from "next/head";
import Link from "next/link";
import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/router";
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

/* ───────────────────────── Leaflet (client only) ───────────────────────── */
const LMap = dynamic(() => import("react-leaflet").then((m) => m.MapContainer), { ssr: false }) as any;
const LTile = dynamic(() => import("react-leaflet").then((m) => m.TileLayer), { ssr: false }) as any;
const LMarker = dynamic(() => import("react-leaflet").then((m) => m.Marker), { ssr: false }) as any;
const LCircle = dynamic(() => import("react-leaflet").then((m) => m.CircleMarker), { ssr: false }) as any;
const LPopup = dynamic(() => import("react-leaflet").then((m) => m.Popup), { ssr: false }) as any;

/* ───────────────────────── Helpers HTTP & utils ───────────────────────── */
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

/* ───────────────────────── Types (API Dynamics) ───────────────────────── */
type HeatmapsProfilesDoc = {
  schema_version: string;
  generated_at: string;
  heatmap: {
    occ_mean: (number | null)[][];         // 7 x 24
    penury_rate: (number | null)[][];      // 7 x 24 (0..1)
    saturation_rate: (number | null)[][];  // 7 x 24 (0..1)
  };
  profiles_occ_by_dow: Record<string, (number | null)[]>; // "0".."6" → 24 points (0..1)
};

type HourlyDoc = {
  schema_version: string;
  generated_at: string;
  rows: Array<{ hour: number; penury_rate: number | null; saturation_rate: number | null }>;
};

type EpisodesDoc = {
  schema_version: string;
  generated_at: string;
  last_days: number;
  rows: Array<{
    station_id: string;
    type: "penury" | "saturation";
    start_utc: string;
    end_utc: string;
    steps: number;
    duration_min: number | null;
  }>;
};

type TensionByStationDoc = {
  schema_version: string;
  generated_at: string;
  last_days: number;
  rows: Array<{
    station_id: string;
    name?: string | null;
    lat?: number | null;
    lon?: number | null;
    penury_rate: number | null;
    saturation_rate: number | null;
    occ_mean: number | null;
    tension_index: number | null; // penury + saturation (0..2)
    n_obs: number;
  }>;
};

/** Unwrap payloads possibly nested under a key */
function unwrapHeatmapsProfiles(payload: any): HeatmapsProfilesDoc | null {
  if (!payload) return null;
  return (payload.heatmaps_profiles ?? payload) as HeatmapsProfilesDoc;
}
function unwrapHourly(payload: any): HourlyDoc | null {
  if (!payload) return null;
  return (payload.hourly ?? payload) as HourlyDoc;
}
function unwrapEpisodes(payload: any): EpisodesDoc | null {
  if (!payload) return null;
  return (payload.episodes ?? payload) as EpisodesDoc;
}
function unwrapTension(payload: any): TensionByStationDoc | null {
  if (!payload) return null;
  return (payload.tension_by_station ?? payload) as TensionByStationDoc;
}

/* ───────────────────────── Stations index (pour noms + coords) ───────────────────────── */
type StationMeta = { station_id: string; name?: string | null; lat?: number | null; lon?: number | null };
async function fetchStationsIndex(): Promise<Record<string, StationMeta>> {
  const arr = await getJSON<any[]>("/stations").catch(() => []);
  const idx: Record<string, StationMeta> = {};
  for (const s of arr) {
    const sid = String((s as any).station_id);
    idx[sid] = {
      station_id: sid,
      name: (s as any).name ?? null,
      lat: Number((s as any).lat ?? NaN),
      lon: Number((s as any).lon ?? NaN),
    };
    if (!Number.isFinite(idx[sid].lat!)) idx[sid].lat = null;
    if (!Number.isFinite(idx[sid].lon!)) idx[sid].lon = null;
  }
  return idx;
}

/* ───────────────────────── Page ───────────────────────── */
export default function DynamicsPage() {
  const router = useRouter();
  const qStation = typeof router.query.station_id === "string" ? router.query.station_id : "";
  const [stationId, setStationId] = useState<string>(qStation);

  const [heat, setHeat] = useState<HeatmapsProfilesDoc | null>(null);
  const [hourly, setHourly] = useState<HourlyDoc | null>(null);
  const [episodes, setEpisodes] = useState<EpisodesDoc | null>(null);
  const [tension, setTension] = useState<TensionByStationDoc | null>(null);
  const [stationsIdx, setStationsIdx] = useState<Record<string, StationMeta>>({});

  const [dowSel, setDowSel] = useState<number>(1); // 1 = Lundi
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { setStationId(qStation); }, [qStation]);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setLoading(true);
        const res = await Promise.allSettled([
          getJSON<any>("/monitoring/network/dynamics/heatmaps_profiles"),
          getJSON<any>("/monitoring/network/dynamics/hourly_pen_sat"),
          getJSON<any>("/monitoring/network/dynamics/episodes"),
          getJSON<any>("/monitoring/network/dynamics/tension_by_station"),
        ]);
        if (!alive) return;

        setHeat(unwrapHeatmapsProfiles(ok(res[0])));
        setHourly(unwrapHourly(ok(res[1])));
        setEpisodes(unwrapEpisodes(ok(res[2])));
        setTension(unwrapTension(ok(res[3])));

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

  /* ───────────────────────── Heatmaps 7×24 ───────────────────────── */
  const heatmap = (title: string, matrix: (number | null)[][], isPct01 = false): JSX.Element => {
    const z = matrix?.map((row) => row?.map((v) => (Number.isFinite(Number(v)) ? (isPct01 ? Number(v) * 100 : Number(v)) : null))) ?? [];
    const y = ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"];
    const x = [...Array(24)].map((_,i)=>`${String(i).padStart(2,"0")}:00`);
    return (
      <Plot
        data={[
          {
            z, x, y,
            type: "heatmap",
            hoverongaps: false,
            colorbar: { title: isPct01 ? "%" : "Occ" },
          } as Partial<Plotly.PlotData> as any
        ]}
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
        style={{ width: "100%" }}
      />
    );
  };

  /* ───────────────────────── Profils par jour ───────────────────────── */
  const selectedProfile = useMemo(() => {
    const key = String(dowSel ?? 1);
    const arr = heat?.profiles_occ_by_dow?.[key] ?? [];
    return arr.map((v) => (Number.isFinite(Number(v)) ? Number(v) * 100 : null));
  }, [heat, dowSel]);

  // ▼ Y-range dynamique pour le profil
  const profileYMax = useMemo(() => {
    const vals = (selectedProfile || []).filter((v) => Number.isFinite(Number(v))) as number[];
    const maxv = vals.length ? Math.max(...vals) : 100;
    return clamp(Math.ceil(maxv + 5), 20, 100);
  }, [selectedProfile]);

  /* ───────────────────────── Barres horaires ───────────────────────── */
  const hourlyBars: Partial<Plotly.PlotData>[] = useMemo(() => {
    const rows = hourly?.rows ?? [];
    const hours = [...Array(24)].map((_,i)=>i);
    return [
      {
        x: hours,
        y: hours.map(h => {
          const r = rows.find(q => q.hour === h);
          return clamp01(r?.penury_rate) != null ? Number(r!.penury_rate) * 100 : null;
        }),
        type: "bar",
        name: "Pénurie (%)",
      },
      {
        x: hours,
        y: hours.map(h => {
          const r = rows.find(q => q.hour === h);
          return clamp01(r?.saturation_rate) != null ? Number(r!.saturation_rate) * 100 : null;
        }),
        type: "bar",
        name: "Saturation (%)",
      },
    ];
  }, [hourly]);

  // ▼ Y-range dynamique pour les barres
  const hourlyYMax = useMemo(() => {
    const vals: number[] = [];
    for (const s of hourlyBars) {
      for (const v of ((s.y as (number|null)[]) ?? [])) {
        if (Number.isFinite(Number(v))) vals.push(Number(v));
      }
    }
    const maxv = vals.length ? Math.max(...vals) : 100;
    return clamp(Math.ceil(maxv + 5), 10, 100);
  }, [hourlyBars]);

  /* ───────────────────────── Episodes (filtre station) ───────────────────────── */
  const episodesFiltered = useMemo(() => {
    const rows = episodes?.rows ?? [];
    const sid = (stationId || "").trim();
    return sid ? rows.filter(r => r.station_id === sid) : rows;
  }, [episodes, stationId]);

  /* ───────────────────────── Tension par station (recherche) ───────────────────────── */
  const [search, setSearch] = useState("");
  const tensionRows = useMemo(() => {
    const rows = tension?.rows ?? [];
    const q = (search || "").toLowerCase();
    const filtered = q
      ? rows.filter(r => r.station_id.toLowerCase().includes(q) || (r.name ?? "").toLowerCase().includes(q))
      : rows;
    return filtered
      .map(r => ({
        ...r,
        _name: stationsIdx[r.station_id]?.name ?? r.name ?? r.station_id,
      }))
      .sort((a,b) => Number(b.tension_index ?? 0) - Number(a.tension_index ?? 0))
      .slice(0, 200);
  }, [tension, search, stationsIdx]);

  /* ───────────────────────── Helpers cartes ───────────────────────── */
  const PARIS = { lat: 48.8566, lon: 2.3522 };

  /* ───────────────────────── Carte des épisodes ───────────────────────── */
  function EpisodesMap() {
    const pts = useMemo(() => {
      const rows = episodesFiltered ?? [];
      const out: Array<{ lat: number; lon: number; name: string; type: "penury"|"saturation"; start: string; end: string; dur: number|null; sid: string }> = [];
      for (const r of rows) {
        const meta = stationsIdx[r.station_id];
        if (meta?.lat != null && meta?.lon != null) {
          out.push({
            lat: meta.lat!, lon: meta.lon!,
            name: meta.name ?? r.station_id,
            type: r.type,
            start: r.start_utc, end: r.end_utc, dur: r.duration_min ?? null,
            sid: r.station_id,
          });
        }
      }
      return out;
    }, [episodesFiltered, stationsIdx]);

    return (
      <div style={{ width: "100%", height: 360, borderRadius: 10, overflow: "hidden" }}>
        <LMap center={[PARIS.lat, PARIS.lon]} zoom={12} style={{ width: "100%", height: "100%" }}>
          <LTile attribution='&copy; OpenStreetMap' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
          {pts.map((p, i) => (
            <LCircle
              key={`${p.sid}-${i}`}
              center={[p.lat, p.lon]}
              radius={6}
              pathOptions={{ color: p.type === "penury" ? "#ef4444" : "#3b82f6", fillOpacity: 0.7 }}
            >
              <LPopup>
                <div style={{ fontSize: 12 }}>
                  <div><b>{p.name}</b> <span style={{ opacity: 0.6 }}>({p.sid})</span></div>
                  <div>Type : <b style={{ color: p.type === "penury" ? "#ef4444" : "#3b82f6" }}>{p.type}</b></div>
                  <div>Début : {new Date(p.start).toLocaleString("fr-FR")}</div>
                  <div>Fin : {new Date(p.end).toLocaleString("fr-FR")}</div>
                  <div>Durée : {fmtInt(p.dur)}</div>
                </div>
              </LPopup>
            </LCircle>
          ))}
        </LMap>
      </div>
    );
  }

  return (
    <>
      <Head>
        <title>Monitoring — Network / Dynamics</title>
        <meta name="description" content="Dynamiques réseau: heatmaps, profils, épisodes, tension par station." />
        {/* ▼ Fix scroll global (comme la page Overview) */}
        <style
          dangerouslySetInnerHTML={{
            __html: `
              html, body, #__next { height: auto !important; }
              html, body { overflow-y: auto !important; }
            `,
          }}
        />
        {/* Leaflet CSS (pour éviter de modifier _app.tsx) */}
        <link
          rel="stylesheet"
          href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
          integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
          crossOrigin=""
        />
      </Head>

      <main style={{ padding: "2rem", maxWidth: 1300, margin: "0 auto" }}>
        {/* Header */}
        <header style={{ display: "flex", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
          <div>
            <h1 style={{ margin: 0 }}>Network — Dynamics</h1>
            <div className="small" style={{ opacity: 0.7 }}>
              {generatedAt ? `Généré : ${new Date(generatedAt).toLocaleString("fr-FR")}` : "—"}
            </div>
          </div>
          <nav style={{ display: "flex", gap: 8 }}>
            <Link href="/monitoring/network/overview"
              style={{ padding: "6px 10px", borderRadius: 8, border: "1px solid #d1d5db", background: "white", color: "#111827", textDecoration: "none" }}>
              Overview
            </Link>
            <Link href="/monitoring/network/stations"
              style={{ padding: "6px 10px", borderRadius: 8, border: "1px solid #d1d5db", background: "white", color: "#111827", textDecoration: "none" }}>
              Stations
            </Link>
          </nav>
        </header>

        {loading && <Banner kind="info">Chargement…</Banner>}
        {error && <Banner kind="error">{error}</Banner>}

        {/* Heatmaps */}
        <section style={{ marginTop: 24 }}>
          <h2 style={{ margin: "12px 0" }}>Heatmaps 7×24</h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(1, minmax(0, 1fr))", gap: 16 }}>
            <Card>{heat ? heatmap("Occupation moyenne (0..1)", heat.heatmap?.occ_mean ?? [], false) : <Empty>—</Empty>}</Card>
            <Card>{heat ? heatmap("Pénurie (%)", heat.heatmap?.penury_rate ?? [], true) : <Empty>—</Empty>}</Card>
            <Card>{heat ? heatmap("Saturation (%)", heat.heatmap?.saturation_rate ?? [], true) : <Empty>—</Empty>}</Card>
          </div>
        </section>

        {/* Profils par jour de semaine */}
        <section style={{ marginTop: 24 }}>
          <h2 style={{ margin: "12px 0" }}>Profils d’occupation par jour</h2>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center", marginBottom: 8 }}>
            <label style={{ fontSize: 13, opacity: 0.8 }}>Jour :</label>
            {[1,2,3,4,5,6,0].map((d) => {
              const lbl = ["Dim","Lun","Mar","Mer","Jeu","Ven","Sam"][d]; // pour ordre visuel
              const val = d === 0 ? 0 : d; // 0..6
              return (
                <button key={d}
                  onClick={() => setDowSel(val)}
                  style={{
                    padding: "6px 10px",
                    borderRadius: 8,
                    border: "1px solid #d1d5db",
                    background: val === dowSel ? "#111827" : "white",
                    color: val === dowSel ? "white" : "#111827",
                    cursor: "pointer",
                  }}>
                  {lbl}
                </button>
              );
            })}
          </div>
          <Card>
            {selectedProfile.length ? (
              <Plot
                data={[{ x: [...Array(24)].map((_,i)=>`${String(i).padStart(2,"0")}:00`), y: selectedProfile, type: "scatter", mode: "lines", name: "Occupation (%)" } as any]}
                layout={{
                  autosize: true,
                  height: 340,
                  margin: { l: 52, r: 10, t: 20, b: 40 },
                  yaxis: { title: { text: "%" }, range: [0, profileYMax] },
                  xaxis: { title: { text: "Heure (locale jour sélectionné)" } },
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: "100%" }}
              />
            ) : <Empty>Profil indisponible.</Empty>}
          </Card>
        </section>

        {/* Barres horaires pen/sat */}
        <section style={{ marginTop: 24 }}>
          <h2 style={{ margin: "12px 0" }}>Pénurie & Saturation par heure</h2>
          <Card>
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
                style={{ width: "100%" }}
              />
            ) : <Empty>—</Empty>}
          </Card>
        </section>

        {/* Episodes */}
        <section style={{ marginTop: 24 }}>
          <h2 style={{ margin: "12px 0" }}>Épisodes (fenêtre récente)</h2>
          <Card>
            <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 10, flexWrap: "wrap" }}>
              <label style={{ fontSize: 13, opacity: 0.8 }}>Filtrer station_id :</label>
              <input
                value={stationId}
                onChange={(e)=>setStationId(e.target.value)}
                placeholder="ex: 12123"
                style={{ padding: "6px 10px", borderRadius: 8, border: "1px solid #d1d5db", outline: "none" }}
              />
              <button
                onClick={()=>router.push({ pathname: "/monitoring/network/dynamics", query: stationId ? { station_id: stationId } : {} }, undefined, { shallow: true })}
                style={{ padding: "6px 10px", borderRadius: 8, border: "1px solid #111827", background: "#111827", color: "white", cursor: "pointer" }}>
                Appliquer
              </button>
              {episodes && <span style={{ fontSize: 12, opacity: 0.7 }}>Fenêtre: {episodes.last_days} j</span>}
            </div>

            {/* ▼ Carte des épisodes */}
            <EpisodesMap />

            {episodesFiltered?.length ? (
              <div style={{ overflowX: "auto", marginTop: 12 }}>
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
            ) : <Empty>Aucun épisode.</Empty>}
          </Card>
        </section>

        {/* Tension par station */}
        <section style={{ marginTop: 24 }}>
          <h2 style={{ margin: "12px 0" }}>Tension par station</h2>
          <Card>
            <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 10 }}>
              <input
                value={search}
                onChange={(e)=>setSearch(e.target.value)}
                placeholder="Recherche station_id ou nom…"
                style={{ padding: "6px 10px", borderRadius: 8, border: "1px solid #d1d5db", outline: "none", minWidth: 280 }}
              />
              {tension && <span style={{ fontSize: 12, opacity: 0.7 }}>Fenêtre: {tension.last_days} j</span>}
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
                            <b>{r._name}</b> <span style={{ opacity: 0.6 }}>({r.station_id})</span>
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
            ) : <Empty>—</Empty>}
          </Card>
        </section>
      </main>
    </>
  );
}

/* ───────────────────────── UI atoms ───────────────────────── */
function Card({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ border: "1px solid #374151", background: "rgba(15, 23, 42, 0.5)", borderRadius: 12, padding: 12 }}>
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
    <div style={{ marginTop: 16, borderRadius: 10, padding: "10px 12px", fontSize: 13, ...style }}>{children}</div>
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
