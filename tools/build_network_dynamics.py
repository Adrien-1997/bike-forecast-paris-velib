# tools/build_network_dynamics.py
from __future__ import annotations

import argparse, sys, math, json
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import folium
    from folium.plugins import TimestampedGeoJson
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False

# --------------------------- Paths & constants ---------------------------

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
FIGS_DIR = ASSETS / "figs" / "network" / "dynamics"
TABLES_DIR = ASSETS / "tables" / "network" / "dynamics"
MAPS_DIR = ASSETS / "maps"

PROFILE_LONG_DAYS = 90
EPISODE_MIN_STEPS = 4    # 4×15min = 1h
FRAME_STEP_MIN = 60

# --------------------------- Utils ---------------------------

def _mkdirs() -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)

def _read_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"[dynamics] Introuvable: {path}")
    df = pd.read_parquet(path)

    # ts
    for tc in ("ts", "tbin_utc"):
        if tc in df.columns:
            df["ts"] = pd.to_datetime(df[tc], errors="coerce")
            break
    if "ts" not in df.columns:
        raise KeyError("[dynamics] Colonne temporelle manquante (ts/tbin_utc)")
    df["ts"] = df["ts"].dt.floor("15min")

    # station_id
    sid = None
    for c in ("station_id", "stationcode", "stationCode", "id"):
        if c in df.columns:
            sid = c
            break
    if sid is None:
        raise KeyError("[dynamics] Identifiant station manquant (station_id/stationcode)")
    df["station_id"] = df[sid].astype(str)

    # bikes
    bikes_col = None
    for c in ("bikes", "nb_velos_bin", "velos_disponibles", "numBikesAvailable", "n_bikes"):
        if c in df.columns:
            bikes_col = c
            break
    if bikes_col is None:
        raise KeyError("[dynamics] Colonne vélos manquante (bikes/nb_velos_bin/velos_disponibles)")
    df["bikes"] = pd.to_numeric(df[bikes_col], errors="coerce")

    # docks + capacity (optionnels)
    docks_col = None
    for c in ("docks", "docks_disponibles", "numDocksAvailable", "places_disponibles"):
        if c in df.columns:
            docks_col = c
            break
    df["docks_avail"] = pd.to_numeric(df.get(docks_col, np.nan), errors="coerce")

    cap_col = None
    for c in ("capacity", "cap", "dock_count", "n_docks_total"):
        if c in df.columns:
            cap_col = c
            break
    df["capacity_src"] = pd.to_numeric(df.get(cap_col, np.nan), errors="coerce")

    # meta
    name_col = None
    for c in ("name", "station_name", "label"):
        if c in df.columns:
            name_col = c
            break
    df["name"] = df.get(name_col, df["station_id"]).astype(str)
    df["lat"]  = pd.to_numeric(df.get("lat", np.nan), errors="coerce")
    df["lon"]  = pd.to_numeric(df.get("lon", np.nan), errors="coerce")

    return df[["ts", "station_id", "name", "lat", "lon", "bikes", "docks_avail", "capacity_src"]].copy()

def _to_local_dt(s: pd.Series, tz: Optional[str]) -> pd.Series:
    return s.dt.tz_localize("UTC").dt.tz_convert(tz) if tz else s

def _save_fig(path: Path) -> None:
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()

def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _estimate_capacity(win: pd.DataFrame) -> pd.Series:
    """Capacité estimée par station (priorité: capacity_src ; sinon q98(bikes+docks) ; sinon q98(bikes))."""
    def est(g: pd.DataFrame) -> float:
        cap_src = pd.to_numeric(g["capacity_src"], errors="coerce")
        cap = cap_src.max()  # peut être NaN/pd.NA
        if pd.notna(cap) and float(cap) > 0:
            return float(cap)
        if g["docks_avail"].notna().any():
            s = (g["bikes"].clip(lower=0) + g["docks_avail"].clip(lower=0)).dropna()
            if len(s):
                return float(s.quantile(0.98))
        b = g["bikes"].clip(lower=0).dropna()
        return float(b.quantile(0.98)) if len(b) else np.nan

    # important: appliquer sur colonnes utiles (pas de colonne de groupe) → pas de FutureWarning
    cols = ["capacity_src", "docks_avail", "bikes"]
    return win.groupby("station_id")[cols].apply(est).rename("capacity_est")

# --------------------------- 1) Heatmaps h×j réseau ---------------------------

def _heatmap_network(df: pd.DataFrame, tz: Optional[str]) -> None:
    if df.empty:
        return

    ldt = _to_local_dt(df["ts"], tz)
    df = df.assign(
        date_local=ldt.dt.date,
        dow=ldt.dt.dayofweek,  # 0=lundi
        hour=ldt.dt.hour
    )

    # Capacité estimée
    cap = _estimate_capacity(df)
    df = df.merge(cap.rename("cap_est"), left_on="station_id", right_index=True, how="left")

    # Occ 0..1 (fallback si cap manquante)
    def _occ_row(x):
        c = x.get("cap_est", np.nan)
        if pd.notna(c) and float(c) > 0:
            return float(np.clip((x["bikes"] if pd.notna(x["bikes"]) else 0.0) / float(c), 0.0, 1.0))
        b = df.loc[df["station_id"] == x["station_id"], "bikes"].clip(lower=0)
        q = b.quantile(0.98) if len(b) else np.nan
        return float(np.clip((x["bikes"] if pd.notna(x["bikes"]) else 0.0) / q, 0.0, 1.0)) if pd.notna(q) and q > 0 else 0.0

    df["occ"] = df.apply(_occ_row, axis=1)

    agg = df.groupby(["dow", "hour"]).agg(
        bikes_mean=("bikes", "mean"),
        occ_mean=("occ", "mean"),
        penury_rate=("bikes", lambda s: float((pd.to_numeric(s, errors="coerce").fillna(0.0) == 0.0).mean())),
        saturation_rate=("docks_avail", lambda s: float((pd.to_numeric(s, errors="coerce").fillna(0.0) == 0.0).mean())),
        n_obs=("bikes", "count")
    ).reset_index()

    # Grille complète 7×24
    full_idx = pd.Index(range(7), name="dow")
    full_cols = pd.Index(range(24), name="hour")

    # 1) Occupation moyenne
    pivot_occ = (agg.pivot(index="dow", columns="hour", values="occ_mean")
                   .reindex(index=full_idx, columns=full_cols))
    mat_occ = np.ma.masked_invalid(pivot_occ.to_numpy(dtype=float))

    plt.figure(figsize=(10, 4))
    plt.imshow(mat_occ, aspect="auto")
    plt.title("Occupation moyenne (0..1) — par jour (lignes) × heure (colonnes)")
    plt.xlabel("Heure")
    plt.ylabel("Jour (0=lundi)")
    plt.colorbar()
    _save_fig(FIGS_DIR / "heatmap_occ.png")

    # 2) Indice de tension = pénurie + saturation
    # on pivote chaque métrique puis on additionne, en reindexant sur la grille complète
    pivot_pen = (agg.pivot(index="dow", columns="hour", values="penury_rate")
                   .reindex(index=full_idx, columns=full_cols))
    pivot_sat = (agg.pivot(index="dow", columns="hour", values="saturation_rate")
                   .reindex(index=full_idx, columns=full_cols))
    pivot_tension = pivot_pen.add(pivot_sat, fill_value=np.nan)
    mat_tension = np.ma.masked_invalid(pivot_tension.to_numpy(dtype=float))

    plt.figure(figsize=(10, 4))
    plt.imshow(mat_tension, aspect="auto")
    plt.title("Indice de tension (pénurie + saturation) — par jour × heure")
    plt.xlabel("Heure")
    plt.ylabel("Jour (0=lundi)")
    plt.colorbar()
    _save_fig(FIGS_DIR / "heatmap_tension.png")

# --------------------------- 2) Indice de tension par station ---------------------------

def _tension_index_by_station(df: pd.DataFrame, last_days: int) -> pd.DataFrame:
    if last_days <= 0 or df.empty:
        return pd.DataFrame()
    tmax = df["ts"].max()
    tmin = tmax - pd.Timedelta(days=last_days)
    win = df[(df["ts"] >= tmin) & (df["ts"] <= tmax)].copy()
    cap = _estimate_capacity(win)
    win = win.merge(cap.rename("cap_est"), left_on="station_id", right_index=True, how="left")

    def _occ_row(x):
        c = x.get("cap_est", np.nan)
        if pd.notna(c) and float(c) > 0:
            return float(np.clip((x["bikes"] if pd.notna(x["bikes"]) else 0.0) / float(c), 0.0, 1.0))
        return np.nan

    win["occ"] = win.apply(_occ_row, axis=1)

    res = win.groupby("station_id").agg(
        penury_rate=("bikes", lambda s: float((pd.to_numeric(s, errors="coerce").fillna(0.0) == 0.0).mean())),
        saturation_rate=("docks_avail", lambda s: float((pd.to_numeric(s, errors="coerce").fillna(0.0) == 0.0).mean())),
        occ_mean=("occ", "mean"),
        name=("name", "last"),
        lat=("lat", "last"),
        lon=("lon", "last"),
        n_obs=("bikes", "count")
    ).reset_index()
    res["tension_index"] = res["penury_rate"] + res["saturation_rate"]
    res.sort_values("tension_index", ascending=False, inplace=True)
    return res

# --------------------------- 3) Régularité "aujourd’hui" ---------------------------

def _regularity_today(df: pd.DataFrame, tz: Optional[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    ldt = _to_local_dt(df["ts"], tz)
    df = df.assign(
        date_local=ldt.dt.date,
        dow=ldt.dt.dayofweek,
        hhmm=ldt.dt.strftime("%H:%M")
    ).copy()

    last_local_day = df["date_local"].iloc[df["ts"].idxmax()]
    today = df[df["date_local"] == last_local_day].copy()
    if today.empty:
        return pd.DataFrame()

    target_dow = int(today["dow"].iloc[0])
    df_long = df[(df["date_local"] < last_local_day) & (df["dow"] == target_dow)].copy()

    def curve_occ(dfg: pd.DataFrame) -> pd.Series:
        cap_s = _estimate_capacity(dfg).reindex(dfg["station_id"].unique())
        cap = cap_s.iloc[0] if len(cap_s) else np.nan
        if pd.notna(cap) and float(cap) > 0:
            occ = (dfg["bikes"].clip(lower=0) / float(cap)).clip(0, 1)
        else:
            q98 = dfg["bikes"].clip(lower=0).quantile(0.98) if len(dfg) else np.nan
            occ = (dfg["bikes"].clip(lower=0) / q98).clip(0, 1) if q98 and q98 > 0 else pd.Series(0.0, index=dfg.index)
        g = pd.DataFrame({"hhmm": dfg["hhmm"].values, "occ": occ.values})
        return g.groupby("hhmm")["occ"].mean()

    cur_today = today.groupby("station_id").apply(curve_occ)
    ref_curves = {}
    for sid, sub in df_long.groupby("station_id"):
        days = sorted(sub["date_local"].unique())
        daily_curves = []
        for d in days[-PROFILE_LONG_DAYS:]:
            c = curve_occ(sub[sub["date_local"] == d])
            daily_curves.append(c)
        if daily_curves:
            ref_curves[sid] = pd.concat(daily_curves, axis=1).median(axis=1)

    rows = []
    for sid, today_curve in cur_today.items():
        ref = ref_curves.get(sid)
        if ref is None or today_curve.empty:
            corr = np.nan
        else:
            idx = today_curve.index.intersection(ref.index)
            if len(idx) >= 8:
                a = today_curve.reindex(idx).astype(float).values
                b = ref.reindex(idx).astype(float).values
                corr = np.nan if (np.std(a) == 0 or np.std(b) == 0) else float(np.corrcoef(a, b)[0, 1])
            else:
                corr = np.nan
        rows.append({"station_id": sid, "regularity_corr_today_vs_typical": corr})
    return pd.DataFrame(rows)

# --------------------------- 4) Détection d’épisodes ---------------------------

def _episodes(df: pd.DataFrame, last_days: int, min_steps: int = EPISODE_MIN_STEPS) -> pd.DataFrame:
    if last_days <= 0 or df.empty:
        return pd.DataFrame()
    tmax = df["ts"].max()
    tmin = tmax - pd.Timedelta(days=last_days)
    win = df[(df["ts"] >= tmin) & (df["ts"] <= tmax)].copy()

    win["is_penury"] = pd.to_numeric(win["bikes"], errors="coerce").fillna(0.0) == 0.0
    win["is_saturation"] = pd.to_numeric(win["docks_avail"], errors="coerce").fillna(0.0) == 0.0

    def _detect_runs(s: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
        runs = []
        start = None
        prev_ts = None
        for ts, v in s.items():
            if v:
                if start is None:
                    start = ts
                prev_ts = ts
            else:
                if start is not None:
                    runs.append((start, prev_ts, int((prev_ts - start).total_seconds() / 60 / 15) + 1))
                    start, prev_ts = None, None
        if start is not None:
            runs.append((start, prev_ts, int((prev_ts - start).total_seconds() / 60 / 15) + 1))
        return [r for r in runs if r[2] >= min_steps]

    rows = []
    for sid, sub in win.sort_values("ts").groupby("station_id"):
        penury_runs = _detect_runs(pd.Series(sub["is_penury"].values, index=sub["ts"].values))
        sat_runs = _detect_runs(pd.Series(sub["is_saturation"].values, index=sub["ts"].values))
        for a, b, k in penury_runs:
            rows.append({"station_id": sid, "type": "penury", "start": a, "end": b, "steps": k})
        for a, b, k in sat_runs:
            rows.append({"station_id": sid, "type": "saturation", "start": a, "end": b, "steps": k})
    return pd.DataFrame(rows).sort_values(["station_id", "start"])

# --------------------------- 5) Agrégations spatiales ---------------------------

def _by_zone(df: pd.DataFrame, last_days: int) -> pd.DataFrame:
    if last_days <= 0 or df.empty:
        return pd.DataFrame()

    def zone_label(lat, lon):
        try:
            lat = float(lat); lon = float(lon)
        except Exception:
            return "NA"
        return f"{round(lat, 3)}|{round(lon, 3)}"

    df = df.copy()
    df["zone"] = [zone_label(a, b) for a, b in zip(df["lat"], df["lon"])]

    tmax = df["ts"].max()
    tmin = tmax - pd.Timedelta(days=last_days)
    win = df[(df["ts"] >= tmin) & (df["ts"] <= tmax)].copy()

    cap = _estimate_capacity(win)
    win = win.merge(cap.rename("cap_est"), left_on="station_id", right_index=True, how="left")
    win["occ"] = np.where((pd.notna(win["cap_est"]) & (win["cap_est"] > 0)),
                          np.clip(win["bikes"].clip(lower=0) / win["cap_est"], 0.0, 1.0),
                          np.nan)

    agg = win.groupby("zone").agg(
        occ_mean=("occ", "mean"),
        penury_rate=("bikes", lambda s: float((pd.to_numeric(s, errors="coerce").fillna(0.0) == 0.0).mean())),
        saturation_rate=("docks_avail", lambda s: float((pd.to_numeric(s, errors="coerce").fillna(0.0) == 0.0).mean())),
        cap_sum=("cap_est", "sum"),
        n_obs=("bikes", "count")
    ).reset_index()
    return agg.sort_values("occ_mean", ascending=True)

# --------------------------- 6) Carte temporelle (animation last 24h) ---------------------------

def _temporal_map_last_day(df: pd.DataFrame, tz: Optional[str]) -> None:
    if not HAS_FOLIUM or df.empty:
        return
    ldt = _to_local_dt(df["ts"], tz)
    df = df.assign(date_local=ldt.dt.date, hour=ldt.dt.hour)
    last_day = df["date_local"].iloc[df["ts"].idxmax()]
    sub = df[df["date_local"] == last_day].copy()
    if sub.empty:
        return

    frames = []
    for (_, h), g in sub.groupby(["date_local", "hour"]):
        g = g.sort_values("ts")
        frames.append(g.iloc[len(g)//2:len(g)//2+1])
    frames = pd.concat(frames, ignore_index=True)

    features = []
    for _, r in frames.iterrows():
        lat, lon = r.get("lat"), r.get("lon")
        if pd.isna(lat) or pd.isna(lon):
            continue
        bikes = r.get("bikes", np.nan)
        docks = r.get("docks_avail", np.nan)
        if (not pd.isna(bikes)) and float(bikes) == 0.0:
            color = "red"
        elif (not pd.isna(docks)) and float(docks) == 0.0:
            color = "black"
        else:
            color = "green"
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
            "properties": {
                "time": pd.to_datetime(r["ts"]).isoformat(),
                "style": {"color": color, "fillColor": color, "opacity": 0.8},
                "icon": "circle",
                "popup": f"{r.get('name', r['station_id'])} — bikes={'' if pd.isna(bikes) else int(bikes)}",
            },
        })

    if not features:
        print("[dynamics] Pas de features pour l’animation.")
    else:
        gj = {"type": "FeatureCollection", "features": features}
        m = folium.Map(location=[float(frames["lat"].median()), float(frames["lon"].median())], zoom_start=12)
        TimestampedGeoJson(gj, period="PT1H", add_last_point=True, duration="PT10M").add_to(m)
        MAPS_DIR.mkdir(parents=True, exist_ok=True)
        m.save(str(MAPS_DIR / "network_lastday.html"))

# --------------------------- Orchestrateur ---------------------------

def run(events_path: Path, last_days: int, tz: Optional[str]) -> None:
    _mkdirs()
    df = _read_events(events_path)
    print(f"[dynamics] events: {len(df):,} rows, stations={df['station_id'].nunique()}  "
          f"span=({df['ts'].min()} → {df['ts'].max()})")

    _heatmap_network(df, tz=tz)

    ti = _tension_index_by_station(df, last_days=last_days)
    ti.to_csv(TABLES_DIR / "tension_by_station.csv", index=False)
    print(f"[dynamics] tension_by_station: {len(ti)} rows → {TABLES_DIR/'tension_by_station.csv'}")

    reg = _regularity_today(df, tz=tz)
    reg.to_csv(TABLES_DIR / "regularity_today.csv", index=False)
    print(f"[dynamics] regularity_today: {len(reg)} rows → {TABLES_DIR/'regularity_today.csv'}")

    epi = _episodes(df, last_days=last_days)
    epi.to_csv(TABLES_DIR / "episodes.csv", index=False)
    print(f"[dynamics] episodes: {len(epi)} rows → {TABLES_DIR/'episodes.csv'}")

    _by = _by_zone(df, last_days=last_days)
    _by.to_csv(TABLES_DIR / "by_zone.csv", index=False)
    print(f"[dynamics] by_zone: {len(_by)} rows → {TABLES_DIR/'by_zone.csv'}")

    _temporal_map_last_day(df, tz=tz)
    if HAS_FOLIUM:
        print(f"[dynamics] map → {MAPS_DIR/'network_lastday.html'}")

def main(events_path: str, last_days: int, tz: Optional[str]) -> None:
    run(Path(events_path), last_days=last_days, tz=tz)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=str, required=True, help="Parquet des évènements (docs/exports/events.parquet)")
    parser.add_argument("--last-days", type=int, default=7, help="Fenêtre récente pour certains indicateurs")
    parser.add_argument("--tz", type=str, default=None, help="Timezone ex: Europe/Paris")
    args = parser.parse_args()
    sys.exit(main(events_path=args.events, last_days=args.last_days, tz=args.tz))
