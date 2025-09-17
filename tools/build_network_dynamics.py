# tools/build_network_dynamics.py
from __future__ import annotations

import argparse, sys, math, json
from pathlib import Path
from typing import Optional, List, Tuple
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import folium
    from folium.plugins import TimestampedGeoJson
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False

import os, locale

# Encodage de sortie : prend PYTHONIOENCODING si présent, sinon encodage Windows (CP-1252)
ENC = os.environ.get("PYTHONIOENCODING") or locale.getpreferredencoding(False) or "cp1252"

# Remplacements de caractères non sûrs pour CP-1252 / consoles Windows
_REPL = {
    "→": "->",
    "—": "-",  "–": "-",
    "’": "'",  "“": '"', "”": '"',
    "≥": ">=", "≤": "<=",
    "…": "...",
    "×": "x",   # optionnel
    " ": " ",   # NBSP -> espace
}
def normalize_text(s: str) -> str:
    for a, b in _REPL.items():
        s = s.replace(a, b)
    return s

# --------------------------- Paths & constants ---------------------------

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
FIGS_DIR = ASSETS / "figs" / "network" / "dynamics"
TABLES_DIR = ASSETS / "tables" / "network" / "dynamics"
MAPS_DIR = ASSETS / "maps"

PROFILE_LONG_DAYS = 90        # fenêtre pour la journée-type
EPISODE_MIN_STEPS = 4         # séquence mini (4 pas = 1h) pour déclarer un épisode
FRAME_STEP_MIN = 60           # pas de temps (minutes) pour l’animation (1 point par heure)

# --- Markdown output (page 1) ---
OUT_MD = DOCS / "network" / "dynamics.md"
OUT_MD.parent.mkdir(parents=True, exist_ok=True)


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

def _placeholder_fig(path: Path, message: str) -> None:
    plt.figure(figsize=(8, 2.4))
    plt.axis("off")
    plt.text(0.5, 0.5, message, ha="center", va="center", fontsize=11)
    _save_fig(path)

def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def rel_from_md(md_path: Path, target: Path) -> str:
    """Chemin relatif (MkDocs, use_directory_urls:true) depuis md_path vers target."""
    md_rel = md_path.resolve().relative_to(DOCS.resolve())
    parts = md_rel.with_suffix("").parts
    depth = len(parts) if parts[-1] != "index" else len(parts) - 1
    prefix = "../" * max(depth, 0)
    rel_from_docs = target.resolve().relative_to(DOCS.resolve()).as_posix()
    return (prefix + rel_from_docs).replace("//", "/")

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

    cols = ["capacity_src", "docks_avail", "bikes"]  # évite FutureWarning
    return win.groupby("station_id")[cols].apply(est).rename("capacity_est")


def _attach_cap_est(frame: pd.DataFrame, cap_s: pd.Series) -> pd.DataFrame:
    """Ajoute la colonne 'cap_est' en joignant une Series indexée par station_id."""
    cap_df = cap_s.rename("cap_est").to_frame()
    out = frame.merge(cap_df, left_on="station_id", right_index=True, how="left")
    if "cap_est" not in out.columns:
        out["cap_est"] = np.nan
    return out


# --------------------------- 1) Heatmaps h×j réseau ---------------------------

def _heatmap_network(df: pd.DataFrame, tz: Optional[str]) -> pd.DataFrame:
    """Heatmaps + agrégés h×j. Retourne aussi le pivot occ/pen/sat pour réutilisation."""
    if df.empty:
        return pd.DataFrame()

    ldt = _to_local_dt(df["ts"], tz)
    df = df.assign(
        date_local=ldt.dt.date,
        dow=ldt.dt.dayofweek,  # 0=lundi
        hour=ldt.dt.hour
    )

    # Capacité estimée
    cap = _estimate_capacity(df)
    df = _attach_cap_est(df, cap)

    # Occ 0..1
    def _occ_row(x):
        c = x.get("cap_est", np.nan)
        if pd.notna(c) and float(c) > 0:
            return float(np.clip((x["bikes"] if pd.notna(x["bikes"]) else 0.0) / float(c), 0.0, 1.0))
        b = df.loc[df["station_id"] == x["station_id"], "bikes"].clip(lower=0)
        q = b.quantile(0.98) if len(b) else np.nan
        return float(np.clip((x["bikes"] if pd.notna(x["bikes"]) else 0.0) / q, 0.0, 1.0)) if pd.notna(q) and q > 0 else 0.0

    df["occ"] = df.apply(_occ_row, axis=1)

    agg = df.groupby(["dow", "hour"]).agg(
        occ_mean=("occ", "mean"),
        penury_rate=("bikes", lambda s: float((pd.to_numeric(s, errors="coerce").fillna(0.0) == 0.0).mean())),
        saturation_rate=("docks_avail", lambda s: float((pd.to_numeric(s, errors="coerce").fillna(0.0) == 0.0).mean())),
        n_obs=("bikes", "count")
    ).reset_index()

    # Grille complète 7×24
    full_idx = pd.Index(range(7), name="dow")
    full_cols = pd.Index(range(24), name="hour")

    pivot_occ = (agg.pivot(index="dow", columns="hour", values="occ_mean")
                   .reindex(index=full_idx, columns=full_cols))
    pivot_pen = (agg.pivot(index="dow", columns="hour", values="penury_rate")
                   .reindex(index=full_idx, columns=full_cols))
    pivot_sat = (agg.pivot(index="dow", columns="hour", values="saturation_rate")
                   .reindex(index=full_idx, columns=full_cols))
    pivot_tension = pivot_pen.add(pivot_sat, fill_value=np.nan)

    # Heatmap occ
    mat_occ = np.ma.masked_invalid(pivot_occ.to_numpy(dtype=float))
    plt.figure(figsize=(10, 4))
    plt.imshow(mat_occ, aspect="auto")
    plt.title("Occupation moyenne (0..1) — par jour (lignes) × heure (colonnes)")
    plt.xlabel("Heure"); plt.ylabel("Jour (0=lundi)")
    plt.colorbar()
    _save_fig(FIGS_DIR / "heatmap_occ.png")

    # Heatmap tension
    mat_tension = np.ma.masked_invalid(pivot_tension.to_numpy(dtype=float))
    plt.figure(figsize=(10, 4))
    plt.imshow(mat_tension, aspect="auto")
    plt.title("Indice de tension (pénurie + saturation) — par jour × heure")
    plt.xlabel("Heure"); plt.ylabel("Jour (0=lundi)")
    plt.colorbar()
    _save_fig(FIGS_DIR / "heatmap_tension.png")

    # Retourne ce qu'il faut pour d'autres vues
    pivot_pen.name = "penury_rate"
    pivot_sat.name = "saturation_rate"
    return pd.concat(
        {"occ": pivot_occ, "pen": pivot_pen, "sat": pivot_sat},
        axis=1
    )

# --------------------------- 1bis) Profils & autres visuels ---------------------------

def _extra_network_views(df: pd.DataFrame, tz: Optional[str], last_days: int) -> None:
    if df.empty:
        _placeholder_fig(FIGS_DIR / "profile_occ_by_dow.png", "Pas de données")
        _placeholder_fig(FIGS_DIR / "hourly_pen_sat.png", "Pas de données")
        _placeholder_fig(FIGS_DIR / "episodes_hist.png", "Pas d'épisodes sur la fenêtre")
        _placeholder_fig(FIGS_DIR / "byzone_tension_top.png", "Pas de zones disponibles")
        return

    ldt = _to_local_dt(df["ts"], tz)
    df = df.assign(
        date_local=ldt.dt.date,
        dow=ldt.dt.dayofweek,
        hour=ldt.dt.hour
    )

    # Capacité + occ
    cap = _estimate_capacity(df)
    df = _attach_cap_est(df, cap)

    def _occ_row(x):
        c = x.get("cap_est", np.nan)
        if pd.notna(c) and float(c) > 0:
            return float(np.clip((x["bikes"] if pd.notna(x["bikes"]) else 0.0) / float(c), 0.0, 1.0))
        b = df.loc[df["station_id"] == x["station_id"], "bikes"].clip(lower=0)
        q = b.quantile(0.98) if len(b) else np.nan
        return float(np.clip((x["bikes"] if pd.notna(x["bikes"]) else 0.0) / q, 0.0, 1.0)) if pd.notna(q) and q > 0 else 0.0
    df["occ"] = df.apply(_occ_row, axis=1)

    # ---- Profils intra-semaine : une ligne par jour (occ moyenne par heure)
    prof = df.groupby(["dow", "hour"])["occ"].mean().unstack("hour").reindex(index=range(7), columns=range(24))
    plt.figure(figsize=(10, 4))
    for d in range(7):
        y = prof.loc[d].to_numpy(dtype=float)
        plt.plot(range(24), y, label=str(d))
    plt.title("Profils intra-semaine — occupation moyenne par heure")
    plt.xlabel("Heure"); plt.ylabel("Occupation (0..1)")
    plt.legend(title="Jour (0=lundi)", ncol=4, fontsize=8)
    _save_fig(FIGS_DIR / "profile_occ_by_dow.png")

    # ---- Pénurie / saturation par heure (moyenne globale)
    hourly = df.groupby("hour").agg(
        penury_rate=("bikes", lambda s: float((pd.to_numeric(s, errors="coerce").fillna(0.0) == 0.0).mean())),
        saturation_rate=("docks_avail", lambda s: float((pd.to_numeric(s, errors="coerce").fillna(0.0) == 0.0).mean())),
    ).reindex(range(24)).reset_index()
    plt.figure(figsize=(10, 4))
    plt.plot(hourly["hour"], hourly["penury_rate"], label="Pénurie")
    plt.plot(hourly["hour"], hourly["saturation_rate"], label="Saturation")
    plt.title("Pénurie & saturation — profils horaires (moyenne réseau)")
    plt.xlabel("Heure"); plt.ylabel("Taux"); plt.legend()
    _save_fig(FIGS_DIR / "hourly_pen_sat.png")

    # ---- Histogramme des épisodes (7 j par défaut)
    epi = _episodes(df, last_days=last_days)
    if len(epi):
        epi = epi.copy()
        epi["duration_min"] = epi["steps"].astype(int) * 15
        bins = list(range(60, 16*60 + 1, 60))  # 1h -> 16h, pas 1h

        # Ne tracer que s'il y a au moins une série non vide
        has_any = False
        plt.figure(figsize=(10, 3.8))
        for typ, label in (("penury", "Pénurie"), ("saturation", "Saturation")):
            arr = epi.loc[epi["type"] == typ, "duration_min"].to_numpy()
            if arr.size > 0:
                has_any = True
                plt.hist(arr, bins=bins, alpha=0.5, label=label, density=True)
        if has_any:
            plt.title(f"Distribution des épisodes (fenêtre {last_days} j) — durée en minutes")
            plt.xlabel("Durée (min)"); plt.ylabel("Densité"); plt.legend()
            _save_fig(FIGS_DIR / "episodes_hist.png")
        else:
            plt.close()
            _placeholder_fig(FIGS_DIR / "episodes_hist.png", f"Aucun épisode détecté ({last_days} j)")
    else:
        _placeholder_fig(FIGS_DIR / "episodes_hist.png", f"Aucun épisode détecté ({last_days} j)")

    # ---- Top zones les plus tendues (pénurie + saturation)
    bz = _by_zone(df, last_days=last_days)
    if len(bz):
        bz = bz.copy()
        bz["tension"] = bz["penury_rate"] + bz["saturation_rate"]
        sel = bz.sort_values("tension", ascending=False).head(15)
        plt.figure(figsize=(10, max(3, 0.38 * len(sel) + 1)))
        plt.barh(sel["zone"].astype(str), sel["tension"])
        plt.gca().invert_yaxis()
        plt.title(f"Zones les plus tendues (pénurie + saturation) — {last_days} j")
        plt.xlabel("Tension"); plt.ylabel("Zone (grille ~1 km si arrondissement indisponible)")
        _save_fig(FIGS_DIR / "byzone_tension_top.png")
    else:
        _placeholder_fig(FIGS_DIR / "byzone_tension_top.png", f"Pas de zones calculées ({last_days} j)")


# --------------------------- 2) Indice de tension par station ---------------------------

def _tension_index_by_station(df: pd.DataFrame, last_days: int) -> pd.DataFrame:
    if last_days <= 0 or df.empty:
        return pd.DataFrame()
    tmax = df["ts"].max()
    tmin = tmax - pd.Timedelta(days=last_days)
    win = df[(df["ts"] >= tmin) & (df["ts"] <= tmax)].copy()
    cap = _estimate_capacity(win)
    win = _attach_cap_est(win, cap)

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

    last_local_day = df.loc[df["ts"].idxmax(), "date_local"]
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
                if np.std(a) == 0 or np.std(b) == 0:
                    corr = np.nan
                else:
                    corr = float(np.corrcoef(a, b)[0, 1])
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
    for sid, sub in win.groupby("station_id"):
        sub = sub.sort_values("ts")
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

    # zone = arrondissement si dispo ; ici fallback grille ~1 km (lat/lon arrondis)
    def zone_label(lat, lon):
        try:
            lat = float(lat); lon = float(lon)
        except Exception:
            return "NA"
        return f"{round(lat, 3)}|{round(lon, 3)}"

    # Prépare la zone
    df = df.copy()
    df["zone"] = [zone_label(a, b) for a, b in zip(df.get("lat"), df.get("lon"))]

    # Fenêtre temporelle récente
    tmax = df["ts"].max()
    tmin = tmax - pd.Timedelta(days=last_days)
    win = df[(df["ts"] >= tmin) & (df["ts"] <= tmax)].copy()

    # Capacité estimée → joindre proprement (assure la présence de cap_est)
    cap = _estimate_capacity(win)
    win = _attach_cap_est(win, cap)
    if "cap_est" not in win.columns:
        win["cap_est"] = np.nan

    # Occupation normalisée (0..1) quand cap dispo > 0
    cap_num = pd.to_numeric(win["cap_est"], errors="coerce")
    bikes_num = pd.to_numeric(win["bikes"], errors="coerce").clip(lower=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        occ = bikes_num / cap_num
    occ = np.where((~np.isnan(cap_num)) & (cap_num > 0), np.clip(occ, 0.0, 1.0), np.nan)
    win["occ"] = occ

    # Agrégations par zone
    agg = win.groupby("zone", dropna=False).agg(
        occ_mean=("occ", "mean"),
        penury_rate=("bikes", lambda s: float((pd.to_numeric(s, errors="coerce").fillna(0.0) == 0.0).mean())),
        saturation_rate=("docks_avail", lambda s: float((pd.to_numeric(s, errors="coerce").fillna(0.0) == 0.0).mean())),
        cap_sum=("cap_est", "sum"),
        n_obs=("bikes", "count"),
    ).reset_index()

    return agg.sort_values("occ_mean", ascending=True)



# --------------------------- 6) Carte temporelle (animation last 24h) ---------------------------

def _temporal_map_last_day(df: pd.DataFrame, tz: Optional[str]) -> None:
    if not HAS_FOLIUM or df.empty:
        return
    ldt = _to_local_dt(df["ts"], tz)
    df = df.assign(date_local=ldt.dt.date, hour=ldt.dt.hour)
    last_day = df.loc[df["ts"].idxmax(), "date_local"]
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


# --------------------------- Markdown template ---------------------------

MD_TEMPLATE = """# Dynamiques spatio-temporelles

## 3) Dynamiques spatio-temporelles (ce que vous trouverez dans `dynamics.md`)

### Objectif
Mettre en évidence les **rythmes** et **déplacements de pression** dans la ville.

### Analyses proposées
- **Heatmaps h×j** (par station et agrégées) : intensité des vélos disponibles ou du taux de pénurie, par heure du jour et jour de semaine.  
- **Saisonnalité courte/longue** :  
  - **Intra-semaine** (lundi→dimanche) : comparaison des profils typiques.  
  - **Intra-année** (si historique suffisant) : effet météo/vacances (qualitatif), glissements de pics.  
- **Cartes temporelles animées** (ou séquences d’instantanés) pour suivre la **vague de saturation → pénurie** sur une journée type.  
- **Flux intra-urbains (qualitatif)** : lecture conjointe des zones qui passent de saturation à pénurie avec un décalage horaire (indication de “courant” de déplacement).

### Indicateurs & méthodes
- **Indice de tension** par station : `penurie_rate + saturation_rate` ({last_days} jours).  
- **Score de régularité** : corrélation de la journée en cours à la journée type (90 derniers jours).  
- **Détection d’épisodes** : séquences ≥ X pas en pénurie/saturation (morphologie binaire).  
- **Agrégations spatiales** : par arrondissement/quartier (moyennes pondérées par capacité).

### Lecture & limites
- Les heatmaps mettent à nu la **récurrence** des phénomènes ; elles n’expliquent pas la cause (météo, événements).  
- Les “flux” sont déduits **visuellement** par co-évolution des zones ; ils ne sont pas des trajectoires individuelles.

---

## Vues générées (auto)

### Heatmaps hebdomadaires
![Occupation moyenne (0..1)]({heatmap_occ_rel})
![Indice de tension (pénurie + saturation)]({heatmap_tension_rel})

### Profils intra-semaine
![Profils d'occupation par heure]({profile_occ_rel})

### Pénurie / Saturation — profils horaires
![Taux par heure]({hourly_pen_sat_rel})

### Distribution des épisodes ({last_days} j)
![Histogrammes des épisodes]({episodes_hist_rel})

### Zones les plus tendues ({last_days} j)
![Top zones par tension]({byzone_top_rel})

### Carte temporelle (dernière journée locale)
<div style="margin: 0.5rem 0;">
  <iframe src="{map_rel}" style="width:100%;height:520px;border:0" loading="lazy" title="Carte temporelle du réseau (dernière journée)"></iframe>
</div>

### Tables d’appui
- **Tension par station** ({last_days} j) : `{tension_csv_rel}`  
- **Régularité — aujourd’hui vs journée-type** : `{regularity_csv_rel}`  
- **Épisodes pénurie/saturation** ({last_days} j) : `{episodes_csv_rel}`  
- **Agrégations par zone** ({last_days} j) : `{byzone_csv_rel}`

---

## Valeur analytique de la section “Réseau”
- **Opérationnel** : repérer rapidement les zones à surveiller (redispatch).  
- **Stratégique** : comprendre les **archétypes d’usage** et leur évolution (clustering).  
- **Communication** : visualisations pédagogiques pour le grand public (profil 24 h, cartes).

### Bonnes pratiques de lecture
- Toujours croiser **pénurie** et **saturation** (les deux faces d’un déséquilibre).  
- Un **taux de disponibilité élevé** ne signifie pas faible tension : regarder la **volatilité**.  
- Les **clusters** aident à comparer *des stations comparables* entre elles.
"""


# --------------------------- Orchestrateur ---------------------------

def run(events_path: Path, last_days: int, tz: Optional[str]) -> None:
    _mkdirs()
    df = _read_events(events_path)
    msg = (f"[dynamics] events: {len(df):,} rows, stations={df['station_id'].nunique()}  "
        f"span=({df['ts'].min()} -> {df['ts'].max()})")
    print(normalize_text(msg))


    # Heatmaps + pivots h×j
    pivots = _heatmap_network(df, tz=tz)

    # Autres visuels (profils, épisodes, zones)
    _extra_network_views(df, tz=tz, last_days=last_days)

    # Tables
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
        print(normalize_text(f"[dynamics] map -> {MAPS_DIR/'network_lastday.html'}"))


    # --- Render Markdown page (page 1) ---
    md = MD_TEMPLATE.format(
        last_days=last_days,
        heatmap_occ_rel=rel_from_md(OUT_MD, FIGS_DIR / "heatmap_occ.png"),
        heatmap_tension_rel=rel_from_md(OUT_MD, FIGS_DIR / "heatmap_tension.png"),
        profile_occ_rel=rel_from_md(OUT_MD, FIGS_DIR / "profile_occ_by_dow.png"),
        hourly_pen_sat_rel=rel_from_md(OUT_MD, FIGS_DIR / "hourly_pen_sat.png"),
        episodes_hist_rel=rel_from_md(OUT_MD, FIGS_DIR / "episodes_hist.png"),
        byzone_top_rel=rel_from_md(OUT_MD, FIGS_DIR / "byzone_tension_top.png"),
        map_rel=rel_from_md(OUT_MD, MAPS_DIR / "network_lastday.html"),
        tension_csv_rel=rel_from_md(OUT_MD, TABLES_DIR / "tension_by_station.csv"),
        regularity_csv_rel=rel_from_md(OUT_MD, TABLES_DIR / "regularity_today.csv"),
        episodes_csv_rel=rel_from_md(OUT_MD, TABLES_DIR / "episodes.csv"),
        byzone_csv_rel=rel_from_md(OUT_MD, TABLES_DIR / "by_zone.csv"),
    )
    md = normalize_text(md)  # supprime flèches “→”, tirets longs, ≥, etc.
    with open(OUT_MD, "w", encoding=ENC, newline="\n") as f:  # plus de "utf-8" forcé
        f.write(md)
    print(f"[dynamics] md  -> {OUT_MD} (encoding={ENC})")


def main(events_path: str, last_days: int, tz: Optional[str]) -> None:
    run(Path(events_path), last_days=last_days, tz=tz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=str, required=True, help="Parquet des évènements (docs/exports/events.parquet)")
    parser.add_argument("--last-days", type=int, default=7, help="Fenêtre récente pour certains indicateurs")
    parser.add_argument("--tz", type=str, default=None, help="Timezone ex: Europe/Paris")
    args = parser.parse_args()

    sys.exit(main(events_path=args.events, last_days=args.last_days, tz=args.tz))
