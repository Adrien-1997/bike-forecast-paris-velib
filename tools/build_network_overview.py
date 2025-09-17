# tools/build_network_overview.py
# Génère la page "Réseau / Aperçu" avec KPIs injectés, carte Folium et figures.
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import branca
import os
import json
import math
import numpy as np
import pandas as pd
import sys
try:
    sys.stdout.reconfigure(encoding="cp1252", errors="replace")
    sys.stderr.reconfigure(encoding="cp1252", errors="replace")
except Exception:
    pass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium
from dateutil import tz

# ------------------------- Chemins & utilitaires -------------------------

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
FIGS = ASSETS / "figs" / "network" / "overview"
TABLES = ASSETS / "tables" / "network" / "overview"
MAPS = ASSETS / "maps"
OUT_MD = DOCS / "network" / "overview.md"

for d in (FIGS, TABLES, MAPS, OUT_MD.parent):
    d.mkdir(parents=True, exist_ok=True)


def rel_from_md(md_path: Path, target: Path) -> str:
    """Chemin relatif (POSIX) depuis md_path vers target, compatible MkDocs
    (use_directory_urls: true). Ex. docs/network/overview.md -> ../../assets/..."""
    md_rel = Path(md_path).resolve().relative_to(DOCS.resolve())
    parts = md_rel.with_suffix("").parts           # ('network','overview') ou ('data','index')
    depth = len(parts) if parts[-1] != "index" else len(parts) - 1
    prefix = "../" * max(depth, 0)
    rel_from_docs = Path(target).resolve().relative_to(DOCS.resolve()).as_posix()
    return (prefix + rel_from_docs).replace("//", "/")





# -------------------------- Détection de colonnes --------------------------

@dataclass
class Cols:
    ts: str
    station: str
    bikes: str
    capacity: Optional[str] = None
    docks_avail: Optional[str] = None
    name: Optional[str] = None
    lat: Optional[str] = None
    lon: Optional[str] = None

def detect_columns(df: pd.DataFrame) -> Cols:
    lower = {c.lower(): c for c in df.columns}
    def any_of(*cands):
        for c in cands:
            if c in lower: return lower[c]
        return None
    ts = any_of("ts","tbin_utc","timestamp","datetime")
    st = any_of("station_id","stationcode","id","station")
    bikes = any_of("bikes","nb_velos_bin","velos","velos_disponibles","bike_available","num_bikes_available")
    cap = any_of("capacity","cap","dock_count","num_docks_total")
    docks = any_of("docks_avail","docks_available","places_disponibles","free_docks","num_docks_available")
    name = any_of("name","station_name","nom")
    lat = any_of("lat","latitude")
    lon = any_of("lon","lng","longitude")
    if not ts or not st or not bikes:
        raise KeyError(f"[overview] Colonnes minimales absentes (trouvé: ts={ts}, station={st}, bikes={bikes})")
    return Cols(ts, st, bikes, cap, docks, name, lat, lon)

# -------------------------- Calculs temporels --------------------------

def to_local(df: pd.DataFrame, ts_col: str, tzname: str) -> pd.Series:
    s = pd.to_datetime(df[ts_col], utc=False, errors="coerce")
    # Les exports sont "UTC naïf arrondi 15 min" -> on les traite comme UTC puis convertit
    s = s.dt.tz_localize("UTC", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert(tzname)
    return s

def today_bounds_local(series_local: pd.Series) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Retourne min et max (début/fin) du jour local de la dernière observation."""
    tmax = series_local.max()
    start = tmax.normalize()
    end = start + pd.Timedelta(days=1)
    return start, end

# -------------------------- KPIs & helpers --------------------------

def part(x: pd.Series) -> float:
    if x.size == 0: return float("nan")
    return float((x.mean() * 100.0).round(2))

def safe_ratio(n: float, d: float) -> float:
    return float("nan") if d == 0 else round(100.0 * n / d, 2)

def compute_snapshot_kpis(df: pd.DataFrame, cols: Cols, tzname: str, station_universe: List[str]) -> Tuple[dict, pd.DataFrame]:
    # Snapshot = dernier timestamp
    df = df.copy()
    df["_ts_loc"] = to_local(df, cols.ts, tzname)
    last_ts = df[cols.ts].max()
    snap = df[df[cols.ts] == last_ts].copy()

    # Dériver docks_avail si absent
    docks_col = cols.docks_avail
    if docks_col is None and cols.capacity:
        docks = (pd.to_numeric(snap[cols.capacity], errors="coerce") - pd.to_numeric(snap[cols.bikes], errors="coerce")).clip(lower=0)
        snap["__docks_avail"] = docks
        docks_col = "__docks_avail"

    bikes = pd.to_numeric(snap[cols.bikes], errors="coerce")
    has_bike = bikes > 0
    has_dock = None
    sat = pen = None
    if docks_col and docks_col in snap.columns:
        docks = pd.to_numeric(snap[docks_col], errors="coerce")
        has_dock = docks > 0
        sat = docks == 0
    pen = bikes == 0

    active = snap[cols.station].astype(str).nunique()
    universe = len(station_universe)
    offline = max(universe - active, 0)

    kpis = {
        "snapshot_ts_utc": str(last_ts),
        "snapshot_ts_local": str(to_local(pd.DataFrame({cols.ts:[last_ts]}), cols.ts, tzname).iloc[0]),
        "stations_universe": universe,
        "stations_active": int(active),
        "stations_offline": int(offline),
        "availability_bike_pct": part(has_bike) if has_bike is not None else float("nan"),
        "availability_dock_pct": part(has_dock) if has_dock is not None else float("nan"),
        "penury_pct": part(pen) if pen is not None else float("nan"),
        "saturation_pct": part(sat) if sat is not None else float("nan"),
    }

    # Distribution instantanée
    dist = pd.DataFrame({
        "metric": ["bike_avail","dock_avail","penury","saturation"],
        "count": [
            int(has_bike.sum()) if has_bike is not None else 0,
            int(has_dock.sum()) if has_dock is not None else 0,
            int(pen.sum()) if pen is not None else 0,
            int(sat.sum()) if sat is not None else 0,
        ],
    })
    dist["total_active"] = active
    dist["pct"] = dist.apply(lambda r: safe_ratio(r["count"], r["total_active"]), axis=1)

    return kpis, dist

def compute_recent_coverage_volatility(df: pd.DataFrame, cols: Cols, tzname: str, last_days: int) -> Tuple[float, float]:
    if last_days <= 0: return float("nan"), float("nan")
    df = df.copy()
    # fenêtre récente
    ts_local = to_local(df, cols.ts, tzname)
    tmax = ts_local.max()
    start = tmax - pd.Timedelta(days=last_days)
    mask = (ts_local >= start) & (ts_local <= tmax)
    win = df.loc[mask].copy()
    if win.empty:
        return float("nan"), float("nan")

    # couverture = moyenne station de (nb_ts_station / nb_ts_total)
    total_ts = win[cols.ts].nunique()
    per_station = (win.groupby(cols.station)[cols.ts].nunique() / max(total_ts,1)).reindex(win[cols.station].unique()).fillna(0.0)
    coverage_pct = float((per_station.mean() * 100.0).round(2))

    # volatilité intra-journalière (jour local courant)
    start_day, end_day = today_bounds_local(to_local(df, cols.ts, tzname))
    today = df[(ts_local >= start_day) & (ts_local < end_day)].copy()
    vol = (today.groupby(cols.station)[cols.bikes].std(ddof=0)).median()
    return coverage_pct, float(0.0 if pd.isna(vol) else round(vol, 2))

def compute_today_vs_median_curve(df: pd.DataFrame, cols: Cols, tzname: str, ref_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    ts_loc = to_local(df, cols.ts, tzname)
    df["_ts_loc"] = ts_loc
    df["_hhmm"] = df["_ts_loc"].dt.strftime("%H:%M")
    df["_weekday"] = df["_ts_loc"].dt.weekday

    # ➜ has_bike avant les slices
    bikes_all = pd.to_numeric(df[cols.bikes], errors="coerce")
    df["__has_bike"] = bikes_all > 0

    # Aujourd'hui (dernier jour local)
    start_day, end_day = today_bounds_local(ts_loc)
    today = df[(ts_loc >= start_day) & (ts_loc < end_day)].copy()
    if today.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Fenêtre de référence: ref_days en amont, même weekday que 'today'
    ref_start = start_day - pd.Timedelta(days=ref_days)
    ref = df[(ts_loc >= ref_start) & (ts_loc < start_day) & (df["_weekday"] == start_day.weekday())].copy()

    def agg_part_has_bike(d: pd.DataFrame) -> pd.DataFrame:
        out = (d.groupby(["_ts_loc","_hhmm"])[["__has_bike"]]
                 .agg(has_bike=("__has_bike", "mean"),
                      n=("__has_bike", "size"))
                 .reset_index())
        out["pct"] = out["has_bike"] * 100.0
        return out

    today_curve = agg_part_has_bike(today)
    ref_curve = agg_part_has_bike(ref)

    # Médiane par hh:mm sur la ref (série)
    ref_med_series = ref_curve.groupby("_hhmm")["pct"].median()

    # Reindexer sur les 96 quarts d'heure pour éviter une ligne vide
    bins = pd.Index(pd.date_range("00:00", "23:45", freq="15min").strftime("%H:%M"))
    ref_med_series = ref_med_series.reindex(bins)

    # -> DataFrame propre avec noms robustes
    ref_median = ref_med_series.reset_index()
    # les deux colonnes issues de reset_index() : [index, pct] ou similaire
    c0, c1 = ref_median.columns[0], ref_median.columns[1]
    ref_median = ref_median.rename(columns={c0: "_hhmm", c1: "pct_median"})

    # Aligner et ordonner
    today_curve = today_curve.merge(ref_median, on="_hhmm", how="left").sort_values("_hhmm")
    ref_median = ref_median.sort_values("_hhmm")

    # Debug
    nunique_days = ref["_ts_loc"].dt.normalize().nunique()
    non_nan = int(ref_median["pct_median"].notna().sum())
    print(f"[overview] ref jours distincts (meme weekday) = {nunique_days}, bins mediane non-NaN = {non_nan}/96")


    return today_curve, ref_median


def save_day_profile_plot(today_curve: pd.DataFrame, ref_median: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(9, 4))

    if today_curve.empty:
        plt.text(0.5, 0.5, "Données insuffisantes pour 'aujourd’hui'", ha="center", va="center")
        plt.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    x = today_curve["_hhmm"].tolist()
    y_today = today_curve["pct"].tolist()

    # mapping médiane
    ref_map = {}
    if not ref_median.empty and "pct_median" in ref_median.columns:
        ref_map = dict(zip(ref_median["_hhmm"], ref_median["pct_median"]))
    y_ref = [ref_map.get(k, np.nan) for k in x]
    ref_valid = np.isfinite(y_ref).sum()

    plt.plot(x, y_today, label="Aujourd’hui (≥1 vélo)", linewidth=2)
    if ref_valid > 0:
        plt.plot(x, y_ref, linestyle="--", label="Médiane (mêmes jours)")
    else:
        plt.annotate("Médiane indisponible (historique insuffisant)",
                     xy=(0.02, 0.92), xycoords="axes fraction", fontsize=9)

    plt.ylim(0, 100)
    plt.xticks(np.linspace(0, len(x)-1, 9))
    plt.ylabel("% stations avec ≥1 vélo")
    plt.xlabel("Heure (locale)")
    plt.title("Aujourd’hui vs médiane (mêmes jours)")
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_map(snapshot: pd.DataFrame, cols: Cols, out_html: Path) -> None:
    lat_col, lon_col = cols.lat, cols.lon
    if lat_col is None or lon_col is None or lat_col not in snapshot.columns or lon_col not in snapshot.columns:
        with open(out_html, "w", encoding="utf-8") as f:
            f.write("<html><body><p>Carte non disponible (lat/lon manquantes).</p></body></html>")
        return

    bikes = pd.to_numeric(snapshot[cols.bikes], errors="coerce")
    docks_col = cols.docks_avail
    if docks_col is None and cols.capacity:
        docks = (pd.to_numeric(snapshot[cols.capacity], errors="coerce") - bikes).clip(lower=0)
        snapshot["__docks_avail"] = docks
        docks_col = "__docks_avail"

    has_docks = bool(docks_col and docks_col in snapshot.columns)

    pen = (bikes == 0)
    sat = (snapshot[docks_col] == 0) if has_docks else pd.Series(False, index=snapshot.index)

    lat0 = float(snapshot[lat_col].median())
    lon0 = float(snapshot[lon_col].median())

    m = folium.Map(location=[lat0, lon0], zoom_start=12, tiles="cartodbpositron")

    for _, row in snapshot.iterrows():
        lat = float(row[lat_col]); lon = float(row[lon_col])
        name = str(row[cols.name]) if cols.name and cols.name in snapshot.columns else str(row[cols.station])
        b = int(pd.to_numeric(row[cols.bikes], errors="coerce"))
        d = int(pd.to_numeric(row[docks_col], errors="coerce")) if has_docks else None
        color = "red" if b == 0 else ("black" if (d == 0 if d is not None else False) else "blue")
        html = f"<b>{name}</b><br/>bikes={b}" + (f" · docks_avail={d}" if d is not None else "")
        folium.CircleMarker([lat, lon], radius=4, color=color, fill=True, fill_opacity=0.7, weight=0.5,
                            popup=folium.Popup(html, max_width=250)).add_to(m)


    # --- Légende simple (compat large) ---
    from branca.element import Element

    legend_items = []
    legend_items.append(("<span style='background:red'></span> Pénurie (0 vélo)"))
    if has_docks:
        legend_items.append(("<span style='background:black'></span> Saturation (0 place)"))
    legend_items.append(("<span style='background:blue'></span> OK / autre"))

    legend_html = f"""
    <style>
      .map-legend {{
        position: fixed; bottom: 24px; left: 24px; z-index: 9999;
        background: #fff; padding: 8px 10px; border: 1px solid #bbb;
        border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,.1);
        font: 12px/1.3 -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial, sans-serif;
      }}
      .map-legend .row {{ display: flex; align-items: center; margin: 3px 0; }}
      .map-legend .dot {{
        display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px;
      }}
      .map-legend .dot.red {{ background:red; }}
      .map-legend .dot.black {{ background:black; }}
      .map-legend .dot.blue {{ background:blue; }}
    </style>
    <div class="map-legend">
      <div style="font-weight:600; margin-bottom:4px;">Légende</div>
      <div class="row"><span class="dot red"></span>Pénurie (0 vélo)</div>
      {('<div class="row"><span class="dot black"></span>Saturation (0 place)</div>' if has_docks else '')}
      <div class="row"><span class="dot blue"></span>OK / autre</div>
    </div>
    """
    m.get_root().html.add_child(Element(legend_html))
    # -------------------------------------

    m.save(str(out_html))


# -------------------------- Rendu Markdown --------------------------

MD_TEMPLATE = """# Aperçu du réseau

Cette page donne un **coup d’œil instantané** à la santé du réseau (snapshot le plus récent) et situe la **journée en cours** par rapport aux semaines précédentes.

## KPIs (snapshot {ts_local})
- **Stations actives** : **{stations_active}** / **{stations_universe}** (offline : {stations_offline})
- **Disponibilité vélo** : **{availability_bike_pct:.2f}%**
- **Disponibilité place** : **{availability_dock_pct:.2f}%**  
- **Taux de pénurie** : **{penury_pct:.2f}%** · **Taux de saturation** : **{saturation_pct:.2f}%**
- **Couverture récente** (sur {last_days} j) : **{coverage_pct:.2f}%**
- **Volatilité intra-journée** (écart-type vélos / station, médiane) : **{volatility_today:.2f}**

---

## Carte instantanée (pénurie / saturation)
<div style="margin: 0.5rem 0;">
  <iframe src="{map_rel}" style="width:100%;height:520px;border:0" loading="lazy" title="Carte instantanée du réseau"></iframe>
</div>

---

## KPIs du jour vs J-7 / J-14 / J-21
![KPIs du jour vs lags]({kpi_fig_rel})

> Comparaison des **parts de stations** (sur la **journée locale entière**) :  
> disponibilité vélo/place, pénurie, saturation. Les barres J-7/J-14/J-21
> sont calculées sur les journées locales correspondantes.

---

## Courbe « Aujourd’hui vs médiane (mêmes jours) »
![Courbe jour]({fig_rel})

> La courbe trace la **part de stations** avec ≥1 vélo à chaque **hh:mm locale** pour **aujourd’hui**, et la compare à la **médiane** des mêmes jours sur {ref_days} jours.

---

## Tables d’appui
- KPIs (CSV/JSON) :  
  - `{kpis_csv_rel}`  
  - `{kpis_json_rel}`
- Distribution instantanée : `{dist_csv_rel}`
- Stations en tension (sur {last_days} j) : `{tension_csv_rel}`

---

## Méthodologie (résumé)
- **Source** : `docs/exports/events.parquet` (timestamps **15 min** UTC naïfs).  
- **Snapshot** : dernier `ts` ; pénurie = `bikes == 0` ; saturation = `docks_avail == 0` (ou `capacity - bikes == 0` si `docks_avail` absent).  
- **Couverture** : moyenne station de `#ts_observés / #ts_total` sur la fenêtre **{last_days} jours**.  
- **Volatilité** : écart-type des vélos par station sur la **journée locale courante**, médiane des stations.  
- **Courbe** : part(≥1 vélo) par hh:mm pour aujourd’hui vs **médiane** des mêmes **weekday** sur **{ref_days} jours**.

> Limites : si `docks_avail` n’est pas disponible, les métriques “place/saturation” sont approximées via `capacity - bikes`.

"""

# -------------------------- Main --------------------------

def compute_daily_kpis_for_day(df: pd.DataFrame, cols: Cols, tzname: str, day_start_local: pd.Timestamp) -> dict:
    ts_loc = to_local(df, cols.ts, tzname)
    day_end = day_start_local + pd.Timedelta(days=1)
    d = df[(ts_loc >= day_start_local) & (ts_loc < day_end)].copy()
    if d.empty:
        return {"avail_bike": np.nan, "avail_dock": np.nan, "pen": np.nan, "sat": np.nan}

    b = pd.to_numeric(d[cols.bikes], errors="coerce")
    has_bike = (b > 0)

    docks = None
    if cols.docks_avail and cols.docks_avail in d.columns:
        docks = pd.to_numeric(d[cols.docks_avail], errors="coerce")
    elif cols.capacity and cols.capacity in d.columns:
        docks = (pd.to_numeric(d[cols.capacity], errors="coerce") - b).clip(lower=0)

    has_dock = (docks > 0) if docks is not None else None
    pen = (b == 0)
    sat = (docks == 0) if docks is not None else None

    return {
        "avail_bike": float((has_bike.mean() * 100.0).round(2)),
        "avail_dock": float((has_dock.mean() * 100.0).round(2)) if has_dock is not None else np.nan,
        "pen":        float((pen.mean() * 100.0).round(2)),
        "sat":        float((sat.mean() * 100.0).round(2)) if sat is not None else np.nan,
    }

def save_kpi_bars_today_vs_lags(k_today: dict, k_lags: dict, out_path: Path) -> None:
    metrics = [("avail_bike","Disponibilité vélo"), ("avail_dock","Disponibilité place"),
               ("pen","Pénurie"), ("sat","Saturation")]
    labels = ["Aujourd'hui","J-7","J-14","J-21"]

    data = []
    for key, _ in metrics:
        row = [k_today.get(key, np.nan),
               k_lags.get("J-7", {}).get(key, np.nan),
               k_lags.get("J-14", {}).get(key, np.nan),
               k_lags.get("J-21", {}).get(key, np.nan)]
        data.append(row)

    x = np.arange(len(metrics)); w = 0.2
    fig = plt.figure(figsize=(9, 3.8))
    for i, lab in enumerate(labels):
        vals = [data[m][i] for m in range(len(metrics))]
        plt.bar(x + (i-1.5)*w, vals, width=w, label=lab)

    plt.xticks(x, [m[1] for m in metrics])
    plt.ylabel("% (part de stations)")
    plt.title("KPIs du jour vs J-7 / J-14 / J-21")
    plt.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_today_sparkline(today_curve: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(9, 1.8))
    if not today_curve.empty:
        plt.plot(today_curve["_hhmm"], today_curve["pct"])
        plt.fill_between(np.arange(len(today_curve)), today_curve["pct"], alpha=0.1)
        plt.ylim(0, 100)
        plt.xticks([], []); plt.yticks([], [])
    else:
        plt.text(0.5, 0.5, "Pas de données pour aujourd'hui", ha="center", va="center")
        plt.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(events_path: Path, tzname: str, last_days: int, ref_days: int) -> None:
    if not events_path.exists():
        raise FileNotFoundError(f"[overview] Introuvable: {events_path}")

    # Lecture minimale
    df = pd.read_parquet(events_path)
    cols = detect_columns(df)

    # Univers de stations = stations vues sur la fenêtre (ref) pour robustesse
    ts_loc_all = to_local(df, cols.ts, tzname)
    tmax = ts_loc_all.max()
    uni_start = tmax - pd.Timedelta(days=max(last_days, ref_days, 7))
    station_universe = (
        df.loc[(ts_loc_all >= uni_start) & (ts_loc_all <= tmax), cols.station]
        .astype(str).dropna().unique().tolist()
    )

    # KPIs snapshot + distribution
    kpis, dist = compute_snapshot_kpis(df, cols, tzname, station_universe)

    # Couverture & Volatilité
    coverage_pct, volatility_today = compute_recent_coverage_volatility(df, cols, tzname, last_days)
    kpis["coverage_pct"] = coverage_pct
    kpis["volatility_today"] = volatility_today
    kpis["last_days"] = last_days
    kpis["ref_days"] = ref_days

    # Sauvegarde tables KPI
    kpis_csv = TABLES / "kpis_today.csv"
    kpis_json = TABLES / "kpis_today.json"
    pd.DataFrame([kpis]).to_csv(kpis_csv, index=False)
    with open(kpis_json, "w", encoding="utf-8") as f:
        json.dump(kpis, f, ensure_ascii=False, indent=2)

    dist_csv = TABLES / "snapshot_distribution.csv"
    dist.to_csv(dist_csv, index=False)

    # Stations en tension sur last_days
    ts_loc = to_local(df, cols.ts, tzname)
    start_ld = tmax - pd.Timedelta(days=last_days)
    win = df[(ts_loc >= start_ld) & (ts_loc <= tmax)].copy()
    # pénurie / saturation par station
    pen_rate = win.groupby(cols.station)[cols.bikes].apply(lambda s: (pd.to_numeric(s, errors="coerce") == 0).mean())
    if cols.docks_avail and cols.docks_avail in win.columns:
        sat_rate = win.groupby(cols.station)[cols.docks_avail].apply(lambda s: (pd.to_numeric(s, errors="coerce") == 0).mean())
    elif cols.capacity and cols.capacity in win.columns:
        cap = pd.to_numeric(win[cols.capacity], errors="coerce")
        bks = pd.to_numeric(win[cols.bikes], errors="coerce")
        sat_rate = ((cap - bks) <= 0).groupby(win[cols.station]).mean()
    else:
        sat_rate = pd.Series(np.nan, index=pen_rate.index)

    tension = pd.DataFrame({
        "station_id": pen_rate.index.astype(str),
        "penury_rate": pen_rate.values,
        "saturation_rate": sat_rate.values,
    }).sort_values(["penury_rate","saturation_rate"], ascending=False)
    tension_csv = TABLES / "stations_tension_last_days.csv"
    tension.to_csv(tension_csv, index=False)

    # Courbe Aujourd'hui vs médiane
    today_curve, ref_med = compute_today_vs_median_curve(df, cols, tzname, ref_days)
    fig_path = FIGS / "day_profile_today_vs_median.png"
    save_day_profile_plot(today_curve, ref_med, fig_path)

    # Jour local courant (début de journée)
    start_day, end_day = today_bounds_local(ts_loc_all)

    # KPIs journaliers (moyenne sur toute la journée locale)
    k_today = compute_daily_kpis_for_day(df, cols, tzname, start_day)
    k_lags = {}
    for dlag, label in [(7,"J-7"), (14,"J-14"), (21,"J-21")]:
        k_lags[label] = compute_daily_kpis_for_day(df, cols, tzname, start_day - pd.Timedelta(days=dlag))

    fig_kpis = FIGS / "kpis_today_vs_lags.png"
    save_kpi_bars_today_vs_lags(k_today, k_lags, fig_kpis)

    # Carte snapshot
    last_ts = df[cols.ts].max()
    snapshot = df[df[cols.ts] == last_ts].copy()
    map_html = MAPS / "network_overview.html"
    build_map(snapshot, cols, map_html)

    # --------------------- Rendu Markdown avec injection ---------------------
    md = MD_TEMPLATE.format(
        ts_local=kpis["snapshot_ts_local"],
        stations_active=kpis["stations_active"],
        stations_universe=kpis["stations_universe"],
        stations_offline=kpis["stations_offline"],
        availability_bike_pct=kpis["availability_bike_pct"] if not pd.isna(kpis["availability_bike_pct"]) else float("nan"),
        availability_dock_pct=kpis["availability_dock_pct"] if not pd.isna(kpis["availability_dock_pct"]) else float("nan"),
        penury_pct=kpis["penury_pct"] if not pd.isna(kpis["penury_pct"]) else float("nan"),
        saturation_pct=kpis["saturation_pct"] if not pd.isna(kpis["saturation_pct"]) else float("nan"),
        coverage_pct=kpis["coverage_pct"] if not pd.isna(kpis["coverage_pct"]) else float("nan"),
        volatility_today=kpis["volatility_today"] if not pd.isna(kpis["volatility_today"]) else float("nan"),
        last_days=last_days,
        ref_days=ref_days,
        map_rel=rel_from_md(OUT_MD, map_html),
        kpi_fig_rel=rel_from_md(OUT_MD, fig_kpis),
        fig_rel=rel_from_md(OUT_MD, fig_path),
        kpis_csv_rel=rel_from_md(OUT_MD, kpis_csv),
        kpis_json_rel=rel_from_md(OUT_MD, kpis_json),
        dist_csv_rel=rel_from_md(OUT_MD, dist_csv),
        tension_csv_rel=rel_from_md(OUT_MD, tension_csv),
    )

    with open(OUT_MD, "w", encoding="utf-8", newline="\n") as f:
        f.write(md)

    print(f"[overview] OK -> {OUT_MD}")
    print(f"[overview] figs -> {fig_path}")
    print(f"[overview] map  -> {map_html}")
    print(f"[overview] tables -> {TABLES}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build Réseau / Aperçu (KPIs, carte, figures, MD)")
    ap.add_argument("--events", type=Path, default=DOCS / "exports" / "events.parquet", help="Chemin vers events.parquet")
    ap.add_argument("--tz", type=str, default="Europe/Paris", help="Timezone locale (ex: Europe/Paris)")
    ap.add_argument("--last-days", type=int, default=7, help="Fenêtre récente (jours) pour la couverture/volatilité et stations en tension")
    ap.add_argument("--ref-days", type=int, default=28, help="Fenêtre de référence (jours) pour la médiane (mêmes weekdays)")
    args = ap.parse_args()

    print(f"[overview] start events={args.events} tz={args.tz} last_days={args.last_days} ref_days={args.ref_days}")
    try:
        main(args.events, args.tz, args.last_days, args.ref_days)
    except Exception as e:
        import traceback
        print("[overview] FATAL:", e)
        traceback.print_exc()
        raise
