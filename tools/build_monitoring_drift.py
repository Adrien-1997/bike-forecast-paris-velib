# tools/build_monitoring_drift.py
# Génère tables/figures de dérive + la page Markdown "docs/monitoring/drift.md" (avec carte statique + lien interactif)
from __future__ import annotations

import argparse, os, json
from pathlib import Path
from typing import Optional, Dict
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import folium
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False

# --------------------------- Chemins ---------------------------

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path(".")
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
TABLES_DIR = ASSETS / "tables" / "monitoring" / "drift"
FIGS_DIR = ASSETS / "figs" / "monitoring" / "drift"
MAPS_DIR = ASSETS / "maps"
OUT_MD = DOCS / "monitoring" / "drift.md"

for d in (TABLES_DIR, FIGS_DIR, MAPS_DIR, OUT_MD.parent):
    d.mkdir(parents=True, exist_ok=True)


# ---- Helpers de chemins relatifs (compatibles MkDocs) ----
def rel_from_md(md_path: Path, target: Path) -> str:
    """
    Construit un lien relatif depuis la page Markdown md_path vers target.
    Compatibles:
      - MkDocs (use_directory_urls: true)  -> pages rendues comme dossiers (/monitoring/drift/)
      - GitHub (lecture directe du .md)
    Règle: on part de /docs/<...>.md vers /docs/<cible>.
    """
    md_rel = Path(md_path).resolve().relative_to(DOCS.resolve())
    parts = md_rel.with_suffix("").parts  # ex: ('monitoring','drift')
    # profondeur = nb de segments du chemin SANS extension -> nb de répertoires à remonter
    # ex: 'monitoring/drift.md' -> depth=2 -> "../../" pour atteindre /docs/
    depth = len(parts) if parts and parts[-1] != "index" else max(len(parts) - 1, 0)
    prefix = "../" * depth
    rel_from_docs = Path(target).resolve().relative_to(DOCS.resolve()).as_posix()
    return (prefix + rel_from_docs).replace("//", "/")

def _rel_from_md(target: Path) -> str:
    # utilitaire si tu veux appeler sans passer OUT_MD à chaque fois
    return rel_from_md(OUT_MD, target)



# --------------------------- IO ---------------------------

def _read_parquet(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"[drift] File not found: {p}")
    return pd.read_parquet(p)


# --------------------------- Helpers colonnes ---------------------------

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Colonnes attendues : ts, station_id, lat, lon, bikes, docks_avail, capacity_src, occ_ratio."""
    if df.empty:
        return df

    # ts
    tcol = None
    for c in ("ts", "tbin_utc", "timestamp"):
        if c in df.columns:
            tcol = c; break
    if tcol is None:
        raise KeyError("[drift] Missing time column (ts/tbin_utc/timestamp)")
    df["ts"] = pd.to_datetime(df[tcol], errors="coerce").dt.floor("15min")

    # station_id
    sid = None
    for c in ("station_id", "stationcode", "stationCode", "id"):
        if c in df.columns:
            sid = c; break
    if sid is None:
        raise KeyError("[drift] Missing station_id/stationcode")
    df["station_id"] = df[sid].astype(str)

    # bikes / docks_avail / capacity_src (création NaN si absent)
    if "bikes" not in df.columns:
        for c in ("num_bikes_available", "n_bikes", "available_bikes"):
            if c in df.columns:
                df["bikes"] = pd.to_numeric(df[c], errors="coerce"); break
    if "bikes" not in df.columns:
        df["bikes"] = np.nan

    if "docks_avail" not in df.columns:
        for c in ("num_docks_available", "n_docks", "available_docks"):
            if c in df.columns:
                df["docks_avail"] = pd.to_numeric(df[c], errors="coerce"); break
    if "docks_avail" not in df.columns:
        df["docks_avail"] = np.nan

    if "capacity_src" not in df.columns:
        for c in ("capacity", "capacity_est", "cap"):
            if c in df.columns:
                df["capacity_src"] = pd.to_numeric(df[c], errors="coerce"); break
    if "capacity_src" not in df.columns:
        df["capacity_src"] = np.nan

    # lat/lon
    if "lat" not in df.columns or "lon" not in df.columns:
        for la, lo in (("latitude", "longitude"), ("lat", "lng"), ("lat", "long")):
            if la in df.columns and lo in df.columns:
                df["lat"] = pd.to_numeric(df[la], errors="coerce")
                df["lon"] = pd.to_numeric(df[lo], errors="coerce")
                break
    if "lat" not in df.columns:
        df["lat"] = np.nan
    if "lon" not in df.columns:
        df["lon"] = np.nan

    # occ_ratio = bikes / capacity (fallback via capacity_est)
    if "occ_ratio" not in df.columns:
        cap_est = _estimate_capacity(df[["station_id", "bikes", "docks_avail", "capacity_src"]].copy())
        df = df.merge(cap_est, on="station_id", how="left")
        cap = df["capacity_src"].fillna(df["capacity_est"])
        with np.errstate(divide="ignore", invalid="ignore"):
            occ = (pd.to_numeric(df["bikes"], errors="coerce") / cap).replace([np.inf, -np.inf], np.nan)
        df["occ_ratio"] = occ.clip(lower=0, upper=1)

    return df[["ts", "station_id", "lat", "lon", "bikes", "docks_avail", "capacity_src", "occ_ratio"]].copy()


def _to_local(df: pd.DataFrame, tz: Optional[str]) -> pd.DataFrame:
    if tz:
        dt = pd.to_datetime(df["ts"], errors="coerce", utc=True).dt.tz_convert(tz)
    else:
        dt = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    return df.assign(date_local=dt.dt.date, dow=dt.dt.dayofweek, hour=dt.dt.hour, ts_local=dt)


def _psi_continuous(ref: pd.Series, cur: pd.Series, bins: int = 20, eps: float = 1e-9) -> float:
    a = pd.to_numeric(ref, errors="coerce").dropna()
    b = pd.to_numeric(cur, errors="coerce").dropna()
    if a.empty or b.empty:
        return np.nan
    q = np.unique(np.nanquantile(a, np.linspace(0, 1, bins + 1)))
    if len(q) < 3:
        q = np.linspace(a.min(), a.max(), bins + 1)
    ca, _ = np.histogram(a, bins=q)
    cb, _ = np.histogram(b, bins=q)
    pa = (ca / max(1, ca.sum())).astype(float) + eps
    pb = (cb / max(1, cb.sum())).astype(float) + eps
    return float(np.sum((pa - pb) * np.log(pa / pb)))


def _ks_stat(ref: pd.Series, cur: pd.Series) -> float:
    a = pd.to_numeric(ref, errors="coerce").dropna()
    b = pd.to_numeric(cur, errors="coerce").dropna()
    if a.empty or b.empty:
        return np.nan
    q = np.unique(np.nanquantile(pd.concat([a, b]), np.linspace(0, 1, 201)))
    ca, _ = np.histogram(a, bins=q)
    cb, _ = np.histogram(b, bins=q)
    cdfa = np.cumsum(ca) / max(1, ca.sum())
    cdfb = np.cumsum(cb) / max(1, cb.sum())
    return float(np.max(np.abs(cdfa - cdfb)))


def _delta_mean_var(ref: pd.Series, cur: pd.Series) -> tuple[float, float]:
    a = pd.to_numeric(ref, errors="coerce").dropna()
    b = pd.to_numeric(cur, errors="coerce").dropna()
    if a.empty or b.empty:
        return (np.nan, np.nan)
    dm = (b.mean() - a.mean()) / (a.std(ddof=1) + 1e-9)
    dv = (b.var(ddof=1) - a.var(ddof=1)) / (a.var(ddof=1) + 1e-9)
    return (float(dm), float(dv))


# --------------------------- Capacité ---------------------------

def _estimate_capacity(win: pd.DataFrame) -> pd.Series:
    """Estime la capacité par station: priorité capacity_src ; sinon quantile 0.98 de bikes+docks ; sinon bikes."""
    def est(g: pd.DataFrame) -> float:
        cap = g["capacity_src"].dropna().max() if "capacity_src" in g.columns else np.nan
        if pd.notna(cap) and cap > 0:
            return float(cap)
        if "docks_avail" in g.columns and g["docks_avail"].notna().any():
            s = (g.get("bikes", pd.Series(dtype=float)).clip(lower=0) + g["docks_avail"].clip(lower=0)).dropna()
            if len(s):
                return float(s.quantile(0.98))
        b = g.get("bikes", pd.Series(dtype=float)).clip(lower=0).dropna()
        return float(b.quantile(0.98)) if len(b) else np.nan

    cols = [c for c in ["station_id", "bikes", "docks_avail", "capacity_src"] if c in win.columns]
    return (win[cols].groupby("station_id", group_keys=False).apply(est).rename("capacity_est"))


def _assign_zone(df: pd.DataFrame) -> pd.Series:
    for c in ("arrondissement", "arr", "zone", "district"):
        if c in df.columns:
            return df[c].astype(str)
    lat = pd.to_numeric(df.get("lat"), errors="coerce")
    lon = pd.to_numeric(df.get("lon"), errors="coerce")
    mask = lat.notna() & lon.notna()
    z = pd.Series(index=df.index, dtype=object, name="zone")
    lat_r = (lat[mask] * 100).round() / 100.0
    lon_r = (lon[mask] * 100).round() / 100.0
    z.loc[mask] = lat_r.astype(str) + "," + lon_r.astype(str)
    return z


# --------------------------- Fallback carte PNG ---------------------------

def _save_zone_scatter_png(pz: pd.DataFrame, out_png: Path):
    """Vue statique du drift par zone, avec mise à l’échelle robuste des bulles."""
    if pz is None or pz.empty:
        return

    x = pd.to_numeric(pz["lon"], errors="coerce")
    y = pd.to_numeric(pz["lat"], errors="coerce")
    psi = pd.to_numeric(pz["psi"], errors="coerce").clip(lower=0)

    m = x.notna() & y.notna() & psi.notna()
    if m.sum() == 0:
        return
    x, y, psi = x[m], y[m], psi[m]

    # Échelle robuste : clip au 95e pct puis normalisation [0,1]
    clip_high = np.nanpercentile(psi, 95)
    if not np.isfinite(clip_high) or clip_high <= 0:
        clip_high = float(psi.max()) if np.isfinite(psi.max()) and psi.max() > 0 else 1.0
    norm = np.minimum(psi, clip_high) / clip_high

    # Mappe vers un RAYON en points, puis convertit en aire (points^2)
    r_min, r_max = 6, 18            # rayons lisibles
    sizes = (r_min + (r_max - r_min) * np.sqrt(norm)) ** 2  # aire = rayon^2

    plt.figure(figsize=(7, 6), dpi=160)
    sc = plt.scatter(x, y, s=sizes, c=psi, cmap="viridis", alpha=0.75, edgecolors="none")
    cbar = plt.colorbar(sc)
    cbar.set_label("PSI (clippé au 95e pct)", rotation=90)

    # Marges + aspect
    pad_x = max(0.01, (x.max() - x.min()) * 0.05)
    pad_y = max(0.01, (y.max() - y.min()) * 0.05)
    plt.xlim(x.min() - pad_x, x.max() + pad_x)
    plt.ylim(y.min() - pad_y, y.max() + pad_y)
    plt.title("Drift agrégé par zone (PSI) — vue statique")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    try:
        plt.axis("equal")
    except Exception:
        pass
    plt.grid(True, linewidth=0.3, alpha=0.4)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# --------------------------- Fenêtres ---------------------------

def _split_windows(df: pd.DataFrame, current_days: int, reference_days: int) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.Timestamp]]:
    if df.empty:
        return df.iloc[0:0].copy(), df.iloc[0:0].copy(), {}
    tmax = df["ts"].max()
    t_cur_start = tmax - pd.Timedelta(days=current_days)
    t_ref_end = t_cur_start
    t_ref_start = t_ref_end - pd.Timedelta(days=reference_days)
    ref = df[(df["ts"] >= t_ref_start) & (df["ts"] < t_ref_end)].copy()
    cur = df[(df["ts"] >= t_cur_start) & (df["ts"] <= tmax)].copy()
    bounds = {"tmax": tmax, "t_cur_start": t_cur_start, "t_ref_start": t_ref_start, "t_ref_end": t_ref_end}
    return ref, cur, bounds


# --------------------------- Calcul Drift ---------------------------

def compute_drift(events: pd.DataFrame, current_days: int, reference_days: int, tz: Optional[str]) -> dict:
    df = _ensure_columns(events)
    df = _to_local(df, tz)

    ref, cur, bounds = _split_windows(df, current_days=current_days, reference_days=reference_days)

    def agg(df_):
        return (df_.groupby(["date_local", "station_id"])
                  .agg(occ_ratio=("occ_ratio", "mean"),
                       bikes=("bikes", "mean"),
                       docks_avail=("docks_avail", "mean"),
                       lat=("lat", "median"),
                       lon=("lon", "median"))
                  .reset_index())
    ref = agg(ref); cur = agg(cur)

    # ---- Metrics (PSI / KS / Δ) ----
    feats = ["occ_ratio", "bikes", "docks_avail"]
    rows_psi, rows_ks, rows_delta = [], [], []
    for f in feats:
        rows_psi.append({"feature": f, "psi": _psi_continuous(ref[f], cur[f])})
        rows_ks.append({"feature": f, "ks": _ks_stat(ref[f], cur[f])})
        dm, dv = _delta_mean_var(ref[f], cur[f])
        rows_delta.append({"feature": f, "delta_mean": dm, "delta_var": dv})
    psi_df = pd.DataFrame(rows_psi)
    ks_df = pd.DataFrame(rows_ks)
    d_df = pd.DataFrame(rows_delta)

    # ---- PSI global journalier (proxy via occ_ratio) + EMA ----
    by_day = (df.groupby("date_local")["occ_ratio"]
                .apply(lambda s: float(np.nanmean(s))).reset_index())
    by_day = by_day.sort_values("date_local")
    alpha = 2 / (7 + 1.0)
    ema, last = [], None
    for _, r in by_day.iterrows():
        x = r["occ_ratio"]
        if pd.isna(x):
            ema.append(np.nan)
            continue
        last = x if last is None else (alpha * x + (1 - alpha) * last)
        ema.append(last)
    psi_daily_ema = pd.DataFrame({"date_local": by_day["date_local"], "psi_ema": ema})

    # ---- Résumé + alertes ----
    psi_top = psi_df.sort_values("psi", ascending=False).reset_index(drop=True)
    psi_global = float(psi_df.loc[psi_df["feature"] == "occ_ratio", "psi"].values[0]) if not psi_df.empty else np.nan
    alerts = []
    if np.isfinite(psi_global) and psi_global >= 0.25:
        alerts.append({"level": "high", "code": "psi_global_high", "text": f"PSI global élevé ({psi_global:.3f})"})
    elif np.isfinite(psi_global) and psi_global >= 0.1:
        alerts.append({"level": "medium", "code": "psi_global_medium", "text": f"PSI global modéré ({psi_global:.3f})"})
    summary = pd.DataFrame([{
        "psi_global": psi_global,
        "top_feature": psi_top.iloc[0]["feature"] if not psi_top.empty else None,
        "top_feature_psi": float(psi_top.iloc[0]["psi"]) if not psi_top.empty else np.nan,
    }])

    # ---- Carte zones (interactive + PNG + placeholder anti-404) ----
    # ⚠️ Assure-toi que MAPS_DIR = ASSETS / "maps" pour éviter les 404 dans ton site.
    map_html_path = MAPS_DIR / "drift_by_zone.html"
    map_png_path  = FIGS_DIR / "drift_by_zone.png"
    map_png_created = False

    # 1) Calcul PSI par zone (+ centroïdes)
    pz = pd.DataFrame()
    try:
        ref = ref.assign(zone=_assign_zone(ref))
        cur = cur.assign(zone=_assign_zone(cur))

        def psi_zone(df_ref: pd.DataFrame, df_cur: pd.DataFrame) -> pd.DataFrame:
            rows = []
            for z, rsub in df_ref.groupby("zone"):
                if pd.isna(z):
                    continue
                csub = df_cur[df_cur["zone"] == z]
                if csub.empty:
                    continue
                rows.append({"zone": z, "psi": _psi_continuous(rsub["occ_ratio"], csub["occ_ratio"])})
            return pd.DataFrame(rows)

        pz = psi_zone(ref, cur)
        cent = (ref.dropna(subset=["lat", "lon"])
                  .groupby("zone")[["lat", "lon"]].median().reset_index())
        pz = pz.merge(cent, on="zone", how="left").dropna(subset=["lat", "lon"])
    except Exception as e:
        print(f"[drift] zone PSI failed: {e}")

    # 2) Si aucune zone calculable -> écrire un HTML minimal (évite 404 sur le site)
    if pz.empty:
        map_html_path.parent.mkdir(parents=True, exist_ok=True)
        with open(map_html_path, "w", encoding="utf-8") as f:
            f.write(
                "<!doctype html><meta charset='utf-8'><title>Carte drift</title>"
                "<style>body{font-family:system-ui;margin:2rem;color:#444}</style>"
                "<h3>Carte non disponible</h3>"
                "<p>Aucune zone calculable sur la période sélectionnée.</p>"
            )
    else:
        # 3a) PNG fallback (toujours utile pour les renderers qui bloquent les iframes)
        _save_zone_scatter_png(pz, map_png_path)
        map_png_created = map_png_path.exists()

        # 3b) Carte interactive Folium (si disponible)
        if HAS_FOLIUM:
            lat0, lon0 = float(pz["lat"].median()), float(pz["lon"].median())
            m = folium.Map(location=[lat0, lon0], zoom_start=12, tiles="cartodbpositron")
            for _, r in pz.iterrows():
                psi = float(r["psi"]) if np.isfinite(r["psi"]) else 0.0
                rad = 3 + min(12, max(0.0, psi) * 20.0)
                folium.CircleMarker(
                    location=[float(r["lat"]), float(r["lon"])],
                    radius=rad,
                    color="red" if psi >= 0.25 else ("orange" if psi >= 0.1 else "blue"),
                    fill=True, fill_opacity=0.8,
                    tooltip=f"zone={r['zone']} • PSI={psi:.3f}"
                ).add_to(m)
            map_html_path.parent.mkdir(parents=True, exist_ok=True)
            m.save(str(map_html_path))
        else:
            # Si Folium indisponible, on fournit au moins un HTML informatif
            map_html_path.parent.mkdir(parents=True, exist_ok=True)
            with open(map_html_path, "w", encoding="utf-8") as f:
                f.write(
                    "<!doctype html><meta charset='utf-8'><title>Carte drift</title>"
                    "<style>body{font-family:system-ui;margin:2rem;color:#444}</style>"
                    "<h3>Carte statique</h3>"
                    "<p>Folium non installé. Utilisez l'image PNG générée dans les figures.</p>"
                )

    return {
        "psi_df": psi_df,
        "ks_df": ks_df,
        "deltas_df": d_df,
        "psi_daily_ema": psi_daily_ema,
        "summary": summary,
        "alerts": alerts,
        "bounds": bounds,
        "map_html": map_html_path,                                 # toujours présent (placeholder si vide)
        "map_png": map_png_path if map_png_created else None,      # présent si pz non vide
    }



# --------------------------- Figures & exports ---------------------------

def _plot_top_features(psi_df: pd.DataFrame, out: Path, top_n: int = 10):
    if psi_df is None or psi_df.empty: return
    d = psi_df.sort_values("psi", ascending=False).head(top_n)
    plt.figure(figsize=(8, 4.5)); plt.bar(d["feature"], d["psi"])
    plt.title("Top features dérivées (PSI)"); plt.ylabel("PSI")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()

def _plot_psi_ema(ema_df: pd.DataFrame, out: Path):
    if ema_df is None or ema_df.empty: return
    plt.figure(figsize=(8, 3.8)); plt.plot(pd.to_datetime(ema_df["date_local"]), ema_df["psi_ema"])
    plt.title("PSI global (EMA)"); plt.ylabel("EMA(occ_ratio)"); plt.xlabel("Date")
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()

def _export_tables(res: dict):
    res["psi_df"].to_csv(TABLES_DIR / "psi_by_feature.csv", index=False)
    res["ks_df"].to_csv(TABLES_DIR / "ks_by_feature.csv", index=False)
    res["deltas_df"].to_csv(TABLES_DIR / "deltas_by_feature.csv", index=False)
    res["psi_daily_ema"].to_csv(TABLES_DIR / "psi_global_daily_ema.csv", index=False)
    res["summary"].to_csv(TABLES_DIR / "drift_summary.csv", index=False)
    with open(TABLES_DIR / "alerts.json", "w", encoding="utf-8") as f:
        json.dump(res.get("alerts", []), f, ensure_ascii=False, indent=2)

def _export_figs(res: dict):
    _plot_top_features(res["psi_df"], FIGS_DIR / "psi_top_features.png")
    _plot_psi_ema(res["psi_daily_ema"], FIGS_DIR / "psi_global_ema.png")


# --------------------------- Rendu Markdown ---------------------------

MD_TEMPLATE = """# 2) Drift des données

> **Objectif** — Détecter la **dérive des distributions** entre une fenêtre **courante** et une fenêtre de **référence**, afin d’anticiper un **risque de dégradation** du modèle.

**Fenêtres analysées (UTC)**  
- Référence : **{ref_start} → {ref_end}**  
- Courante : **{cur_start} → {tmax}**

---

## Questions auxquelles la page répond
- Quelles **features** ont le plus dérivé ? La **cible** (`y_true`) a-t-elle changé de régime ?
- La dérive est-elle **globale** ou concentrée sur certains **segments** (clusters, zones, heures) ?
- Les dérives détectées sont-elles **persistantes** (structurelles) ou **ponctuelles** (événement) ?

---

## Indicateurs & tests
- **PSI/CSI** par variable (binning robuste).  
  *Interprétation usuelle PSI* : **< 0,10** faible · **0,10–0,25** modérée · **> 0,25** forte.
- **K–S** (variables continues), **χ²** (catégorielles).
- **Δ moyenne/variance** normalisés (z-scores).
- **Drift de cible** (prior shift) : évolution de la distribution de `y_true`.
- **Drift conditionnel** : par **cluster de stations**, par **heure du jour**, par **arrondissement/zone** (*si disponibles*).

> **Résumé courant**  
> PSI global (occ_ratio) : **{psi_global}** · Feature la plus dérivée : **{top_feature}** (PSI={top_feature_psi})

---

## Visualisations
### Top dérives (PSI)
![Top PSI]({psi_top_fig})

### Tendance du drift (EMA, occ_ratio)
![PSI global EMA]({psi_ema_fig})

### Carte — drift agrégé par zone
![Carte statique]({map_png})

[Ouvrir la carte interactive]({map_rel})

<!-- L'iframe ci-dessous est conservée pour les environnements qui l'autorisent (certains renderers la bloquent). -->
<div style="margin: 0.5rem 0;">
  <iframe src="{map_rel}" style="width:100%;height:520px;border:0" loading="lazy" title="Carte drift par zone"></iframe>
</div>

---

## Tables d’appui
- PSI par variable : `{psi_csv}`  
- K–S par variable : `{ks_csv}`  
- Δ moyenne/variance : `{delta_csv}`  
- PSI global journalier (EMA) : `{psi_ema_csv}`  
- Résumé & alertes : `{summary_csv}`, `{alerts_json}`  
- Drift de cible : `{target_csv}` (*si généré avec `--perf`*)

---

## Seuils / Alertes (par défaut, ajustables)
- **PSI global** (médiane des features clés) **> 0,10** sur **3 jours consécutifs** → **Alerte**.  
- **PSI d’une feature critique** **> 0,25** sur **2 jours** → **Alerte majeure**.  
- **Drift de cible notable** (Δ moyenne **> 1 σ**) → **Alerte**.

> ⚠️ Un **drift** n’implique pas nécessairement une **dégradation** du modèle. Consulter la page **Santé du modèle** pour corroborer (perf vs temps).

---

## Méthodes
- **Fenêtrage** : référence glissante (**{ref_days} j**) vs courant (**{cur_days} j**), sans chevauchement.  
- **Stratification** : métriques par **segment** (clusters réseau, zones).  
- **Stabilité** : lissage **EMA** pour éviter les sur-réactions au bruit.

**Artefacts & source**  
- Source : `docs/exports/events.parquet` (pas de 15 min, timestamps UTC naïfs).  
- Figures : `{figs_dir_rel}` · Tables : `{tables_dir_rel}` · Carte : `{map_rel}`
"""

def _render_markdown(res: dict, cur_days: int, ref_days: int, perf_used: bool) -> None:
    bounds = res.get("bounds", {})
    def fmt(ts):
        if ts is None: return "—"
        try:
            return pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return str(ts)

    ref_start = fmt(bounds.get("t_ref_start"))
    ref_end   = fmt(bounds.get("t_ref_end"))
    cur_start = fmt(bounds.get("t_cur_start"))
    tmax      = fmt(bounds.get("tmax"))

    summary = res.get("summary", pd.DataFrame())
    psi_global = np.nan
    top_feature = "—"; top_feature_psi = np.nan
    if not summary.empty:
        psi_global = summary["psi_global"].iloc[0]
        top_feature = summary["top_feature"].iloc[0] if pd.notna(summary["top_feature"].iloc[0]) else "—"
        top_feature_psi = summary["top_feature_psi"].iloc[0]

    psi_top_fig = _rel_from_md(FIGS_DIR / "psi_top_features.png")
    psi_ema_fig = _rel_from_md(FIGS_DIR / "psi_global_ema.png")
    map_rel    = rel_from_md(OUT_MD, res["map_html"])
    map_png    = rel_from_md(OUT_MD, res["map_png"]) if res.get("map_png") else "(non disponible)"
    psi_top_fig = rel_from_md(OUT_MD, FIGS_DIR / "psi_top_features.png")
    psi_ema_fig = rel_from_md(OUT_MD, FIGS_DIR / "psi_global_ema.png")
    psi_csv     = rel_from_md(OUT_MD, TABLES_DIR / "psi_by_feature.csv")
    ks_csv      = rel_from_md(OUT_MD, TABLES_DIR / "ks_by_feature.csv")
    delta_csv   = rel_from_md(OUT_MD, TABLES_DIR / "deltas_by_feature.csv")
    psi_ema_csv = rel_from_md(OUT_MD, TABLES_DIR / "psi_global_daily_ema.csv")
    summary_csv = rel_from_md(OUT_MD, TABLES_DIR / "drift_summary.csv")
    alerts_json = rel_from_md(OUT_MD, TABLES_DIR / "alerts.json")
    target_csv  = rel_from_md(OUT_MD, TABLES_DIR / "target_drift.csv") if perf_used else "(non généré)"


    md = MD_TEMPLATE.format(
        ref_start=ref_start, ref_end=ref_end, cur_start=cur_start, tmax=tmax,
        psi_global="nan" if pd.isna(psi_global) else f"{psi_global:.3f}",
        top_feature=top_feature,
        top_feature_psi="nan" if pd.isna(top_feature_psi) else f"{top_feature_psi:.3f}",
        psi_top_fig=psi_top_fig, psi_ema_fig=psi_ema_fig,
        map_rel=map_rel, map_png=map_png,
        psi_csv=psi_csv, ks_csv=ks_csv, delta_csv=delta_csv, psi_ema_csv=psi_ema_csv,
        summary_csv=summary_csv, alerts_json=alerts_json, target_csv=target_csv,
        ref_days=ref_days, cur_days=cur_days,
        figs_dir_rel=_rel_from_md(FIGS_DIR),
        tables_dir_rel=_rel_from_md(TABLES_DIR),
    )

    with open(OUT_MD, "w", encoding="utf-8", newline="\n") as f:
        f.write(md)


# --------------------------- Drift cible (optionnel) ---------------------------

def compute_target_drift(perf: pd.DataFrame, current_days: int, reference_days: int, tz: Optional[str]) -> Optional[pd.DataFrame]:
    if perf is None or perf.empty:
        return None
    perf = perf.copy()
    if "ts" not in perf.columns:
        for c in ("ts","tbin_utc","timestamp"):
            if c in perf.columns: perf.rename(columns={c:"ts"}, inplace=True); break
    perf["ts"] = pd.to_datetime(perf["ts"], errors="coerce")
    tmax = perf["ts"].max()
    t_cur_start = tmax - pd.Timedelta(days=current_days)
    t_ref_end = t_cur_start
    t_ref_start = t_ref_end - pd.Timedelta(days=reference_days)
    ref = perf[(perf["ts"] >= t_ref_start) & (perf["ts"] < t_ref_end)].copy()
    cur = perf[(perf["ts"] >= t_cur_start) & (perf["ts"] <= tmax)].copy()

    def _ks(a,b):
        a = pd.to_numeric(a, errors="coerce").dropna(); b = pd.to_numeric(b, errors="coerce").dropna()
        if a.empty or b.empty: return np.nan
        q = np.unique(np.nanquantile(pd.concat([a, b]), np.linspace(0, 1, 201)))
        ca,_ = np.histogram(a, bins=q); cb,_ = np.histogram(b, bins=q)
        cdfa = np.cumsum(ca)/max(1, ca.sum()); cdfb = np.cumsum(cb)/max(1, cb.sum())
        return float(np.max(np.abs(cdfa - cdfb)))

    def _dmv(a,b):
        a = pd.to_numeric(a, errors="coerce").dropna(); b = pd.to_numeric(b, errors="coerce").dropna()
        if a.empty or b.empty: return (np.nan, np.nan)
        dm = (b.mean() - a.mean()) / (a.std(ddof=1) + 1e-9)
        dv = (b.var(ddof=1) - a.var(ddof=1)) / (a.var(ddof=1) + 1e-9)
        return float(dm), float(dv)

    ks = _ks(ref.get("y_true", pd.Series(dtype=float)), cur.get("y_true", pd.Series(dtype=float)))
    dm, dv = _dmv(ref.get("y_true", pd.Series(dtype=float)), cur.get("y_true", pd.Series(dtype=float)))
    return pd.DataFrame([{"ks": ks, "delta_mean": dm, "delta_var": dv}])


# --------------------------- CLI ---------------------------

def main(
    events_path: str,
    current_days: int,
    reference_days: int,
    tz: Optional[str],
    perf_path: Optional[str] = None,
):
    events = _read_parquet(events_path)
    events = _ensure_columns(events)

    # calculs
    res = compute_drift(events, current_days=current_days, reference_days=reference_days, tz=tz)

    # exports
    _export_tables(res)
    _export_figs(res)

    # drift cible (optionnel)
    perf_used = False
    if perf_path:
        try:
            perf = _read_parquet(perf_path)
            tgt = compute_target_drift(perf, current_days=current_days, reference_days=reference_days, tz=tz)
            if tgt is not None:
                tgt.to_csv(TABLES_DIR / "target_drift.csv", index=False)
                perf_used = True
        except Exception as e:
            print(f"[drift] target drift skipped: {e}")

    # page md
    _render_markdown(res, current_days, reference_days, perf_used)

    print("[drift] Done.")
    print(f"[drift] Markdown -> {OUT_MD}")
    print(f"[drift] Tables   -> {TABLES_DIR}")
    print(f"[drift] Figures  -> {FIGS_DIR}")
    if (MAPS_DIR / "drift_by_zone.html").exists():
        print(f"[drift] Map      -> {MAPS_DIR / 'drift_by_zone.html'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, help="docs/exports/events.parquet")
    ap.add_argument("--current-days", type=int, default=7)
    ap.add_argument("--reference-days", type=int, default=28)
    ap.add_argument("--tz", default="Europe/Paris")
    ap.add_argument("--perf", default=None, help="docs/exports/perf.parquet (optional)")
    args = ap.parse_args()

    main(
        events_path=args.events,
        current_days=args.current_days,
        reference_days=args.reference_days,
        tz=args.tz,
        perf_path=args.perf,
    )
