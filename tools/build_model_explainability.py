# tools/build_model_explainability.py
# -----------------------------------------------------------------------------
# Modèle — Explicabilité & calibration
#
# Rôle
# ----
# Produire les **diagnostics résidus**, **calibration** (globale & segments),
# **importance** (si bundle dispo), **PDP** top-3, et **incertitude** (si colonnes).
# Générer la page `docs/model/explainability.md`.
#
# Entrées
# -------
# - `docs/exports/perf.parquet` (doit contenir y_true, y_pred, baseline, ts, station_id)
# - (optionnel) `assets/tables/network/stations/station_clusters.csv` pour segments
#
# Sorties
# -------
# - `docs/assets/figs/model/explainability/*.png` (+ map HTML si folium dispo)
# - `docs/assets/tables/model/explainability/*.csv`
# - `docs/assets/maps/bias_by_station.html` (si coords)
# - `docs/model/explainability.md`
#
# Notes
# -----
# - Cadence temporelle : pas **5 minutes** (arrondi des timestamps).
# - L’overview se contente d’**associations** (pas de causalité).
#
# CLI
# ---
# python tools/build_model_explainability.py \
#   --perf docs/exports/perf.parquet --last-days 7 --tz Europe/Paris
#   # ajouter --no-md pour ne pas écrire la page Markdown
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

# Matplotlib (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Carte biais optionnelle
try:
    import folium
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False


# --------------------------- Paths ---------------------------

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
FIGS_DIR = ASSETS / "figs" / "model" / "explainability"
TABLES_DIR = ASSETS / "tables" / "model" / "explainability"
MAPS_DIR = ASSETS / "maps"
OUT_MD = DOCS / "model" / "explainability.md"
STATION_CLUSTERS = ASSETS / "tables" / "network" / "stations" / "station_clusters.csv"

# --------------------------- Utils ---------------------------

def _mkdirs() -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

def _save_fig(path: Path) -> None:
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()

def _migrate_old_table_paths() -> None:
    """Déplace d'anciens CSV écrits par erreur sous figs/ vers tables/."""
    candidates = [
        (FIGS_DIR.parent / "acf_values.csv", TABLES_DIR / "acf_values.csv"),
        (FIGS_DIR.parent / "heteroscedasticity_by_true_quantiles.csv", TABLES_DIR / "heteroscedasticity_by_true_quantiles.csv"),
        (FIGS_DIR.parent / "error_episodes_by_station.csv", TABLES_DIR / "error_episodes_by_station.csv"),
    ]
    moved = []
    for src, dst in candidates:
        try:
            if src.exists() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.replace(dst)
                moved.append((src.name, "moved"))
        except Exception:
            moved.append((src.name, "skip-error"))
    if moved:
        print("[explain] migrate old tables:", moved)

def _read_perf(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"[explain] Introuvable: {path}")
    df = pd.read_parquet(path)

    # ts
    if "ts" not in df.columns:
        raise KeyError("[explain] Colonne 'ts' manquante")
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True).dt.floor("5min")

    # station id
    sid = None
    for c in ("station_id", "stationcode", "stationCode", "id"):
        if c in df.columns:
            sid = c
            break
    if sid is None:
        raise KeyError("[explain] Identifiant station manquant (station_id/stationcode)")
    df["station_id"] = df[sid].astype(str)

    # y_true / y_pred
    if "y_true" not in df.columns:
        raise KeyError("[explain] Colonne 'y_true' manquante")
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df.get("y_pred", np.nan), errors="coerce")

    # baseline optionnelle
    if "y_pred_baseline" in df.columns:
        df["y_pred_baseline"] = pd.to_numeric(df["y_pred_baseline"], errors="coerce")
    else:
        df["y_pred_baseline"] = np.nan

    # lat/lon optionnels — colonnes toujours présentes (NaN sinon)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce") if "lat" in df.columns else np.nan
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce") if "lon" in df.columns else np.nan
    return df

def _localize(df: pd.DataFrame, tz: Optional[str]) -> pd.DataFrame:
    if tz:
        ldt = df["ts"].dt.tz_convert(tz)
        return df.assign(date_local=ldt.dt.date, dow=ldt.dt.dayofweek, hour=ldt.dt.hour)
    # ts est déjà UTC aware
    return df.assign(date_local=df["ts"].dt.tz_convert("UTC").dt.date,
                     dow=df["ts"].dt.dayofweek, hour=df["ts"].dt.hour)

def _metrics(y_true: pd.Series, y_hat: pd.Series) -> Dict[str, float]:
    e = (y_true - y_hat).astype(float)
    return {
        "mae": float(np.nanmean(np.abs(e))),
        "rmse": float(np.sqrt(np.nanmean(e**2))),
        "me": float(np.nanmean(e)),
    }

def _qqplot_points(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.array([]), np.array([])
    xs = np.sort((x - np.mean(x)) / (np.std(x) if np.std(x) > 0 else 1.0))
    n = xs.size
    p = (np.arange(1, n + 1) - 0.5) / n
    th = np.sqrt(2) * erfinv(2 * p - 1)
    return th, xs

def erfinv(y: np.ndarray) -> np.ndarray:
    a = 0.147
    sgn = np.sign(y)
    ln = np.log(1 - y**2)
    first = 2 / (np.pi * a) + ln / 2
    second = ln / a
    return sgn * np.sqrt(np.sqrt(first**2 - second) - first)

def _acf(x: pd.Series, nlags: int = 144) -> np.ndarray:
    """
    ACF sur le résidu moyen (nlags par défaut ≈ 12h à 5 min → 12*60/5 = 144).
    """
    x = pd.Series(x).astype(float)
    x = x - x.mean()
    acf = np.zeros(nlags + 1)
    denom = (x**2).sum()
    if denom == 0 or len(x) == 0:
        return acf
    for k in range(nlags + 1):
        num = (x.iloc[:-k or None] * x.shift(k).iloc[:-k or None]).sum() if k > 0 else denom
        acf[k] = num / denom
    return acf

def _group_apply(gb, func):
    """Compat pandas 2.2 (include_groups) et versions antérieures."""
    try:
        return gb.apply(func, include_groups=False)
    except TypeError:
        return gb.apply(func)


# --------------------------- Sections (assets) ---------------------------

def residual_diagnostics(perf: pd.DataFrame, out_dir_figs: Path) -> None:
    df = perf[perf["y_pred"].notna()].copy()
    if df.empty:
        return

    df["resid"] = (df["y_true"] - df["y_pred"]).astype(float)

    # Histogramme
    plt.figure(figsize=(6, 3.5))
    plt.hist(df["resid"].values, bins=60)
    plt.title("Résidus — histogramme")
    plt.xlabel("Erreur (y_true - y_pred)")
    plt.ylabel("Comptes")
    _save_fig(out_dir_figs / "residual_hist.png")

    # QQ-plot vs N(0,1)
    th, xs = _qqplot_points(df["resid"].values)
    plt.figure(figsize=(6, 3.5))
    if xs.size > 0:
        xs_std = (xs - xs.mean()) / (xs.std() if xs.std() > 0 else 1.0)
        plt.scatter(th, xs_std, s=10)
        plt.plot([np.min(th), np.max(th)], [np.min(th), np.max(th)], linewidth=1)
    plt.title("QQ-plot des résidus vs N(0,1)")
    plt.xlabel("Quantiles théoriques")
    plt.ylabel("Quantiles empiriques")
    _save_fig(out_dir_figs / "residual_qqplot.png")

    # ACF du résidu moyen par timestamp (lag 5 min)
    ts_mean = (df.groupby("ts")["resid"].mean())
    ac = _acf(ts_mean, nlags=144)  # 12h à 5 min
    pd.DataFrame({"lag": np.arange(len(ac)), "acf": ac}).to_csv(
        TABLES_DIR / "acf_values.csv", index=False
    )
    plt.figure(figsize=(7, 3.5))
    plt.stem(np.arange(len(ac)), ac, basefmt=" ")
    plt.title("ACF du résidu moyen (lag 5 min)")
    plt.xlabel("Lag (×5 min)")
    plt.ylabel("ACF")
    _save_fig(out_dir_figs / "residual_acf.png")

    # Hétéroscédasticité : MAE vs quantiles de y_true
    q = pd.qcut(df["y_true"], q=20, duplicates="drop")
    het = (_group_apply(
        df.assign(bin=q).groupby("bin", observed=False),
        lambda g: pd.Series({
            "mae": float(np.nanmean(np.abs(g["y_true"] - g["y_pred"]))),
            "n": int(len(g))
        })
    ).reset_index())
    het.to_csv(TABLES_DIR / "heteroscedasticity_by_true_quantiles.csv", index=False)

    plt.figure(figsize=(7, 3.5))
    plt.plot(np.arange(len(het)), het["mae"], marker="o")
    plt.title("Hétéroscédasticité — MAE par quantile de y_true")
    plt.xlabel("Quantiles de y_true (20)")
    plt.ylabel("MAE")
    _save_fig(out_dir_figs / "heteroscedasticity_mae_by_true_quantile.png")

def error_episodes(perf: pd.DataFrame) -> None:
    df = perf[perf["y_pred"].notna()].copy()
    if df.empty:
        return
    df["resid"] = (df["y_true"] - df["y_pred"]).astype(float)
    df = df.sort_values(["station_id", "ts"])

    TH = 4.0
    def tag_sequences(g: pd.DataFrame) -> pd.DataFrame:
        x = (np.abs(g["resid"].values) >= TH).astype(int)
        best = 0
        cur = 0
        for v in x:
            if v:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return pd.Series({"max_run": int(best), "n": int(len(g))})

    episodes = (_group_apply(df.groupby("station_id"), tag_sequences)
                .reset_index().sort_values("max_run", ascending=False))
    episodes.to_csv(TABLES_DIR / "error_episodes_by_station.csv", index=False)

def calibration(perf: pd.DataFrame, out_dir_figs: Path, out_dir_tables: Path) -> None:
    df = perf[perf["y_pred"].notna()].copy()
    if df.empty:
        return

    # Fit global
    x = df["y_pred"].astype(float).values
    y = df["y_true"].astype(float).values
    if np.isfinite(x).all() and np.isfinite(y).all() and x.size > 1:
        b, a = np.polyfit(x, y, 1)  # y ~ a + b*x
    else:
        a, b = np.nan, np.nan

    # Scatter
    plt.figure(figsize=(5.8, 5.2))
    plt.scatter(x[::200], y[::200], s=4, alpha=0.3)
    lim = [np.nanmin([x.min(), y.min()]), np.nanmax([x.max(), y.max()])]
    if np.isfinite(lim).all():
        plt.plot(lim, lim, linewidth=1, label="y = x")
    if np.isfinite(a) and np.isfinite(b):
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)
        plt.plot(xs, a + b * xs, linewidth=1, label=f"fit: y = {a:.2f} + {b:.2f}x")
    plt.title("Calibration globale")
    plt.xlabel("y_pred"); plt.ylabel("y_true"); plt.legend(loc="best")
    _save_fig(out_dir_figs / "calibration_scatter.png")

    # Binning 20 quantiles
    q = pd.qcut(df["y_pred"], q=20, duplicates="drop")
    cal_bin = (_group_apply(
        df.groupby(q, observed=False),
        lambda g: pd.Series({
            "y_pred_mean": float(np.nanmean(g["y_pred"])) if g["y_pred"].notna().any() else np.nan,
            "y_true_mean": float(np.nanmean(g["y_true"])) if g["y_true"].notna().any() else np.nan,
            "n": int(len(g))
        })
    ).reset_index(names=["bin"]))
    cal_bin.to_csv(out_dir_tables / "calibration_binned.csv", index=False)

    plt.figure(figsize=(5.8, 5.2))
    plt.scatter(cal_bin["y_pred_mean"], cal_bin["y_true_mean"], s=25)
    lim = [np.nanmin([cal_bin["y_pred_mean"].min(), cal_bin["y_true_mean"].min()]),
           np.nanmax([cal_bin["y_pred_mean"].max(), cal_bin["y_true_mean"].max()])]
    if np.isfinite(lim).all():
        plt.plot(lim, lim)
    plt.title("Calibration (binning en 20 quantiles)")
    plt.xlabel("y_pred moyen"); plt.ylabel("y_true moyen")
    _save_fig(out_dir_figs / "calibration_curve.png")

    # β par heure
    by_hour = (df.groupby("hour")
                 .apply(lambda g: pd.Series({
                     "alpha": float(np.polyfit(g["y_pred"], g["y_true"], 1)[1]) if len(g) > 1 else np.nan,
                     "beta": float(np.polyfit(g["y_pred"], g["y_true"], 1)[0]) if len(g) > 1 else np.nan,
                     "n": int(len(g))
                 })).reset_index())
    by_hour.to_csv(out_dir_tables / "calibration_by_hour.csv", index=False)
    plt.figure(figsize=(8, 3.6))
    plt.plot(by_hour["hour"], by_hour["beta"], marker="o", label="β (pente)")
    plt.axhline(1.0, linewidth=1)
    plt.title("Calibration — pente par heure")
    plt.xlabel("Heure"); plt.ylabel("β")
    plt.legend(loc="best")
    _save_fig(out_dir_figs / "calibration_beta_by_hour.png")

    # Erreur relative (Bas/Moyen/Haut)
    tert = pd.qcut(df["y_true"], q=3, duplicates="drop", labels=["Bas", "Moyen", "Haut"])
    rel = (_group_apply(
        df.assign(level=tert).groupby("level", observed=False),
        lambda g: pd.Series({
            "mape_like": float(np.nanmean(np.abs(g["y_true"] - g["y_pred"]) / np.maximum(1.0, g["y_true"]))),
            "n": int(len(g))
        })
    ).reset_index())
    rel.to_csv(out_dir_tables / "relative_error_by_level.csv", index=False)
    plt.figure(figsize=(6.4, 3.6))
    plt.bar(rel["level"].astype(str), rel["mape_like"].astype(float))
    plt.title("Erreur relative par niveau d'occupation (proxy MAPE)")
    plt.xlabel("Niveau"); plt.ylabel("Erreur relative")
    _save_fig(out_dir_figs / "relative_error_by_level.png")

    # Biais par station + carte (si coords)
    bias_station = (df.assign(resid=(df["y_true"] - df["y_pred"]))
                      .groupby("station_id")
                      .agg(bias=("resid","mean"),
                           lat=("lat","last"),
                           lon=("lon","last"),
                           n=("resid","size"))
                      .reset_index())
    bias_station.to_csv(out_dir_tables / "bias_by_station.csv", index=False)

    if HAS_FOLIUM and bias_station["lat"].notna().any() and bias_station["lon"].notna().any():
        lat0 = float(bias_station["lat"].median())
        lon0 = float(bias_station["lon"].median())
        m = folium.Map(location=[lat0, lon0], tiles="cartodbpositron", zoom_start=12, control_scale=True)
        for _, row in bias_station.iterrows():
            if np.isnan(row["lat"]) or np.isnan(row["lon"]):
                continue
            color = "green" if row["bias"] <= 0 else "red"
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=max(3, min(12, abs(row["bias"]))),
                fill=True, fill_opacity=0.8,
                color=color,
                tooltip=f"Station {row['station_id']} — biais={row['bias']:.2f} (n={row['n']})"
            ).add_to(m)
        m.save(str(MAPS_DIR / "bias_by_station.html"))

def segments_by_cluster(perf: pd.DataFrame, out_dir_tables: Path) -> None:
    if not STATION_CLUSTERS.exists():
        return
    try:
        clusters = pd.read_csv(STATION_CLUSTERS, dtype={"station_id": str})
    except Exception:
        return

    df = perf.copy()
    df["station_id"] = df["station_id"].astype(str)

    # join clusters and keep only rows with a cluster
    df = (
        df.merge(clusters[["station_id", "cluster"]], on="station_id", how="left")
          .dropna(subset=["cluster"])
    )
    if df.empty:
        return

    # compute errors
    e = (pd.to_numeric(df["y_true"], errors="coerce") -
         pd.to_numeric(df["y_pred"], errors="coerce")).astype(float)
    df["abs_err"] = e.abs()

    # one-pass aggregation to avoid index order issues
    def _per_cluster(g: pd.DataFrame) -> pd.Series:
        err = (g["y_true"] - g["y_pred"]).astype(float)
        return pd.Series({
            "mae": float(np.nanmean(np.abs(err))),
            "rmse": float(np.sqrt(np.nanmean(err**2))),
            "n": int(len(g)),
        })

    # compat with pandas 2.2+ and older via your helper if present
    try:
        by_cluster = df.groupby("cluster").apply(_per_cluster, include_groups=False).reset_index()
    except TypeError:
        by_cluster = df.groupby("cluster").apply(_per_cluster).reset_index()

    by_cluster.to_csv(out_dir_tables / "errors_by_cluster.csv", index=False)

# --------------------------- Importance & PDP ---------------------------

def _load_bundle() -> tuple[object | None, list[str] | None]:
    """Retourne (model, features) quel que soit le format renvoyé par src.forecast.load_model_bundle."""
    try:
        mod = importlib.import_module("src.forecast")
    except Exception:
        return None, None

    loader = getattr(mod, "load_model_bundle", None)
    if loader is None:
        return None, None

    try:
        bundle = loader()
    except Exception:
        return None, None

    model, features = None, None

    if isinstance(bundle, dict):
        model = (bundle.get("model") or bundle.get("estimator") or bundle.get("clf") or bundle.get("regressor"))
        features = (bundle.get("feat_cols") or bundle.get("features") or bundle.get("X_cols") or bundle.get("columns") or bundle.get("feature_names"))

    if model is None or features is None:
        try:
            items = list(bundle) if not isinstance(bundle, dict) else []
        except TypeError:
            items = []
        for obj in items:
            if hasattr(obj, "predict"):
                model = obj
                break
        for obj in items:
            if isinstance(obj, (list, tuple)) and all(isinstance(c, str) for c in obj):
                features = list(obj)
                break
        if hasattr(bundle, "_fields"):
            for name in bundle._fields:
                val = getattr(bundle, name)
                if model is None and hasattr(val, "predict"):
                    model = val
                if features is None and isinstance(val, (list, tuple)) and all(isinstance(c, str) for c in val):
                    features = list(val)

    if features is not None and not isinstance(features, (list, tuple)):
        try:
            features = list(features)
        except Exception:
            features = None

    return model, (list(features) if features is not None else None)

def feature_importance(perf: pd.DataFrame, out_dir_tables: Path) -> None:
    model, X_cols = _load_bundle()
    if model is None or X_cols is None:
        return

    df = perf.dropna(subset=["y_pred"]).copy()
    if df.empty:
        return

    miss = [c for c in X_cols if c not in df.columns]
    if miss:
        return

    X = df[X_cols].astype(float)
    y = df["y_true"].astype(float)

    base = _metrics(y, model.predict(X))["mae"]
    res = []
    rng = np.random.default_rng(42)
    for col in X_cols:
        Xp = X.copy()
        Xp[col] = rng.permutation(Xp[col].values)
        mae = _metrics(y, model.predict(Xp))["mae"]
        res.append({"feature": col, "delta_mae": float(mae - base)})

    imp = (pd.DataFrame(res).sort_values("delta_mae", ascending=False).reset_index(drop=True))
    imp.to_csv(out_dir_tables / "permutation_importance.csv", index=False)

def pdp_top3(perf: pd.DataFrame, out_dir_figs: Path) -> None:
    model, X_cols = _load_bundle()
    if model is None or X_cols is None:
        return

    df = perf.dropna(subset=["y_pred"]).copy()
    miss = [c for c in X_cols if c not in df.columns]
    if miss or df.empty:
        return

    imp_path = TABLES_DIR / "permutation_importance.csv"
    if imp_path.exists():
        try:
            imp = pd.read_csv(imp_path)
            top = (imp.sort_values("delta_mae", ascending=False)["feature"].head(3).tolist())
        except Exception:
            top = []
    else:
        X = df[X_cols].astype(float)
        y = df["y_true"].astype(float)
        base = _metrics(y, model.predict(X))["mae"]
        res = []
        rng = np.random.default_rng(42)
        for col in X_cols:
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            mae = _metrics(y, model.predict(Xp))["mae"]
            res.append({"feature": col, "delta_mae": float(mae - base)})
        top = (pd.DataFrame(res).sort_values("delta_mae", ascending=False)["feature"].head(3).tolist())

    if not top:
        return

    dfX = df[X_cols].astype(float)
    try:
        _ = model.predict(dfX)
    except Exception:
        return

    for feat in top:
        xs = np.linspace(dfX[feat].quantile(0.01), dfX[feat].quantile(0.99), 30)
        ys = []
        Xtmp = dfX.copy()
        for v in xs:
            Xtmp[feat] = v
            try:
                ys.append(float(np.nanmean(model.predict(Xtmp))))
            except Exception:
                ys.append(np.nan)
        plt.figure(figsize=(6, 3.6))
        plt.plot(xs, ys)
        plt.title(f"PDP — {feat}")
        plt.xlabel(feat); plt.ylabel("y_pred moyen")
        _save_fig(out_dir_figs / f"pdp_{feat}.png")


# --------------------------- Incertitude ---------------------------

def uncertainty_coverage(perf: pd.DataFrame, out_dir_tables: Path) -> None:
    df = perf.copy()
    if not {"yhat_lo", "yhat_hi", "y_pred"}.issubset(df.columns):
        return
    df = df.dropna(subset=["y_pred", "yhat_lo", "yhat_hi"])
    if df.empty:
        return
    inside = ((df["y_true"] >= df["yhat_lo"]) & (df["y_true"] <= df["yhat_hi"])).astype(int)
    rate = float(inside.mean())
    pd.DataFrame([{"coverage_empirical": rate, "n": int(len(df))}]).to_csv(
        out_dir_tables / "uncertainty_coverage.csv", index=False
    )


# --------------------------- DOC (overview-style) ---------------------------

MD_TEMPLATE = """# Modèle — Explicabilité & calibration

> Rendre les prévisions **intelligibles** (quelles variables comptent ? quand, où ?) et **fiables** (calibration, biais, incertitudes).

---

## Objectif
Rendre les prévisions **intelligibles** (quelles variables comptent ? quand, où ?) et **fiables** (calibration, biais, incertitudes).

---

## Résidus & diagnostic
- **Résidus** `y_true − y_pred` : distribution, QQ-plot, autocorrélation.  
- **Hétéroscédasticité** : variance des résidus vs niveau d’occupation.  
- **Outliers/épisodes** : séquences d’erreurs anormalement longues (liées à ruptures d’ingestion, événements).

{residuals_grid}

**Tables d’appui**  
{residuals_tables}

---

## Importance & explications
- **Permutation importance** (globale) sur échantillon **time-aware**.  
- **Ablation** par familles de features (lags, saisonnalité, météo) pour la **valeur incrémentale**.  
- **Profils moyens conditionnels** (PDP/ICE) sur variables clés.  
- **Segments** : importance et erreurs **par cluster de stations** (transparence sur où le modèle “comprend” mieux).

**Sorties disponibles**  
{importance_tables}

{pdp_grid}

---

## Calibration & biais
- **Régression d’étalonnage** `y_true = α + β·y_pred` :  
  - **β ≈ 1** & **α ≈ 0** → bonne calibration moyenne.  
  - Pentes par **segments** (heure, cluster, capacité, zone) pour détecter des biais structurels.  
- **Erreur relative** par niveau d’occupation (bas/moyen/haut) — utile pour l’opérationnel.

{calib_grid}

{bias_map}

**Tables d’appui**  
{calib_tables}

---

## Incertitude (si activée)
- **Intervalles** par quantiles ou par **jackknife/bootstrap**.  
- **Coverage nominal vs empirique** (ex. 80 % nominal ↔ ~80 % observé).  
- Signalement des **stations à forte incertitude** (utile pour le monitoring).

{uncertainty_tables}

---

## Visualisations attendues
- **Cartes** du biais par station/zone, **barres** d’importance, **PDP** pour 2–3 features clés, **courbes** de calibration globales et par segments.

---

## Lecture & limites
- L’explicabilité **décrit des associations**, pas des causalités.  
- La calibration moyenne peut être bonne tout en étant **mauvaise localement** : d’où l’analyse par segments.

---

## Valeur de la section “Modèle”
- **Opérationnel** : savoir quand/où la prévision est fiable, et de combien elle améliore la baseline.  
- **Ingénierie** : pipeline clair, versionné, reproductible.  
- **Confiance** : transparence sur **pourquoi** le modèle prédit ce qu’il prédit, et **comment** il se comporte selon les contextes.
"""

def _rel_from_md(target: Path) -> str:
    rel_fs = os.path.relpath(target, OUT_MD.parent).replace("\\", "/")
    return "../" + rel_fs  # <- compense 'directory URLs'

def _grid(fig_paths: List[Tuple[Path, str]], cols: int) -> str:
    items = []
    for p, alt in fig_paths:
        if p.exists():
            items.append(f'<figure><img src="{_rel_from_md(p)}" alt="{alt}" /><figcaption>{alt}</figcaption></figure>')
    if not items:
        return "_(aucune figure disponible)_"
    return f'<div style="display:grid;grid-template-columns:repeat({cols},1fr);gap:12px">' + "".join(items) + "</div>"

def _tables_list(rows: List[Tuple[Optional[Path], str]]) -> str:
    lines = []
    for p, lab in rows:
        if p is not None and p.exists():
            lines.append(f"- {lab} → `{_rel_from_md(p)}`")
    return "\n".join(lines) if lines else "_(aucune table disponible)_"

def build_overview_md() -> str:
    # Résidus (figures)
    residuals_grid = _grid([
        (FIGS_DIR / "residual_hist.png", "Distribution des résidus"),
        (FIGS_DIR / "residual_qqplot.png", "QQ-plot vs N(0,1)"),
        (FIGS_DIR / "residual_acf.png", "ACF du résidu moyen (lag 5 min)"),
        (FIGS_DIR / "heteroscedasticity_mae_by_true_quantile.png", "MAE par quantile de y_true (hétéroscédasticité)"),
    ], cols=2)

    # Résidus (tables) — avec fallback si de vieux fichiers restent sous figs/
    acf_candidates = [
        TABLES_DIR / "acf_values.csv",
        FIGS_DIR.parent / "acf_values.csv",
    ]
    acf_file = next((p for p in acf_candidates if p.exists()), None)
    het_candidates = [
        TABLES_DIR / "heteroscedasticity_by_true_quantiles.csv",
        FIGS_DIR.parent / "heteroscedasticity_by_true_quantiles.csv",
    ]
    het_file = next((p for p in het_candidates if p.exists()), None)
    epi_candidates = [
        TABLES_DIR / "error_episodes_by_station.csv",
        FIGS_DIR.parent / "error_episodes_by_station.csv",
    ]
    epi_file = next((p for p in epi_candidates if p.exists()), None)
    residuals_tables = _tables_list([
        (het_file, "Hétéroscédasticité (20 quantiles)"),
        (acf_file, "Valeurs ACF"),
        (epi_file, "Épisodes d’erreurs par station"),
    ])

    # Calibration (figures)
    calib_grid = _grid([
        (FIGS_DIR / "calibration_scatter.png", "Calibration globale — y_true = α + β·y_pred"),
        (FIGS_DIR / "calibration_curve.png", "Calibration par binning (20 quantiles de y_pred)"),
        (FIGS_DIR / "calibration_beta_by_hour.png", "Pente β par heure locale"),
        (FIGS_DIR / "relative_error_by_level.png", "Erreur relative (proxy MAPE) par niveau d’occupation"),
    ], cols=2)

    # Carte biais
    bias_map = ""
    map_html = MAPS_DIR / "bias_by_station.html"
    if map_html.exists():
        bias_map = (
            '<div style="margin:0.5rem 0;">'
            f'<iframe src="{_rel_from_md(map_html)}" style="width:100%;height:520px;border:0" loading="lazy" title="Carte du biais par station"></iframe>'
            '</div>'
        )

    # Calibration (tables)
    calib_tables = _tables_list([
        (TABLES_DIR / "calibration_binned.csv", "Binning calibration"),
        (TABLES_DIR / "calibration_by_hour.csv", "β/α par heure"),
        (TABLES_DIR / "relative_error_by_level.csv", "Erreur relative (Bas/Moyen/Haut)"),
        (TABLES_DIR / "bias_by_station.csv", "Biais par station"),
    ])

    # Importance
    importance_tables = _tables_list([
        (TABLES_DIR / "permutation_importance.csv", "Importance (CSV)"),
        (TABLES_DIR / "errors_by_cluster.csv", "Erreurs par cluster"),
    ])
    pdp_grid = _grid([(p, p.stem.replace("pdp_", "PDP — ")) for p in sorted(FIGS_DIR.glob("pdp_*.png"))], cols=3)

    # Incertitude
    uncertainty_tables = _tables_list([
        (TABLES_DIR / "uncertainty_coverage.csv", "Coverage des intervalles (empirique)"),
    ])

    return MD_TEMPLATE.format(
        residuals_grid=residuals_grid,
        residuals_tables=residuals_tables,
        importance_tables=importance_tables,
        pdp_grid=pdp_grid,
        calib_grid=calib_grid,
        bias_map=bias_map,
        calib_tables=calib_tables,
        uncertainty_tables=uncertainty_tables,
    )


# --------------------------- Main ---------------------------

def main(perf_path: str, last_days: Optional[int], tz: Optional[str], write_md: bool) -> None:
    _mkdirs()
    perf = _read_perf(Path(perf_path))
    if last_days:
        tmax = perf["ts"].max()
        # last_days exprimé en jours → filtre en minutes (pas 5 min déjà géré par floor plus haut)
        perf = perf[perf["ts"] >= (tmax - np.timedelta64(int(last_days * 24 * 60), "m"))].copy()
    # Localisation horaire pour features de découpe (hour/dow/date)
    perf = _localize(perf, tz)

    # Assets
    residual_diagnostics(perf, FIGS_DIR)
    error_episodes(perf)
    calibration(perf, FIGS_DIR, TABLES_DIR)
    segments_by_cluster(perf, TABLES_DIR)
    feature_importance(perf, TABLES_DIR)   # optionnel
    pdp_top3(perf, FIGS_DIR)               # optionnel
    uncertainty_coverage(perf, TABLES_DIR) # optionnel

    # Migration éventuelle d'anciens emplacements
    _migrate_old_table_paths()

    # Doc
    if write_md:
        md = build_overview_md()
        with open(OUT_MD, "w", encoding="utf-8", newline="\n") as f:
            f.write(md)
        print(f"[explain] Page écrite -> {OUT_MD}")

    # Logs utiles
    print("[explain] figs ->", [p.name for p in sorted(FIGS_DIR.glob("*.png"))])
    print("[explain] tables ->", [p.name for p in sorted(TABLES_DIR.glob("*.csv"))])
    if (MAPS_DIR / "bias_by_station.html").exists():
        print("[explain] map -> bias_by_station.html")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf", type=str, required=True, help="Chemin vers docs/exports/perf.parquet")
    ap.add_argument("--last-days", type=int, default=None, help="Limiter à N derniers jours")
    ap.add_argument("--tz", type=str, default="Europe/Paris", help="Timezone locale (ex: Europe/Paris)")
    ap.add_argument("--no-md", action="store_true", help="Ne pas écrire docs/model/explainability.md")
    args = ap.parse_args()
    main(perf_path=args.perf, last_days=args.last_days, tz=args.tz, write_md=not args.no_md)
