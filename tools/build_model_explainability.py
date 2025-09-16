# tools/build_model_explainability.py
# Page builder — "Modèle / Explicabilité & calibration"
#
# Produit :
# - Résidus & diagnostics : histogramme, QQ-plot normal, ACF (résidu moyen temporel),
#   hétéroscédasticité (MAE par niveau d’occupation), épisodes d’erreurs.
# - Calibration : régression y_true ~ y_pred (global & segments), erreurs relatives par niveaux.
# - Segments : erreurs par cluster (si table disponible).
# - Importance (si bundle modèle + features dispo) : permutation importance globale
#   + ablation par familles de features ; PDP (top-3).
# - Incertitude (si colonnes présentes) : coverage nominal vs empirique.
#
# CLI :
#   python tools/build_model_explainability.py --perf docs/exports/perf.parquet --last-days 14 --tz Europe/Paris
#
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional
try:
    import folium  # pour carte biais par station (optionnel)
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

STATION_CLUSTERS = ASSETS / "tables" / "network" / "stations" / "station_clusters.csv"


# --------------------------- Utils ---------------------------

def _mkdirs() -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)

def _save_fig(path: Path) -> None:
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()

def _read_perf(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"[explain] Introuvable: {path}")
    df = pd.read_parquet(path)
    # ts
    if "ts" not in df.columns:
        raise KeyError("[explain] Colonne 'ts' manquante")
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce").dt.floor("15min")
    # station
    sid = None
    for c in ("station_id", "stationcode", "stationCode", "id"):
        if c in df.columns:
            sid = c
            break
    if sid is None:
        raise KeyError("[explain] Identifiant station manquant (station_id/stationcode)")
    df["station_id"] = df[sid].astype(str)
    # targets & preds
    if "y_true" not in df.columns:
        raise KeyError("[explain] Colonne 'y_true' manquante")
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df.get("y_pred", np.nan), errors="coerce")
    # baseline si utile pour comparaison de diag
    if "y_pred_baseline" in df.columns:
        df["y_pred_baseline"] = pd.to_numeric(df["y_pred_baseline"], errors="coerce")
    else:
        df["y_pred_baseline"] = np.nan
    # meta (lat/lon optionnels pour carte biais) — toujours présentes (NaN si absentes)
    if "lat" in df.columns:
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    else:
        df["lat"] = np.nan
    if "lon" in df.columns:
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    else:
        df["lon"] = np.nan
    return df

def _localize(df: pd.DataFrame, tz: Optional[str]) -> pd.DataFrame:
    if tz:
        ldt = df["ts"].dt.tz_localize("UTC").dt.tz_convert(tz)
        return df.assign(date_local=ldt.dt.date, dow=ldt.dt.dayofweek, hour=ldt.dt.hour)
    return df.assign(date_local=df["ts"].dt.date, dow=df["ts"].dt.dayofweek, hour=df["ts"].dt.hour)

def _metrics(y_true: pd.Series, y_hat: pd.Series) -> Dict[str, float]:
    e = (y_true - y_hat).astype(float)
    mae = float(np.nanmean(np.abs(e)))
    rmse = float(np.sqrt(np.nanmean(e**2)))
    me = float(np.nanmean(e))
    return {"mae": mae, "rmse": rmse, "me": me}

def _qqplot_points(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Retourne (th_quantiles_normal, sample_sorted) pour QQ-plot vs N(0,1)."""
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.array([]), np.array([])
    xs = np.sort((x - np.mean(x)) / (np.std(x) if np.std(x) > 0 else 1.0))
    n = xs.size
    # quantiles théoriques ~ N(0,1) via probit des rangs (i-0.5)/n
    p = (np.arange(1, n + 1) - 0.5) / n
    th = np.sqrt(2) * erfinv(2 * p - 1)
    return th, xs

def erfinv(y: np.ndarray) -> np.ndarray:
    """Approximation num. de l'inverse de l'erreur."""
    # Source: Winitzki approx (suffisant pour QQ-plot)
    a = 0.147
    sgn = np.sign(y)
    ln = np.log(1 - y**2)
    first = 2 / (np.pi * a) + ln / 2
    second = ln / a
    return sgn * np.sqrt(np.sqrt(first**2 - second) - first)

def _acf(x: pd.Series, nlags: int = 48) -> np.ndarray:
    """ACF naïf jusqu'à nlags."""
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

# --------------------------- Sections ---------------------------

def residual_diagnostics(perf: pd.DataFrame, out_dir_figs: Path, tz: Optional[str]) -> None:
    """Histogramme des résidus, QQ-plot, ACF, hétéroscédasticité (MAE vs y_true)."""
    df = perf.copy()
    mask = df["y_pred"].notna()
    df = df[mask].copy()
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
        plt.scatter(th, (xs - xs.mean()) / (xs.std() if xs.std() > 0 else 1.0), s=10)
        plt.plot([th.min(), th.max()], [th.min(), th.max()], linewidth=1)
    plt.title("QQ-plot des résidus vs N(0,1)")
    plt.xlabel("Quantiles théoriques"); plt.ylabel("Quantiles empiriques")
    _save_fig(out_dir_figs / "residual_qqplot.png")

    # ACF du résidu moyen par timestamp
    ts_mean = (df.groupby("ts")["resid"].mean())
    ac = _acf(ts_mean, nlags=48)  # jusque 12h à 15min
    pd.DataFrame({"lag": np.arange(len(ac)), "acf": ac}).to_csv(out_dir_figs.parent / "acf_values.csv", index=False)
    plt.figure(figsize=(7, 3.5))
    plt.stem(np.arange(len(ac)), ac, basefmt=" ")
    plt.title("ACF du résidu moyen (lag 15min)")
    plt.xlabel("Lag (×15 min)"); plt.ylabel("ACF")
    _save_fig(out_dir_figs / "residual_acf.png")

    # Hétéroscédasticité : MAE vs niveau d'occupation (en quantiles de y_true)
    q = pd.qcut(df["y_true"], q=20, duplicates="drop")
    df = df.assign(bin=q)
    het = (df.groupby("bin", observed=False)
             .apply(lambda g: pd.Series({
                 "mae": float(np.nanmean(np.abs(g["y_true"] - g["y_pred"]))),
                 "n": int(len(g))
             }))
             .reset_index())
    het.to_csv(out_dir_figs.parent / "heteroscedasticity_by_true_quantiles.csv", index=False)
    plt.figure(figsize=(7, 3.5))
    plt.plot(np.arange(len(het)), het["mae"], marker="o")
    plt.title("Hétéroscédasticité — MAE par quantile de y_true")
    plt.xlabel("Quantiles de y_true (20)"); plt.ylabel("MAE")
    _save_fig(out_dir_figs / "heteroscedasticity_mae_by_true_quantile.png")

def error_episodes(perf: pd.DataFrame, out_dir_figs: Path) -> None:
    """Détecte épisodes continus où |erreur| dépasse un seuil (p.ex. 4 vélos)."""
    df = perf.copy()
    df = df[df["y_pred"].notna()].copy()
    if df.empty:
        return
    df["resid"] = (df["y_true"] - df["y_pred"]).astype(float)
    df = df.sort_values(["station_id", "ts"])

    # tag sequences au-dessus du seuil
    TH = 4.0
    def tag_sequences(g: pd.DataFrame) -> pd.DataFrame:
        x = (np.abs(g["resid"].values) >= TH).astype(int)
        # plus longue séquence continue de 1
        count = 0
        best = 0
        for v in x:
            if v == 1:
                count += 1
                best = max(best, count)
            else:
                count = 0
        return pd.Series({"max_run": int(best), "n": int(len(g))})

    episodes = (df.groupby("station_id")
                  .apply(tag_sequences)
                  .reset_index()
                  .sort_values("max_run", ascending=False))
    episodes.to_csv(out_dir_figs.parent / "error_episodes_by_station.csv", index=False)

def calibration(perf: pd.DataFrame, out_dir_figs: Path, out_dir_tables: Path) -> None:
    df = perf.copy()
    mask = df["y_pred"].notna()
    df = df[mask].copy()
    if df.empty:
        return

    # Global linear fit y = a + b*x
    x = df["y_pred"].astype(float).values
    y = df["y_true"].astype(float).values
    if np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and x.size > 1:
        b, a = np.polyfit(x, y, 1)  # y ~ a + b*x
    else:
        a, b = np.nan, np.nan

    # Scatter + y=x + fit
    plt.figure(figsize=(5.8, 5.2))
    plt.scatter(x[::200], y[::200], s=4, alpha=0.3)  # sous-échantillonné pour lisibilité
    lim = [np.nanmin([x.min(), y.min()]), np.nanmax([x.max(), y.max()])]
    if np.isfinite(lim).all():
        plt.plot(lim, lim, linewidth=1, label="y = x")
    if np.isfinite(a) and np.isfinite(b):
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)
        plt.plot(xs, a + b * xs, linewidth=1, label=f"fit: y = {a:.2f} + {b:.2f}x")
    plt.title("Calibration globale")
    plt.xlabel("y_pred"); plt.ylabel("y_true"); plt.legend(loc="best")
    _save_fig(out_dir_figs / "calibration_scatter.png")

    # Courbe calibration par binning (quantiles de y_pred)
    q = pd.qcut(df["y_pred"], q=20, duplicates="drop")
    cal_bin = (df.groupby(q, observed=False)
                 .apply(lambda g: pd.Series({
                     "y_pred_mean": float(
                         np.nanmean(g["y_pred"]) if g["y_pred"].notna().any() else np.nan
                     ),
                     "y_true_mean": float(
                         np.nanmean(g["y_true"]) if g["y_true"].notna().any() else np.nan
                     ),
                     "n": int(len(g))
                 }))
                 .reset_index(names=["bin"]))
    cal_bin.to_csv(out_dir_tables / "calibration_binned.csv", index=False)

    plt.figure(figsize=(5.8, 5.2))
    plt.scatter(cal_bin["y_pred_mean"], cal_bin["y_true_mean"], s=25)
    lim = [np.nanmin([cal_bin["y_pred_mean"].min(), cal_bin["y_true_mean"].min()]),
           np.nanmax([cal_bin["y_pred_mean"].max(), cal_bin["y_true_mean"].max()])]
    if np.isfinite(lim).all():
        plt.plot(lim, lim)  # y=x
    plt.title("Calibration (binning en 20 quantiles)")
    plt.xlabel("y_pred moyen"); plt.ylabel("y_true moyen")
    _save_fig(out_dir_figs / "calibration_curve.png")

    # Par segments (heure, éventuellement cluster)
    # Heures
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

    # Erreur relative par niveau d'occupation (bas/moyen/haut)
    tert = pd.qcut(df["y_true"], q=3, duplicates="drop", labels=["Bas", "Moyen", "Haut"])
    rel = (df.assign(level=tert)
             .groupby("level", observed=False)
             .apply(lambda g: pd.Series({
                 "mape_like": float(np.nanmean(np.abs(g["y_true"] - g["y_pred"]) / np.maximum(1.0, g["y_true"]))),
                 "n": int(len(g))
             }))
             .reset_index())
    rel.to_csv(out_dir_tables / "relative_error_by_level.csv", index=False)
    plt.figure(figsize=(6.4, 3.6))
    plt.bar(rel["level"].astype(str), rel["mape_like"].astype(float))
    plt.title("Erreur relative par niveau d'occupation (proxy MAPE)")
    plt.xlabel("Niveau"); plt.ylabel("Erreur relative")
    _save_fig(out_dir_figs / "relative_error_by_level.png")

    # Biais par station (moyenne résidu) + carte optionnelle
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
        m = folium.Map(location=[lat0, lon0], 
                       tiles="cartodbpositron",
                       zoom_start=12, control_scale=True)
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
    """Erreurs agrégées par cluster si table clusters dispo."""
    if not STATION_CLUSTERS.exists():
        return
    try:
        clusters = pd.read_csv(STATION_CLUSTERS, dtype={"station_id": str})
    except Exception:
        return
    df = (perf.merge(clusters[["station_id", "cluster"]], on="station_id", how="left")
               .dropna(subset=["cluster"]))
    if df.empty:
        return
    df["abs_err"] = (df["y_true"] - df["y_pred"]).abs()
    by_cluster = (df.groupby("cluster")
             .apply(lambda g: pd.Series({
                 "mae": float(np.nanmean(g["abs_err"])),
                 "rmse": float(np.sqrt(np.nanmean((g["y_true"] - g["y_pred"])**2))),
                 "n": int(len(g))
             }))).reset_index()
    by_cluster.to_csv(out_dir_tables / "errors_by_cluster.csv", index=False)

# --------------------------- Importance & PDP (si dispo) ---------------------------

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

    # Cas dict-like
    if isinstance(bundle, dict):
        model = (bundle.get("model")
                 or bundle.get("estimator")
                 or bundle.get("clf")
                 or bundle.get("regressor"))
        features = (bundle.get("features")
                    or bundle.get("X_cols")
                    or bundle.get("columns")
                    or bundle.get("feature_names"))

    # Cas tuple/list/namedtuple
    if model is None or features is None:
        try:
            items = list(bundle) if not isinstance(bundle, dict) else []
        except TypeError:
            items = []
        # Cherche un objet avec .predict pour le modèle
        for obj in items:
            if hasattr(obj, "predict"):
                model = obj
                break
        # Cherche une liste/tuple de str pour les features
        for obj in items:
            if isinstance(obj, (list, tuple)) and all(isinstance(c, str) for c in obj):
                features = list(obj)
                break
        # Support namedtuple (accès par attributs)
        if hasattr(bundle, "_fields"):
            for name in bundle._fields:
                val = getattr(bundle, name)
                if model is None and hasattr(val, "predict"):
                    model = val
                if features is None and isinstance(val, (list, tuple)) and all(isinstance(c, str) for c in val):
                    features = list(val)

    # Normalise features en list[str]
    if features is not None and not isinstance(features, (list, tuple)):
        try:
            features = list(features)
        except Exception:
            features = None

    return model, (list(features) if features is not None else None)


def feature_importance(perf: pd.DataFrame, out_dir_tables: Path) -> None:
    """Permutation importance globale si bundle dispo (robuste dict/tuple)."""
    model, X_cols = _load_bundle()
    if model is None or X_cols is None:
        return

    df = perf.dropna(subset=["y_pred"]).copy()
    if df.empty:
        return

    # Vérifie que les colonnes features existent dans perf
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

    imp = (pd.DataFrame(res)
             .sort_values("delta_mae", ascending=False)
             .reset_index(drop=True))
    imp.to_csv(out_dir_tables / "permutation_importance.csv", index=False)


    # échantillonner X/y à partir des lignes non-na
    df = perf.dropna(subset=["y_pred"]).copy()
    if df.empty:
        return
    # supposer que perf contient les features live encodées (sinon abandon)
    miss = [c for c in X_cols if c not in df.columns]
    if miss:
        return
    X = df[X_cols].astype(float)
    y = df["y_true"].astype(float)

    # permutation importance rudimentaire (delta MAE)
    base = _metrics(y, model.predict(X))["mae"]
    res = []
    rng = np.random.default_rng(42)
    for col in X_cols:
        Xp = X.copy()
        Xp[col] = rng.permutation(Xp[col].values)
        mae = _metrics(y, model.predict(Xp))["mae"]
        res.append({"feature": col, "delta_mae": float(mae - base)})
    imp = (pd.DataFrame(res)
             .sort_values("delta_mae", ascending=False)
             .reset_index(drop=True))
    imp.to_csv(out_dir_tables / "permutation_importance.csv", index=False)

def pdp_top3(perf: pd.DataFrame, out_dir_figs: Path) -> None:
    """PDP 1D pour les 3 features les plus importantes (robuste dict/tuple)."""
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
            top = (imp.sort_values("delta_mae", ascending=False)["feature"]
                     .head(3).tolist())
        except Exception:
            top = []
    else:
        # Fallback: recalc rapide
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
        top = (pd.DataFrame(res)
                 .sort_values("delta_mae", ascending=False)["feature"]
                 .head(3).tolist())

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


    # importance si déjà écrite (sinon recalcul simple)
    imp_path = TABLES_DIR / "permutation_importance.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path)
        top = imp.sort_values("delta_mae", ascending=False)["feature"].head(3).tolist()
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
        top = (pd.DataFrame(res)
                 .sort_values("delta_mae", ascending=False)["feature"]
                 .head(3).tolist())

    if not top:
        return
    # PDP naïf par feature (moyenne des prédictions quand on fait varier la feature)
    dfX = df[X_cols].astype(float)
    yhat = None
    try:
        yhat = model.predict(dfX)
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

# --------------------------- Incertitude (si colonnes) ---------------------------

def uncertainty_coverage(perf: pd.DataFrame, out_dir_tables: Path) -> None:
    """Compare coverage nominal vs empirique si colonnes d'intervalle présentes."""
    df = perf.copy()
    cols = [c for c in ("yhat_lo", "yhat_hi", "y_pred") if c in df.columns]
    if not set(("yhat_lo", "yhat_hi", "y_pred")).issubset(set(df.columns)):
        return
    df = df.dropna(subset=["y_pred", "yhat_lo", "yhat_hi"])
    if df.empty:
        return

    inside = ((df["y_true"] >= df["yhat_lo"]) & (df["y_true"] <= df["yhat_hi"])).astype(int)
    rate = float(inside.mean())
    pd.DataFrame([{"coverage_empirical": rate, "n": int(len(df))}]).to_csv(
        out_dir_tables / "uncertainty_coverage.csv", index=False
    )

# --------------------------- Main ---------------------------

def main(perf_path: str, last_days: Optional[int], tz: Optional[str]) -> None:
    _mkdirs()
    perf = _read_perf(Path(perf_path))
    if last_days:
        tmax = perf["ts"].max()
        perf = perf[perf["ts"] >= (tmax - np.timedelta64(int(last_days * 24 * 60), "m"))].copy()

    perf = _localize(perf, tz)

    # Sections
    residual_diagnostics(perf, FIGS_DIR, tz)
    error_episodes(perf, FIGS_DIR)
    calibration(perf, FIGS_DIR, TABLES_DIR)
    segments_by_cluster(perf, TABLES_DIR)
    feature_importance(perf, TABLES_DIR)          # optionnel
    pdp_top3(perf, FIGS_DIR)                      # optionnel
    uncertainty_coverage(perf, TABLES_DIR)        # optionnel

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf", type=str, required=True, help="Chemin vers docs/exports/perf.parquet")
    ap.add_argument("--last-days", type=int, default=None, help="Limiter à N derniers jours")
    ap.add_argument("--tz", type=str, default=None, help="Timezone locale (ex: Europe/Paris)")
    args = ap.parse_args()
    main(perf_path=args.perf, last_days=args.last_days, tz=args.tz)
