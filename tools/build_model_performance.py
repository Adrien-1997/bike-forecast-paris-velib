# tools/build_model_performance.py
# Page builder — "Modèle / Performance & baseline"
#
# Mesure la qualité des prévisions vs baseline (persistance) et produit :
# - Tables: global_metrics, daily_error, error_by_station/hour/dow/(cluster si dispo), coverage
# - Figures: lift quotidien (sparkline), histogramme des résidus, MAE par heure, obs vs préd pour stations échantillon
#
# CLI :
#   python tools/build_model_performance.py --perf docs/exports/perf.parquet --last-days 7 --horizon 60 --tz Europe/Paris
#
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------- Paths ---------------------------

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
FIGS_DIR = ASSETS / "figs" / "model" / "performance"
TABLES_DIR = ASSETS / "tables" / "model" / "performance"


# --------------------------- Utils ---------------------------

def _mkdirs() -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

def _read_perf(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"[performance] Introuvable: {path}")
    df = pd.read_parquet(path)

    # ts
    if "ts" not in df.columns:
        raise KeyError("[performance] Colonne 'ts' manquante")
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce").dt.floor("15min")

    # station_id
    sid = None
    for c in ("station_id", "stationcode", "stationCode", "id"):
        if c in df.columns:
            sid = c
            break
    if sid is None:
        raise KeyError("[performance] Identifiant station manquant (station_id/stationcode)")
    df["station_id"] = df[sid].astype(str)

    # y_true & preds
    if "y_true" not in df.columns:
        # tolérance: certains jeux utilisent 'nb_velos_bin' comme vérité ramenée à T
        if "nb_velos_bin" in df.columns:
            df["y_true"] = pd.to_numeric(df["nb_velos_bin"], errors="coerce")
        else:
            raise KeyError("[performance] Colonne 'y_true' manquante")
    else:
        df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")

    if "y_pred_baseline" not in df.columns:
        # baseline = persistance; si absente on duplique y_true décalée (impossible ici) → fallback y_pred_baseline = y_true (lift=0)
        df["y_pred_baseline"] = df["y_true"]
    else:
        df["y_pred_baseline"] = pd.to_numeric(df["y_pred_baseline"], errors="coerce")

    if "y_pred" in df.columns:
        df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    else:
        df["y_pred"] = np.nan  # couverture 0%

    # horizon (info)
    if "horizon_min" not in df.columns:
        df["horizon_min"] = np.nan
    return df[["ts", "station_id", "y_true", "y_pred", "y_pred_baseline", "horizon_min"]].copy()

def _localize(df: pd.DataFrame, tz: Optional[str]) -> pd.DataFrame:
    if tz:
        ldt = df["ts"].dt.tz_localize("UTC").dt.tz_convert(tz)
        df = df.assign(
            date_local=ldt.dt.date,
            dow=ldt.dt.dayofweek,     # 0=Mon
            hour=ldt.dt.hour
        )
    else:
        df = df.assign(
            date_local=df["ts"].dt.date,
            dow=df["ts"].dt.dayofweek,
            hour=df["ts"].dt.hour
        )
    return df


def _metrics(y_true: pd.Series, y_hat: pd.Series) -> dict:
    err = y_true - y_hat
    mae = float(np.nanmean(np.abs(err)))
    rmse = float(np.sqrt(np.nanmean(np.square(err))))
    me = float(np.nanmean(err))
    return {"mae": mae, "rmse": rmse, "me": me}

def _lift(mae_base: float, mae_model: float) -> float:
    if mae_base is None or np.isnan(mae_base) or mae_base == 0 or mae_model is None or np.isnan(mae_model):
        return np.nan
    return float((mae_base - mae_model) / mae_base)

def _save_fig(path: Path) -> None:
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


# --------------------------- Core computations ---------------------------

def compute_global_and_daily(df: pd.DataFrame, last_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Global
    m_model = _metrics(df["y_true"], df["y_pred"])
    m_base = _metrics(df["y_true"], df["y_pred_baseline"])
    cov = float(df["y_pred"].notna().mean() * 100.0)
    horizon = df["horizon_min"].dropna().unique()
    horizon = int(horizon[0]) if len(horizon) else None

    global_row = {
        "horizon_min": horizon,
        "coverage_pred_pct": round(cov, 2),
        "mae_model": m_model["mae"], "rmse_model": m_model["rmse"], "me_model": m_model["me"],
        "mae_baseline": m_base["mae"], "rmse_baseline": m_base["rmse"], "me_baseline": m_base["me"],
        "lift_vs_baseline": _lift(m_base["mae"], m_model["mae"]),
        "n_rows": int(len(df)),
        "n_stations": int(df["station_id"].nunique()),
        "ts_min": df["ts"].min().isoformat() if len(df) else None,
        "ts_max": df["ts"].max().isoformat() if len(df) else None,
    }
    global_df = pd.DataFrame([global_row])

    # Daily (local date)
    daily = (df.groupby("date_local")
               .apply(lambda g: pd.Series({
                   "mae_model": _metrics(g["y_true"], g["y_pred"])["mae"],
                   "mae_baseline": _metrics(g["y_true"], g["y_pred_baseline"])["mae"],
                   "rmse_model": _metrics(g["y_true"], g["y_pred"])["rmse"],
                   "rmse_baseline": _metrics(g["y_true"], g["y_pred_baseline"])["rmse"],
                   "coverage_pred_pct": float(g["y_pred"].notna().mean() * 100.0),
                   "n": int(len(g)),
               })))
    daily["lift_vs_baseline"] = daily.apply(
        lambda r: _lift(r["mae_baseline"], r["mae_model"]), axis=1
    )
    daily = daily.reset_index(names="date")
    # Keep last N days if requested
    if last_days and last_days > 0:
        daily = daily.sort_values("date").tail(last_days)

    return global_df, daily

def compute_by_segments(df: pd.DataFrame, clusters_csv: Optional[Path] = None) -> dict[str, pd.DataFrame]:
    by = {}

    # by station
    grp = df.groupby("station_id")
    by_station = grp.apply(lambda g: pd.Series({
        "mae_model": _metrics(g["y_true"], g["y_pred"])["mae"],
        "mae_baseline": _metrics(g["y_true"], g["y_pred_baseline"])["mae"],
        "rmse_model": _metrics(g["y_true"], g["y_pred"])["rmse"],
        "rmse_baseline": _metrics(g["y_true"], g["y_pred_baseline"])["rmse"],
        "coverage_pred_pct": float(g["y_pred"].notna().mean() * 100.0),
        "n": int(len(g)),
    })).reset_index()
    by_station["lift_vs_baseline"] = by_station.apply(
        lambda r: _lift(r["mae_baseline"], r["mae_model"]), axis=1
    )
    by["station"] = by_station

    # by hour of day
    by_hour = (df.groupby("hour")
                 .apply(lambda g: pd.Series({
                     "mae_model": _metrics(g["y_true"], g["y_pred"])["mae"],
                     "mae_baseline": _metrics(g["y_true"], g["y_pred_baseline"])["mae"],
                     "coverage_pred_pct": float(g["y_pred"].notna().mean() * 100.0),
                     "n": int(len(g)),
                 })).reset_index())
    by_hour["lift_vs_baseline"] = by_hour.apply(lambda r: _lift(r["mae_baseline"], r["mae_model"]), axis=1)
    by["hour"] = by_hour

    # by day-of-week
    by_dow = (df.groupby("dow")
                .apply(lambda g: pd.Series({
                    "mae_model": _metrics(g["y_true"], g["y_pred"])["mae"],
                    "mae_baseline": _metrics(g["y_true"], g["y_pred_baseline"])["mae"],
                    "coverage_pred_pct": float(g["y_pred"].notna().mean() * 100.0),
                    "n": int(len(g)),
                })).reset_index())
    by_dow["lift_vs_baseline"] = by_dow.apply(lambda r: _lift(r["mae_baseline"], r["mae_model"]), axis=1)
    by["dow"] = by_dow

    # by cluster (optional)
    if clusters_csv and clusters_csv.exists():
        clusters = pd.read_csv(clusters_csv, dtype={"station_id": str})
        clusters = clusters[["station_id", "cluster"]].drop_duplicates()
        tmp = df.merge(clusters, on="station_id", how="left")
        if tmp["cluster"].notna().any():
            by_cluster = (tmp.dropna(subset=["cluster"])
                            .groupby("cluster")
                            .apply(lambda g: pd.Series({
                                "mae_model": _metrics(g["y_true"], g["y_pred"])["mae"],
                                "mae_baseline": _metrics(g["y_true"], g["y_pred_baseline"])["mae"],
                                "coverage_pred_pct": float(g["y_pred"].notna().mean() * 100.0),
                                "n": int(len(g)),
                            })).reset_index())
            by_cluster["lift_vs_baseline"] = by_cluster.apply(
                lambda r: _lift(r["mae_baseline"], r["mae_model"]), axis=1
            )
            by["cluster"] = by_cluster

    return by


# --------------------------- Figures ---------------------------

def plot_daily_lift(daily: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(9, 3.6))
    plt.plot(daily["date"].astype(str), daily["lift_vs_baseline"].values, marker="o")
    plt.axhline(0.0, linewidth=1)
    plt.xticks(rotation=0)
    plt.ylabel("Lift vs baseline")
    plt.title("Lift quotidien (positif = mieux que persistance)")
    _save_fig(out_png)

def plot_residual_hist(df: pd.DataFrame, out_png: Path) -> None:
    # résidus sur lignes prédites uniquement
    mask = df["y_pred"].notna()
    if mask.any():
        err = (df.loc[mask, "y_true"] - df.loc[mask, "y_pred"]).astype(float)
    else:
        err = pd.Series([], dtype=float)
    plt.figure(figsize=(7, 4))
    plt.hist(err.values, bins=40)
    plt.title("Distribution des résidus (y_true − y_pred)")
    plt.xlabel("Erreur")
    plt.ylabel("Fréquence")
    _save_fig(out_png)

def plot_mae_by_hour(by_hour: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(by_hour["hour"], by_hour["mae_baseline"], marker="o", label="Baseline")
    plt.plot(by_hour["hour"], by_hour["mae_model"], marker="o", label="Modèle")
    plt.title("MAE par heure (local)")
    plt.xlabel("Heure")
    plt.ylabel("MAE")
    plt.legend(loc="best")
    _save_fig(out_png)

def plot_obs_vs_pred_examples(df: pd.DataFrame, stations: list[str], hours: int, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    if "ts_target" not in df.columns:
        raise ValueError("ts_target is missing. Compute it before plotting.")

    # Fenêtre sur l'axe cible (T+h) → supprime le décalage visuel
    tmax = df["ts_target"].max()
    tmin = tmax - pd.Timedelta(hours=hours)
    win = df[(df["ts_target"] > tmin) & (df["ts_target"] <= tmax)].copy()

    for sid in stations:
        sub = win[win["station_id"].astype(str) == str(sid)].sort_values("ts_target")
        if sub.empty:
            continue

        plt.figure(figsize=(10, 3.8))
        x = sub["ts_target"]

        # Observé (y_true à T+h)
        if "y_true" in sub.columns:
            plt.plot(x, sub["y_true"], linewidth=2, label="Observé")

        # Modèle (si présent)
        if "y_pred" in sub.columns and sub["y_pred"].notna().any():
            plt.plot(x, sub["y_pred"], linewidth=1.5, label="Modèle")

        # Baseline (persistance à T, ré-indexée sur l'axe cible pour la comparaison visuelle)
        if "y_pred_baseline" in sub.columns:
            plt.plot(x, sub["y_pred_baseline"], linewidth=1.0, label="Baseline")

        plt.title(f"Station {sid} — {hours}h")
        plt.xlabel("Temps")
        plt.ylabel("Vélos")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_dir / f"station_{sid}_obs_pred.png", dpi=150)
        plt.close()



# --------------------------- Main ---------------------------

def main(perf_path: Path, last_days: int, horizon: Optional[int], tz: Optional[str]) -> None:
    _mkdirs()

    # 1) Charger & préparer
    df = _read_perf(perf_path)
    df = _localize(df, tz=tz)
    # Horodatage cible : où vivent réellement y_true et les prédictions (T + horizon)
    # Aligner l'affichage sur la cible (T+h)
    if horizon is None:
        if "horizon_min" in df.columns and df["horizon_min"].notna().any():
            h = int(df["horizon_min"].dropna().iloc[0])
        else:
            h = 60
    else:
        h = int(horizon)
    df["ts_target"] = df["ts"] + pd.to_timedelta(h, unit="m")

    # 2) Global + quotidien
    global_df, daily = compute_global_and_daily(df, last_days=last_days)
    global_df.to_csv(TABLES_DIR / "global_metrics.csv", index=False)
    daily.to_csv(TABLES_DIR / "daily_error.csv", index=False)

    # 3) Segments
    clusters_csv = ASSETS / "tables" / "network" / "stations" / "station_clusters.csv"
    segs = compute_by_segments(df, clusters_csv=clusters_csv if clusters_csv.exists() else None)
    segs["station"].to_csv(TABLES_DIR / "error_by_station.csv", index=False)
    segs["hour"].to_csv(TABLES_DIR / "error_by_hour.csv", index=False)
    segs["dow"].to_csv(TABLES_DIR / "error_by_dow.csv", index=False)
    if "cluster" in segs:
        segs["cluster"].to_csv(TABLES_DIR / "error_by_cluster.csv", index=False)

    # 4) Coverage table
    coverage = pd.DataFrame({
        "coverage_pred_pct": [float(df["y_pred"].notna().mean() * 100.0)],
        "rows": [int(len(df))],
        "stations": [int(df["station_id"].nunique())]
    })
    coverage.to_csv(TABLES_DIR / "coverage.csv", index=False)

    # 5) Figures
    if not daily.empty:
        plot_daily_lift(daily, FIGS_DIR / "lift_daily.png")
    plot_residual_hist(df, FIGS_DIR / "residuals_hist.png")
    if "hour" in segs:
        plot_mae_by_hour(segs["hour"], FIGS_DIR / "mae_by_hour.png")

    # 6) Exemples obs vs préd (dernieres 48h) — choisir stations représentatives
    #    - top 4 par volume de données (ou par écart-type sur y_true)
    by_station = segs["station"].copy()
    # si y_true présent pour variance
    try:
        var_by_station = df.groupby("station_id")["y_true"].std().rename("std_true")
        by_station = by_station.merge(var_by_station, on="station_id", how="left")
        top = by_station.sort_values(["std_true", "n"], ascending=[False, False]).head(4)["station_id"].tolist()
    except Exception:
        top = by_station.sort_values("n", ascending=False).head(4)["station_id"].tolist()
    plot_obs_vs_pred_examples(df, top, hours=48, out_dir=FIGS_DIR)

    print("[model/performance] Done.")
    print(f"[model/performance] Global metrics → {TABLES_DIR / 'global_metrics.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build 'Model / Performance & baseline' assets from perf.parquet")
    ap.add_argument("--perf", type=Path, required=True, help="Path to docs/exports/perf.parquet")
    ap.add_argument("--last-days", type=int, default=14, help="Fenêtre pour séries quotidiennes & exemples")
    ap.add_argument("--horizon", type=int, default=None, help="Horizon minutes (indicatif, non filtrant)")
    ap.add_argument("--tz", type=str, default=None, help="Fuseau d'affichage pour groupes heure/jour")
    args = ap.parse_args()

    main(perf_path=args.perf, last_days=args.last_days, horizon=args.horizon, tz=args.tz)
