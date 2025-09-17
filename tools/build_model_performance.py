# tools/build_model_performance.py
# Page builder — "Modèle / Performance & baseline"
#
# Mesure la qualité des prévisions vs baseline (persistance) et produit :
# - Tables: global_metrics, daily_error, error_by_station/hour/dow/(cluster si dispo), coverage
# - Figures: lift quotidien (sparkline), histogramme des résidus, MAE par heure, obs vs préd (échantillon)
# - Markdown: docs/model/performance.md (structure éditoriale + embeds de figures & liens CSV)
#
# CLI :
#   python tools/build_model_performance.py \
#       --perf docs/exports/perf.parquet --last-days 14 --horizon 60 --tz Europe/Paris
#
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------- Paths ---------------------------

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
FIGS_DIR = ASSETS / "figs" / "model" / "performance"
TABLES_DIR = ASSETS / "tables" / "model" / "performance"
OUT_MD = DOCS / "model" / "performance.md"

# mkdirs
for d in (FIGS_DIR, TABLES_DIR, OUT_MD.parent):
    d.mkdir(parents=True, exist_ok=True)


# --------------------------- Utils ---------------------------

def rel_from_md(md_path: Path, target: Path) -> str:
    """
    Chemin relatif (POSIX) depuis md_path vers target, compatible MkDocs
    (use_directory_urls: true). Ex. docs/model/performance.md -> ../../assets/...
    """
    # position du .md dans /docs
    md_rel = Path(md_path).resolve().relative_to(DOCS.resolve())
    # retirer le suffixe .md, compter la profondeur (sauf si index)
    parts = md_rel.with_suffix("").parts
    depth = len(parts) if parts[-1] != "index" else len(parts) - 1
    prefix = "../" * max(depth, 0)
    rel_from_docs = Path(target).resolve().relative_to(DOCS.resolve()).as_posix()
    return (prefix + rel_from_docs).replace("//", "/")

def _read_perf(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"[performance] Introuvable: {path}")
    df = pd.read_parquet(path)

    # ts (UTC naïf, arrondi 15 min)
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
        if "nb_velos_bin" in df.columns:
            df["y_true"] = pd.to_numeric(df["nb_velos_bin"], errors="coerce")
        else:
            raise KeyError("[performance] Colonne 'y_true' manquante")
    else:
        df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")

    if "y_pred_baseline" not in df.columns:
        # fallback neutre si absent
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

    # colonnes additionnelles si déjà présentes (depuis datasets.py)
    if "ts_target" in df.columns:
        df["ts_target"] = pd.to_datetime(df["ts_target"], errors="coerce")
    if "ts_decision" in df.columns:
        df["ts_decision"] = pd.to_datetime(df["ts_decision"], errors="coerce")

    return df[["ts", "station_id", "y_true", "y_pred", "y_pred_baseline", "horizon_min",
               *([c for c in ["ts_target", "ts_decision"] if c in df.columns])]].copy()

def _localize(df: pd.DataFrame, tz: Optional[str]) -> pd.DataFrame:
    # Localisation sur l'axe décision "ts" pour groupages heure/jour
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

def _fmt_dt(ts: pd.Timestamp, tz: Optional[str]) -> str:
    if pd.isna(ts):
        return "—"
    t = pd.Timestamp(ts)
    if tz:
        t = t.tz_localize("UTC").tz_convert(tz)
    else:
        t = t.tz_localize("UTC")
    # ISO local court
    return t.strftime("%Y-%m-%d %H:%M %Z")

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

    # by hour of day (local)
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

def plot_obs_vs_pred_examples(df: pd.DataFrame, stations: list[str], hours: int, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Fenêtre et abscisse sur l'axe cible (T+h) → supprime le décalage visuel
    if "ts_target" not in df.columns:
        raise ValueError("ts_target is missing. Compute it before plotting.")

    tmax = df["ts_target"].max()
    tmin = tmax - pd.Timedelta(hours=hours)
    win = df[(df["ts_target"] > tmin) & (df["ts_target"] <= tmax)].copy()

    out_paths: list[Path] = []
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

        # Baseline (persistance à T, réindexée sur l'axe cible pour comparaison visuelle)
        if "y_pred_baseline" in sub.columns:
            plt.plot(x, sub["y_pred_baseline"], linewidth=1.0, label="Baseline")

        plt.title(f"Station {sid} — {hours}h")
        plt.xlabel("Temps")
        plt.ylabel("Vélos")
        plt.legend(loc="best")
        plt.tight_layout()
        out_file = out_dir / f"station_{sid}_obs_pred.png"
        plt.savefig(out_file, dpi=150)
        plt.close()
        out_paths.append(out_file)
    return out_paths


# --------------------------- Markdown template ---------------------------

MD_TEMPLATE = """# Performance & baseline

## Objectif
Mesurer la **qualité des prévisions** du modèle et la situer **par rapport à une baseline** simple (persistance).

## Questions auxquelles la page répond
- Quelle est l’erreur moyenne **globale** (toutes stations, toutes heures) ?
- Dans quels **segments** (heure, jour, station, cluster, zone) le modèle est-il le plus/moins performant ?
- Quel est le **gain vs baseline** (lift) et comment évolue-t-il dans le temps ?

## Métriques principales
- **MAE** (Mean Absolute Error) — robustesse et lisibilité opérationnelle.
- **RMSE** — pénalise davantage les gros écarts.
- **ME (biais)** — moyenne des erreurs signée (sous/sur-prédiction).
- **Coverage prédictif** — part d’horodatages pour lesquels une prédiction existe.
- **Lift vs baseline** = `(MAE_baseline − MAE_modèle) / MAE_baseline` (positif = mieux que la persistance).
- **R²** (optionnel, sur séries agrégées) — à manier avec prudence pour des données bornées/peu linéaires.

---

## Résumé chiffré (fenêtre)
- **Horizon (min)** : **{horizon_min}**  
- **Couverture prédictive** : **{coverage_pred_pct:.2f}%**  
- **MAE** — modèle : **{mae_model:.3f}** · baseline : **{mae_baseline:.3f}**  
- **RMSE** — modèle : **{rmse_model:.3f}** · baseline : **{rmse_baseline:.3f}**  
- **Biais (ME)** — modèle : **{me_model:.3f}** · baseline : **{me_baseline:.3f}**  
- **Lift vs baseline** : **{lift_pct:.2f}%**  
- **Données** : **{n_rows}** lignes · **{n_stations}** stations · **{ts_min_local} → {ts_max_local}**  

> Les séries temporelles sont agrégées sur l’axe **décision T (local)** pour les découpages heure/jour.  
> Les tracés *observé vs prédit* sont alignés sur **l’axe cible T+h** (colonne `ts_target`) pour éviter tout décalage visuel.

---

## Découpages & comparaisons
- **Par station** (top/bottom-10, distribution), **par cluster** (archétypes d’usage), **par heure du jour**, **semaine/week-end**, **par arrondissement/quartier**.
- **Chronologique** : courbe MAE quotidienne/hebdomadaire, détection de dégradations.
- **Capacité** : erreur normalisée par capacité estimée (si disponible) pour comparer des stations hétérogènes.

---

## Visualisations
### Lift quotidien
![Lift quotidien]({lift_daily_rel})

### Distribution des résidus
![Histogramme des résidus]({residuals_hist_rel})

### MAE par heure (local)
![MAE par heure]({mae_by_hour_rel})

### Observé vs prédit — exemples ({examples_count})
{examples_md}

---

## Tables d’appui
- **Global** : `{global_csv_rel}`  
- **Quotidien** : `{daily_csv_rel}`  
- **Par heure** : `{by_hour_csv_rel}` · **Par jour de semaine** : `{by_dow_csv_rel}`  
- **Par station** : `{by_station_csv_rel}`{by_cluster_line}
- **Coverage** : `{coverage_csv_rel}`

---

## Lecture & limites
- La **persistance** (dernier état connu) est une baseline forte à court terme ; le lift est donc une mesure exigeante.
- Les métriques agrégées peuvent masquer des comportements **station-spécifiques** (d’où l’analyse segmentée).

"""

# --------------------------- Main ---------------------------

def main(perf_path: Path, last_days: int, horizon: Optional[int], tz: Optional[str]) -> None:
    # 1) Charger & préparer
    df = _read_perf(perf_path)
    df = _localize(df, tz=tz)

    # Horodatage cible (si absent) = ts + horizon
    if "ts_target" not in df.columns:
        if horizon is None:
            if "horizon_min" in df.columns and df["horizon_min"].notna().any():
                h = int(df["horizon_min"].dropna().iloc[0])
            else:
                h = 60
        else:
            h = int(horizon)
        df["ts_target"] = df["ts"] + pd.to_timedelta(h, unit="m")
    else:
        # tenter d'inférer h si dispo
        h = int(horizon) if horizon is not None else (int(df["horizon_min"].dropna().iloc[0]) if df["horizon_min"].notna().any() else 60)

    # 2) Global + quotidien
    global_df, daily = compute_global_and_daily(df, last_days=last_days)
    global_path = TABLES_DIR / "global_metrics.csv"
    daily_path = TABLES_DIR / "daily_error.csv"
    global_df.to_csv(global_path, index=False)
    daily.to_csv(daily_path, index=False)

    # 3) Segments
    clusters_csv = ASSETS / "tables" / "network" / "stations" / "station_clusters.csv"
    segs = compute_by_segments(df, clusters_csv=clusters_csv if clusters_csv.exists() else None)
    by_station_path = TABLES_DIR / "error_by_station.csv"
    by_hour_path = TABLES_DIR / "error_by_hour.csv"
    by_dow_path = TABLES_DIR / "error_by_dow.csv"
    segs["station"].to_csv(by_station_path, index=False)
    segs["hour"].to_csv(by_hour_path, index=False)
    segs["dow"].to_csv(by_dow_path, index=False)
    by_cluster_path = None
    if "cluster" in segs:
        by_cluster_path = TABLES_DIR / "error_by_cluster.csv"
        segs["cluster"].to_csv(by_cluster_path, index=False)

    # 4) Coverage table
    coverage = pd.DataFrame({
        "coverage_pred_pct": [float(df["y_pred"].notna().mean() * 100.0)],
        "rows": [int(len(df))],
        "stations": [int(df["station_id"].nunique())]
    })
    coverage_path = TABLES_DIR / "coverage.csv"
    coverage.to_csv(coverage_path, index=False)

    # 5) Figures
    lift_daily_path = FIGS_DIR / "lift_daily.png"
    residuals_hist_path = FIGS_DIR / "residuals_hist.png"
    mae_by_hour_path = FIGS_DIR / "mae_by_hour.png"

    if not daily.empty:
        plot_daily_lift(daily, lift_daily_path)
    plot_residual_hist(df, residuals_hist_path)
    if "hour" in segs:
        plot_mae_by_hour(segs["hour"], mae_by_hour_path)

    # 6) Exemples obs vs préd (dernières 48h) — choisir stations représentatives
    by_station = segs["station"].copy()
    try:
        var_by_station = df.groupby("station_id")["y_true"].std().rename("std_true")
        by_station = by_station.merge(var_by_station, on="station_id", how="left")
        top = by_station.sort_values(["std_true", "n"], ascending=[False, False]).head(4)["station_id"].tolist()
    except Exception:
        top = by_station.sort_values("n", ascending=False).head(4)["station_id"].tolist()
    examples_paths = plot_obs_vs_pred_examples(df, top, hours=48, out_dir=FIGS_DIR)

    # --------------------- Rendu Markdown ---------------------
    g = global_df.iloc[0].to_dict()
    lift_pct = float(g["lift_vs_baseline"] * 100.0) if pd.notna(g["lift_vs_baseline"]) else np.nan

    ts_min_local = _fmt_dt(pd.to_datetime(g.get("ts_min")), tz)
    ts_max_local = _fmt_dt(pd.to_datetime(g.get("ts_max")), tz)

    # lignes d'exemple
    examples_md_lines = []
    for p in examples_paths:
        # nom station depuis le fichier
        name = p.stem.replace("_", " ").replace("station ", "Station ")
        examples_md_lines.append(f"![{name}]({rel_from_md(OUT_MD, p)})")
    examples_md = "\n".join(examples_md_lines) if examples_md_lines else "_Pas d’exemple disponible sur la fenêtre._"

    by_cluster_line = ""
    if by_cluster_path is not None:
        by_cluster_line = f"\n- **Par cluster** : `{rel_from_md(OUT_MD, by_cluster_path)}`"

    md = MD_TEMPLATE.format(
        horizon_min=int(g["horizon_min"]) if pd.notna(g["horizon_min"]) else "—",
        coverage_pred_pct=float(g["coverage_pred_pct"]) if pd.notna(g["coverage_pred_pct"]) else float("nan"),
        mae_model=float(g["mae_model"]), rmse_model=float(g["rmse_model"]), me_model=float(g["me_model"]),
        mae_baseline=float(g["mae_baseline"]), rmse_baseline=float(g["rmse_baseline"]), me_baseline=float(g["me_baseline"]),
        lift_pct=lift_pct if pd.notna(lift_pct) else float("nan"),
        n_rows=int(g["n_rows"]), n_stations=int(g["n_stations"]),
        ts_min_local=ts_min_local, ts_max_local=ts_max_local,
        lift_daily_rel=rel_from_md(OUT_MD, lift_daily_path),
        residuals_hist_rel=rel_from_md(OUT_MD, residuals_hist_path),
        mae_by_hour_rel=rel_from_md(OUT_MD, mae_by_hour_path),
        examples_count=len(examples_paths),
        examples_md=examples_md,
        global_csv_rel=rel_from_md(OUT_MD, global_path),
        daily_csv_rel=rel_from_md(OUT_MD, daily_path),
        by_hour_csv_rel=rel_from_md(OUT_MD, by_hour_path),
        by_dow_csv_rel=rel_from_md(OUT_MD, by_dow_path),
        by_station_csv_rel=rel_from_md(OUT_MD, by_station_path),
        by_cluster_line=by_cluster_line,
        coverage_csv_rel=rel_from_md(OUT_MD, coverage_path),
    )

    with open(OUT_MD, "w", encoding="utf-8", newline="\n") as f:
        f.write(md)

    print("[model/performance] Done.")
    print(f"[model/performance] Global metrics → {global_path}")
    print(f"[model/performance] Markdown → {OUT_MD}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build 'Model / Performance & baseline' assets & page from perf.parquet")
    ap.add_argument("--perf", type=Path, required=True, help="Path to docs/exports/perf.parquet")
    ap.add_argument("--last-days", type=int, default=14, help="Fenêtre pour séries quotidiennes & exemples")
    ap.add_argument("--horizon", type=int, default=None, help="Horizon minutes (indicatif, non filtrant)")
    ap.add_argument("--tz", type=str, default=None, help="Fuseau d'affichage pour groupes heure/jour")
    args = ap.parse_args()

    main(perf_path=args.perf, last_days=args.last_days, horizon=args.horizon, tz=args.tz)
