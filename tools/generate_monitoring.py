# tools/generate_monitoring.py
from __future__ import annotations
import json, math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FIGS = DOCS / "assets" / "figs"
EXPORTS = DOCS / "exports"
MODEL_PATH = ROOT / "models" / "lgb_nbvelos_T+60min.joblib"

FIGS.mkdir(parents=True, exist_ok=True)
EXPORTS.mkdir(parents=True, exist_ok=True)

PARQUET = EXPORTS / "velib.parquet"

# ---------------- Utilities ----------------
def _save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=144)
    plt.close()

def _series_plot(ts: pd.Series, title: str, path: Path) -> None:
    plt.figure(figsize=(9, 3.5))
    ts.plot()
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel("value")
    _save_fig(path)

def _fmt(x: float, nd=2) -> str:
    if isinstance(x, (int,)) or (isinstance(x, float) and np.isfinite(x)):
        return f"{x:.{nd}f}"
    return "n/a"

def _badge(label: str, value: str) -> str:
    return f'<span class="metric-badge">{label}: {value}</span>'

# ---------------- Load parquet ----------------
if not PARQUET.exists():
    raise SystemExit(f"Missing {PARQUET}; run aggregation first.")

df = pd.read_parquet(PARQUET)
if df.empty:
    raise SystemExit("Parquet is empty.")

# Timestamps
df["tbin"] = pd.to_datetime(df["tbin_utc"], utc=True).dt.tz_localize(None)
df["hour_utc"] = pd.to_datetime(df["hour_utc"], utc=True).dt.tz_localize(None)

# Try reconstruct y_true if absent: shift -4 bins on nb_velos_bin per station
if "y_true" not in df.columns:
    df = df.sort_values(["stationcode", "tbin"])
    df["y_true"] = df.groupby("stationcode")["nb_velos_bin"].shift(-4)

# y_pred may be missing if not yet online; fallback: copy previous hour value (naive)
if "y_pred" not in df.columns:
    df["y_pred"] = df.groupby("stationcode")["nb_velos_bin"].shift(4)

# Basic clean
df = df.dropna(subset=["y_true", "y_pred"])
if df.empty:
    raise SystemExit("No aligned y_true/y_pred available to compute metrics.")

# ---------------- Aggregate metrics over time ----------------
agg = df.groupby("tbin").apply(lambda g: pd.Series({
    "MAE": (g["y_true"] - g["y_pred"]).abs().mean(),
    "RMSE": math.sqrt(((g["y_true"] - g["y_pred"])**2).mean()),
    "RelErr": ((g["y_true"] - g["y_pred"]).abs() / g["y_true"].clip(lower=1)).mean()
})).sort_index()

last_24h = agg.last("24H")
last_7d = agg.last("7D")

metrics = {
    "mae_24h": float(last_24h["MAE"].mean()) if not last_24h.empty else float("nan"),
    "rmse_24h": float(last_24h["RMSE"].mean()) if not last_24h.empty else float("nan"),
    "relerr_24h": float(last_24h["RelErr"].mean()) if not last_24h.empty else float("nan"),
    "mae_7d": float(last_7d["MAE"].mean()) if not last_7d.empty else float("nan"),
    "rmse_7d": float(last_7d["RMSE"].mean()) if not last_7d.empty else float("nan"),
}

# ---------------- Plots time series ----------------
_series_plot(agg["MAE"], "MAE over time (15-min bins)", FIGS / "mae_full.png")
_series_plot(agg["RMSE"], "RMSE over time (15-min bins)", FIGS / "rmse_full.png")
if not last_24h.empty:
    _series_plot(last_24h["MAE"], "MAE — last 24h", FIGS / "mae_24h.png")
    _series_plot(last_24h["RMSE"], "RMSE — last 24h", FIGS / "rmse_24h.png")
if not last_7d.empty:
    _series_plot(last_7d["MAE"], "MAE — last 7d", FIGS / "mae_7d.png")
    _series_plot(last_7d["RMSE"], "RMSE — last 7d", FIGS / "rmse_7d.png")

# ---------------- Top errors by station (24h) ----------------
recent = df[df["tbin"] >= (df["tbin"].max() - pd.Timedelta(days=1))]
if not recent.empty:
    station_err = (recent.assign(err=(recent["y_true"] - recent["y_pred"]).abs())
                          .groupby("stationcode")["err"].mean()
                          .sort_values(ascending=False).head(10))
    plt.figure(figsize=(9, 3.5))
    station_err.plot(kind="bar")
    plt.title("Top 10 stations by MAE (last 24h)")
    plt.xlabel("stationcode")
    plt.ylabel("MAE")
    _save_fig(FIGS / "top10_err_24h.png")

# ---------------- Spatial error scatter (24h) ----------------
if not recent.empty and {"lat","lon"}.issubset(df.columns):
    geo = (recent.assign(err=(recent["y_true"] - recent["y_pred"]).abs())
                 .groupby(["stationcode","lat","lon"])["err"].mean().reset_index())
    plt.figure(figsize=(6, 6))
    plt.scatter(geo["lon"], geo["lat"], s=20, alpha=0.6, c=geo["err"])
    plt.title("Spatial error scatter — mean MAE by station (24h)")
    plt.xlabel("lon"); plt.ylabel("lat")
    _save_fig(FIGS / "error_scatter.png")

# ---------------- Drift: PSI (7d vs history) ----------------
psi_summary = {}
cand = [c for c in ["occ_ratio_bin","temp_C","wind_mps","precip_mm"] if c in df.columns]
if cand:
    cutoff = df["tbin"].max() - pd.Timedelta(days=7)
    ref = df[df["tbin"] < cutoff]
    cur = df[df["tbin"] >= cutoff]
    def psi(x_ref, x_cur, bins=10):
        if len(x_ref.dropna()) == 0 or len(x_cur.dropna()) == 0:
            return np.nan
        q = np.linspace(0, 1, bins + 1)
        edges = np.quantile(x_ref.dropna(), q)
        ref_hist, _ = np.histogram(x_ref.dropna(), bins=edges)
        cur_hist, _ = np.histogram(x_cur.dropna(), bins=edges)
        ref_p = np.clip(ref_hist / (ref_hist.sum() or 1), 1e-6, 1.0)
        cur_p = np.clip(cur_hist / (cur_hist.sum() or 1), 1e-6, 1.0)
        return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))
    for f in cand:
        psi_summary[f] = psi(ref[f], cur[f])
    plt.figure(figsize=(9, 3.5))
    pd.Series(psi_summary).sort_values(ascending=False).plot(kind="bar")
    plt.title("Feature PSI (last 7d vs history)")
    plt.xlabel("feature"); plt.ylabel("PSI")
    _save_fig(FIGS / "psi_summary.png")

# Cible distribution
plt.figure(figsize=(9, 3.5))
df["y_true"].plot(kind="hist", bins=30)
plt.title("y_true distribution (all)"); plt.xlabel("bikes (T+1h)"); plt.ylabel("count")
_save_fig(FIGS / "ytrue_hist.png")

# Occupancy distribution ref vs cur
if "occ_ratio_bin" in df.columns:
    cutoff = df["tbin"].max() - pd.Timedelta(days=7)
    ref = df[df["tbin"] < cutoff]["occ_ratio_bin"].dropna()
    cur = df[df["tbin"] >= cutoff]["occ_ratio_bin"].dropna()
    plt.figure(figsize=(9, 3.5))
    plt.hist(ref, bins=30, alpha=0.6, label="history")
    plt.hist(cur, bins=30, alpha=0.6, label="last 7d")
    plt.title("Occupancy ratio distribution — history vs last 7d")
    plt.xlabel("occ_ratio"); plt.ylabel("count")
    plt.legend()
    _save_fig(FIGS / "occ_hist_ref_cur.png")

# ---------------- Model artefact: importances & corr ----------------
train_summary = {"mae": None, "rmse": None, "n_valid": None}
if MODEL_PATH.exists():
    import joblib
    bundle = joblib.load(MODEL_PATH)
    model = bundle.get("model")
    feat_cols = bundle.get("feat_cols") or []
    # Offline scores saved by train()?
    for k in ("mae","rmse","n_valid"):
        if k in bundle:
            train_summary[k] = bundle[k]
    # Feature importances
    try:
        fi = pd.Series(model.feature_importances_, index=feat_cols)
        fi = fi.sort_values(ascending=False).head(30)  # top 30 for readability
        plt.figure(figsize=(9, 6))
        fi.plot(kind="barh")
        plt.gca().invert_yaxis()
        plt.title("LightGBM — Feature importances (top 30)")
        plt.xlabel("importance (gain or split)")
        _save_fig(FIGS / "feature_importances.png")
    except Exception:
        pass

# Correlation matrix on numeric features
num_cols = []
for c in ["nb_velos_bin","nb_bornes_bin","capacity_bin","occ_ratio_bin",
          "temp_C","precip_mm","wind_mps"]:
    if c in df.columns:
        num_cols.append(c)
# add a few lags if available
for c in [f"lag_nb_{b}b" for b in (1,2,4,8,16)]:
    if c in df.columns:
        num_cols.append(c)
corr = None
if num_cols:
    sample = (df[num_cols]
              .apply(pd.to_numeric, errors="coerce")
              .replace([np.inf,-np.inf], np.nan)
              .dropna())
    if not sample.empty:
        corr = sample.corr()
        plt.figure(figsize=(6, 6))
        plt.imshow(corr, interpolation="nearest")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.index)), corr.index)
        plt.title("Correlation matrix (numeric features)")
        plt.colorbar()
        _save_fig(FIGS / "corr_matrix.png")

# ---------------- Write metrics.json ----------------
with open(EXPORTS / "metrics.json", "w", encoding="utf-8") as f:
    json.dump({"metrics": metrics, "psi": psi_summary, "train": train_summary}, f, ensure_ascii=False, indent=2)

# ---------------- Write Markdown pages ----------------
# Monitoring overview
idx = DOCS / "monitoring" / "index.md"
idx.parent.mkdir(parents=True, exist_ok=True)
idx.write_text(f"""# Monitoring — Vue d’ensemble

{_badge("MAE 24h", _fmt(metrics["mae_24h"]))}
{_badge("RMSE 24h", _fmt(metrics["rmse_24h"]))}
{_badge("MAE 7j", _fmt(metrics["mae_7d"]))}
{_badge("RMSE 7j", _fmt(metrics["rmse_7d"]))}
{_badge("Rel. Err 24h", _fmt(metrics["relerr_24h"]))}

<div class="figure-grid">
  <img src="../assets/figs/mae_24h.png" alt="MAE 24h">
  <img src="../assets/figs/rmse_24h.png" alt="RMSE 24h">
  <img src="../assets/figs/mae_7d.png" alt="MAE 7j">
  <img src="../assets/figs/rmse_7d.png" alt="RMSE 7j">
</div>

> Generated by `tools/generate_monitoring.py` from `docs/exports/velib.parquet`.
""", encoding="utf-8")

# Metrics detail
metrics_md = DOCS / "monitoring" / "metrics.md"
metrics_md.write_text(f"""# Métriques détaillées

- **MAE 24h**: {_fmt(metrics["mae_24h"])}  
- **RMSE 24h**: {_fmt(metrics["rmse_24h"])}  
- **MAE 7j**: {_fmt(metrics["mae_7d"])}  
- **RMSE 7j**: {_fmt(metrics["rmse_7d"])}  

## Séries temporelles
![MAE (full)](../assets/figs/mae_full.png)
![RMSE (full)](../assets/figs/rmse_full.png)

## Top erreurs (24h)
![Top10 stations (24h)](../assets/figs/top10_err_24h.png)

## Erreur spatiale (scatter 24h)
![Erreur spatiale](../assets/figs/error_scatter.png)

## Distributions occupation
![Occ ratio hist (ref vs cur)](../assets/figs/occ_hist_ref_cur.png)
""", encoding="utf-8")

# Drift
drift = DOCS / "monitoring" / "data_drift.md"
psi_md = "\n".join([f"- **{k}**: {_fmt(v)}" for k, v in psi_summary.items()]) if psi_summary else "_No PSI computed._"
drift.write_text(f"""# Drift des données

## PSI (7j vs historique)
{psi_md}

![PSI](../assets/figs/psi_summary.png)

## Distribution cible
![y_true distribution](../assets/figs/ytrue_hist.png)
""", encoding="utf-8")

# Model page
model_md = DOCS / "monitoring" / "model.md"
mae_v = _fmt((train_summary.get("mae") or float("nan")))
rmse_v = _fmt((train_summary.get("rmse") or float("nan")))
n_valid = int(train_summary.get("n_valid") or 0)
model_md.write_text(f"""# Modèle — Importances & Score Offline

- **Dernier entraînement**  
  {_badge("MAE valid", mae_v)} {_badge("RMSE valid", rmse_v)} {_badge("n_valid", str(n_valid))}

## Importances (Gain)
![Feature importances](../assets/figs/feature_importances.png)

## Corrélations (features num.)
![Correlation matrix](../assets/figs/corr_matrix.png)
""", encoding="utf-8")

print("[OK] Monitoring pages & figures updated.")
