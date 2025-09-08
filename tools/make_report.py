# tools/make_report.py
# Génère docs/results.md + figures de qualité pour MkDocs (Material)
import pathlib, io, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Seuils rupture
try:
    from src.config import RUPTURE_LOW, RUPTURE_HIGH
except Exception:
    RUPTURE_LOW, RUPTURE_HIGH = 0.20, 0.80

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXP  = ROOT / "exports"
DEXP = ROOT / "docs" / "exports"
ASSETS = ROOT / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

def _load_pair(stem: str) -> pd.DataFrame:
    for p in [EXP/f"{stem}.parquet", EXP/f"{stem}.csv", DEXP/f"{stem}.parquet", DEXP/f"{stem}.csv"]:
        if p.exists():
            try:
                df = pd.read_parquet(p) if p.suffix==".parquet" else pd.read_csv(p)
                return df
            except Exception:
                pass
    return pd.DataFrame()

def _to_dt(s): return pd.to_datetime(s, errors="coerce")

def _nice_fig(w=10, h=4):
    fig, ax = plt.subplots(figsize=(w, h), dpi=160)
    return fig, ax

def _save(fig, name):
    p = ASSETS / name
    fig.tight_layout()
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return f"assets/{name}"

# --- Load data
hour = _load_pair("velib_hourly")
fc   = _load_pair("velib_forecast_24h")

for d in (hour, fc):
    if not d.empty:
        d["hour_utc"] = _to_dt(d["hour_utc"])
        d["stationcode"] = d["stationcode"].astype(str)

if not hour.empty:
    hour["capacity_est"] = (hour.get("bikes_avg",0).fillna(0) + hour.get("docks_avg",0).fillna(0)).replace(0, np.nan)

# --- KPIs
n_snap = len(hour) if not hour.empty else 0
n_st   = hour["stationcode"].nunique() if not hour.empty else 0
hmin   = hour["hour_utc"].min() if not hour.empty else None
hmax   = hour["hour_utc"].max() if not hour.empty else None

# ---------- HERO: historique + forecast + zones seuils ----------
hero_path = None
if not hour.empty and not fc.empty:
    last = hmax
    lookback = last - pd.Timedelta("18h")
    hist = hour[(hour["hour_utc"] >= lookback) & (hour["hour_utc"] <= last)].copy()
    fut  = fc[(fc["hour_utc"] > last) & (fc["hour_utc"] <= last + pd.Timedelta("6h"))].copy()

    # Retenir une station “lisible”: forte variance récente ou top risque
    # 1) variance
    stds = (hist.groupby("stationcode")["occ_ratio_hour"].std().sort_values(ascending=False))
    sc1 = stds.index[0] if len(stds) else None
    # 2) risque max T+3h
    def risk_row(x):
        # score simple 0..1 centré sur 0.20 / 0.80
        low  = np.clip((RUPTURE_LOW - x)/RUPTURE_LOW, 0, 1)
        high = np.clip((x - RUPTURE_HIGH)/(1-RUPTURE_HIGH), 0, 1)
        return np.maximum(low, high)
    w3 = fut[fut["hour_utc"] <= last + pd.Timedelta("3h")].copy()
    if not w3.empty and "pred_occ" in w3.columns:
        w3["risk"] = risk_row(w3["pred_occ"].astype(float))
        sc2 = (w3.groupby("stationcode")["risk"].max().sort_values(ascending=False).index[0])
    else:
        sc2 = None
    sc = sc2 or sc1 or (hour["stationcode"].iloc[0] if not hour.empty else "0000")

    # séries pour la station choisie
    hs = hist[hist["stationcode"]==sc].sort_values("hour_utc")
    fs = fut[fut["stationcode"]==sc].sort_values("hour_utc")
    name = hs["name"].dropna().iloc[-1] if "name" in hs and len(hs) else sc

    fig, ax = _nice_fig(10, 4.2)
    ax.axhspan(0, RUPTURE_LOW,  color="#ff4d4f", alpha=0.08, label=f"< {RUPTURE_LOW:.0%}")
    ax.axhspan(RUPTURE_HIGH, 1, color="#ff4d4f", alpha=0.08, label=f"> {RUPTURE_HIGH:.0%}")
    ax.axvspan(last, hs["hour_utc"].max() + pd.Timedelta("6h"), color="#ddd", alpha=0.12, label="Prévision")
    ax.plot(hs["hour_utc"], hs["occ_ratio_hour"], lw=2, label="Historique")
    if "pred_occ" in fs.columns and len(fs):
        ax.plot(fs["hour_utc"], fs["pred_occ"], lw=2, ls="--", label="Prévision")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Occupation (0–1)")
    ax.set_xlabel("Heure (locale)")
    ax.set_title(f"Occupation — {name}")
    ax.legend(loc="upper left", frameon=False)
    hero_path = _save(fig, "hero_occ.png")

# ---------- TOP risques T+3h ----------
top_risk_path = None
if not fc.empty and not hour.empty and "pred_occ" in fc.columns:
    last = hmax
    w3 = fc[(fc["hour_utc"] > last) & (fc["hour_utc"] <= last + pd.Timedelta("3h"))].copy()
    if not w3.empty:
        w3["risk"] = w3["pred_occ"].astype(float).pipe(lambda x: np.maximum(
            np.clip((RUPTURE_LOW - x)/RUPTURE_LOW, 0, 1),
            np.clip((x - RUPTURE_HIGH)/(1-RUPTURE_HIGH), 0, 1)
        ))
        g = (w3.groupby(["stationcode"], as_index=False)
               .agg(risk=("risk","max"))
               .sort_values("risk", ascending=False)
               .head(10))
        g = g.merge(hour[["stationcode","name"]].drop_duplicates("stationcode"),
                    on="stationcode", how="left")
        fig, ax = _nice_fig(8, 4.2)
        ax.barh(g["name"].fillna(g["stationcode"]).iloc[::-1], g["risk"].iloc[::-1], height=0.55)
        ax.set_xlim(0, 1); ax.set_xlabel("Risque (0–1)"); ax.set_title("Top risques — T+3 h")
        top_risk_path = _save(fig, "top_risk.png")

# ---------- TOP volatilité (48h) ----------
top_vol_path = None
if not hour.empty:
    last = hmax
    h48 = hour[hour["hour_utc"] >= last - pd.Timedelta("48h")]
    g = (h48.groupby("stationcode")["occ_ratio_hour"].std()
           .sort_values(ascending=False).head(10).reset_index(name="std_occ"))
    g = g.merge(hour[["stationcode","name"]].drop_duplicates("stationcode"),
                on="stationcode", how="left")
    fig, ax = _nice_fig(8, 4.2)
    ax.barh(g["name"].fillna(g["stationcode"]).iloc[::-1], g["std_occ"].iloc[::-1], height=0.55)
    ax.set_xlabel("Écart-type (48 h)"); ax.set_title("Top volatilité — 48 h")
    top_vol_path = _save(fig, "top_vol.png")

# ---------- Corrélation simple (occ vs température) ----------
corr_path = None
if not hour.empty and {"temp_C"}.issubset(hour.columns):
    sub = hour.dropna(subset=["occ_ratio_hour","temp_C"]).sample(min(5000, len(hour)), random_state=42)
    if len(sub):
        fig, ax = _nice_fig(7.5, 4.5)
        ax.scatter(sub["temp_C"], sub["occ_ratio_hour"], s=8, alpha=0.25)
        # tendance linéaire
        try:
            z = np.polyfit(sub["temp_C"], sub["occ_ratio_hour"], 1)
            xx = np.linspace(sub["temp_C"].min(), sub["temp_C"].max(), 100)
            ax.plot(xx, z[0]*xx + z[1], lw=2, alpha=0.8)
        except Exception:
            pass
        ax.set_xlabel("Température (°C)")
        ax.set_ylabel("Occupation (0–1)")
        ax.set_title("Corrélation simple (échantillon)")
        corr_path = _save(fig, "corr_occ_temp.png")

# ---------- Résumé Markdown (peu de texte, visuel) ----------
md = io.StringIO()
print(
f"""---
title: Results
hide:
  - toc
---

# Résultats

**Snapshots** : {n_snap:,} &nbsp;•&nbsp; **Stations** : {n_st} &nbsp;•&nbsp; **Couverture** : {hmin} → {hmax}  
*(Heure affichée : Europe/Paris)*

""", file=md)

if hero_path:
    print(f"![Historique + prévision](/{hero_path})\n", file=md)

if top_risk_path:
    print("## À traiter en priorité (T+3h)\n", file=md)
    print(f"![Top risques](/{top_risk_path})\n", file=md)

if top_vol_path:
    print("## Stations les plus volatiles (48 h)\n", file=md)
    print(f"![Top volatilité](/{top_vol_path})\n", file=md)

if corr_path:
    print("## Corrélation simple\n", file=md)
    print(f"![Occ vs Temp](/{corr_path})\n", file=md)

# Liens d’export minimalistes
exports = []
for stem, label in [("velib_hourly","Hourly"), ("velib_forecast_24h","Forecast 24h")]:
    for ext in ["parquet","csv"]:
        rel = f"exports/{stem}.{ext}"
        if (DEXP/ f"{stem}.{ext}").exists():
            exports.append(f"[{label} ({ext})](/exports/{stem}.{ext})")

if exports:
    print("## Exports\n", file=md)
    print(" • ".join(exports), file=md)

( ROOT / "docs" / "results.md").write_text(md.getvalue(), encoding="utf-8")
print("OK — docs/results.md mis à jour.")
