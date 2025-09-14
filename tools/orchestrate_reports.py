# tools/orchestrate_reports.py
from __future__ import annotations

import subprocess, sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FIGS = DOCS / "assets" / "figs"
TABLES = DOCS / "assets" / "tables"
AUTO = DOCS / "exports" / "auto"
SCRIPTS = ROOT / "tools"

def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"[ERROR] {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    else:
        print(proc.stdout.strip())

def read_csv_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists(): return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def frac(x): 
    try: return f"{100*float(x):.1f}%"
    except: return "—"

def datefmt(ts):
    try: return pd.to_datetime(ts).strftime("%d %b %Y")
    except: return str(ts)

def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")

def build_usage(last_days: int = 7):
    print("\n[STEP] Usage …")
    run([sys.executable, str(SCRIPTS / "build_usage.py"), "--events", str(DOCS / "exports" / "events.parquet"), "--last-days", str(last_days)])

def build_performance(last_days: int = 7, horizon: int = 60, station: str | None = None):
    print("\n[STEP] Performance …")
    cmd = [sys.executable, str(SCRIPTS / "build_performance.py"), "--perf", str(DOCS / "exports" / "perf.parquet"),
           "--last-days", str(last_days), "--horizon", str(horizon)]
    if station: cmd += ["--station", station]
    run(cmd)

def build_monitoring(last_days: int = 14, horizon: int = 60):
    print("\n[STEP] Monitoring …")
    run([sys.executable, str(SCRIPTS / "build_monitoring.py"),
         "--events", str(DOCS / "exports" / "events.parquet"),
         "--perf", str(DOCS / "exports" / "perf.parquet"),
         "--last-days", str(last_days), "--horizon", str(horizon)])

def make_usage_insights() -> str:
    kpi = read_csv_safe(TABLES / "kpis_usage.csv")
    daily = read_csv_safe(TABLES / "usage_daily.csv")
    stats = read_csv_safe(TABLES / "station_stats.csv")
    clusters = read_csv_safe(TABLES / "station_clusters.csv")

    # helpers pour récupérer les parts low/high, avec fallback recalculé depuis events.parquet
    def get_share_from_kpi(row, keys):
        if row is None:
            return None
        for k in keys:
            if k in row.index:
                try:
                    return float(row[k])
                except Exception:
                    pass
        return None

    low_keys  = ["share_low_<10%", "share_low_0_10", "share_low_lt_10", "low_share", "share_low"]
    high_keys = ["share_high_>90%", "share_high_90_100", "share_high_gt_90", "high_share", "share_high"]

    r = kpi.iloc[0] if (kpi is not None and not kpi.empty) else None
    share_low = get_share_from_kpi(r, low_keys)
    share_high = get_share_from_kpi(r, high_keys)

    # Fallback : recalcul depuis events.parquet si non trouvés
    if share_low is None or share_high is None:
        try:
            ev_path = DOCS / "exports" / "events.parquet"
            ev = pd.read_parquet(ev_path) if ev_path.exists() else pd.read_csv(DOCS / "exports" / "events.csv")
            cap = pd.to_numeric(ev.get("capacity"), errors="coerce").replace(0, np.nan)
            occ = (pd.to_numeric(ev.get("bikes"), errors="coerce") / cap).clip(0, 1)
            share_low = float((occ < 0.10).mean())
            share_high = float((occ > 0.90).mean())
        except Exception:
            # en dernier recours, laisse None → sera rendu "—"
            pass

    def maybe(x, fmt="{:.2f}"):
        try:
            return fmt.format(float(x))
        except Exception:
            return "—"

    lines = ["# Points clés — Usage du réseau", ""]

    if r is not None:
        lines += [
            f"- **Occupation moyenne** : {maybe(r.get('occ_mean'))} — médiane {maybe(r.get('occ_median'))} (IQR {maybe(r.get('occ_iqr'))}).",
            f"- **Sous-tension (<10%)** : {frac(share_low)} ; **Surtension (>90%)** : {frac(share_high)}.",
        ]

    if daily is not None and not daily.empty:
        tmin, tmax = daily["ts"].min(), daily["ts"].max()
        lines.append(f"- **Fenêtre analysée** : {datefmt(tmin)} → {datefmt(tmax)}.")

    if stats is not None and not stats.empty:
        top_vol = stats.sort_values("std", ascending=False).head(5)["station_id"].astype(str).tolist()
        lines.append(f"- **Stations les plus volatiles** (écart-type d’occupation) : {', '.join(top_vol)}.")

    if clusters is not None and not clusters.empty:
        share = clusters["cluster"].value_counts(normalize=True).sort_index()
        parts = [f"C{int(c)}: {100*s:.1f}%" for c, s in share.items()]
        lines.append(f"- **Répartition des typologies (clustering)** : " + " | ".join(parts))

    lines += [
        "",
        "### Lecture rapide",
        "- La **courbe journalière** (`usage_daily_timeseries.png`) montre la stabilité/tension du réseau.",
        "- Le **profil horaire** (`usage_hourly_profile.png`) met en évidence les pics matin/soir.",
        "- La **heatmap heure×jour** (`usage_heatmap_hour_dow.png`) révèle les jours/horaires critiques.",
        "- La **variabilité station** (`usage_station_variability.png`) cible les points chauds pour l’opérationnel.",
        "- La **carte** (`assets/maps/usage_map.html`) visualise les zones par typologie/tension.",
    ]
    return "\n".join(lines)


def make_perf_insights() -> str:
    met = read_csv_safe(TABLES / "model_metrics.csv")
    lines = ["# Points clés — Performance des prévisions", ""]
    if met is not None and not met.empty:
        met = met.sort_values("horizon")
        best = met.iloc[0]
        lines += [
            f"- **MAE**: {best['MAE']:.2f} — **RMSE**: {best['RMSE']:.2f} au meilleur horizon ({int(best['horizon'])} min).",
            f"- **MAPE**: {100*best.get('MAPE', np.nan):.1f}% ; **sMAPE**: {100*best.get('sMAPE', np.nan):.1f}% (indicatifs).",
        ]
    lines += [
        "- **Obs vs Préd** agrégé (`mon_pred_vs_true.png`) : vérifie l’alignement temporel.",
        "- **Focus station** (`obs_vs_pred_station_24h.png`) : lisibilité des sous/sur-prédictions locales.",
        "- **MAE heure×jour** (`errors_hour_x_dow.png`) : créneaux où le modèle est fragile.",
        "- **Résidus** (`residual_hist.png`) & **biais horaire** (`bias_over_time.png`) : dérives systématiques.",
        "- **Calibration** (`calibration_plot.png`) : cohérence niveaux prédits vs réalisés.",
    ]
    return "\n".join(lines)

def make_monitoring_insights() -> str:
    health = read_csv_safe(TABLES / "data_health.csv")
    drift = read_csv_safe(TABLES / "psi_features.csv")
    imp = read_csv_safe(TABLES / "feature_importance_proxy.csv")
    daily_err = read_csv_safe(TABLES / "daily_error.csv")

    lines = ["# Points clés — Monitoring (données & modèle)", ""]
    if health is not None and not health.empty:
        hdict = {r["field"]: r.get("missing", np.nan) for _, r in health.iterrows()}
        miss_bikes = hdict.get("bikes", np.nan)
        miss_cap = hdict.get("capacity", np.nan)
        lines.append(f"- **Données manquantes** — bikes: {frac(miss_bikes)}, capacity: {frac(miss_cap)}.")
    if drift is not None and not drift.empty:
        top = drift.sort_values("psi", ascending=False).head(3)
        top_str = " ; ".join([f"{r['feature']} (PSI {r['psi']:.3f})" for _, r in top.iterrows()])
        lines.append(f"- **Drift (PSI)** — top: {top_str}.")
    if imp is not None and not imp.empty:
        topf = imp.sort_values("abs_corr", ascending=False).head(5)["feature"].tolist()
        lines.append(f"- **Features dominantes (proxy corrélation)** : {', '.join(topf)}.")
    if daily_err is not None and not daily_err.empty:
        last = daily_err.sort_values("day").tail(1)["err"].values[0]
        lines.append(f"- **MAE quotidien (dernier jour)** : {last:.2f} (cf. `mon_error_trend.png`).")

    lines += [
        "",
        "### Règles & seuils (recommandations)",
        "- **SLO** : surveiller MAE et RMSE par horizon ; alerte si dérive >+20% sur 3 jours.",
        "- **Réentraînement** : déclencher si **PSI ≥ 0.2** sur une feature clé **ou** MAPE ↑ soutenue.",
        "- **Sanity checks** : clamp [0, capacity], entrées hors-plage, valeurs manquantes critiques.",
    ]
    return "\n".join(lines)

def main(h_last=7, p_last=7, m_last=14, horizon=60, station=None):
    build_usage(last_days=h_last)
    build_performance(last_days=p_last, horizon=horizon, station=station)
    build_monitoring(last_days=m_last, horizon=horizon)

    usage_md = make_usage_insights()
    perf_md = make_perf_insights()
    mon_md = make_monitoring_insights()

    write_text(AUTO / "usage_insights.md", usage_md)
    write_text(AUTO / "perf_insights.md", perf_md)
    write_text(AUTO / "monitoring_insights.md", mon_md)

    print("\n[OK] Insights rédigés :")
    print(f" - {AUTO / 'usage_insights.md'}")
    print(f" - {AUTO / 'perf_insights.md'}")
    print(f" - {AUTO / 'monitoring_insights.md'}")
    print("\nAstuce (MkDocs Material) : ajoute l'extension 'pymdownx.snippets' et insère :")
    print('  --8<-- "exports/auto/usage_insights.md"')
    print('  --8<-- "exports/auto/perf_insights.md"')
    print('  --8<-- "exports/auto/monitoring_insights.md"')

if __name__ == "__main__":
    # Valeurs par défaut ; pour personnaliser, exécuter via CLI (à ajouter si besoin)
    main()