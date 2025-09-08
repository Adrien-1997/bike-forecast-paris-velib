# src/monitoring.py
import numpy as np
import pandas as pd

def _to_naive_ns(x):
    dt = pd.to_datetime(x, errors="coerce", utc=True)
    try:
        dt = dt.dt.tz_localize(None)
    except AttributeError:
        dt = dt.tz_localize(None)
    return dt.astype("datetime64[ns]")

def _psi_1d(base: pd.Series, curr: pd.Series, bins=10) -> float:
    base = pd.to_numeric(base, errors="coerce").dropna()
    curr = pd.to_numeric(curr, errors="coerce").dropna()
    if len(base) < 10 or len(curr) < 10:
        return np.nan
    # bornes par quantiles sur base
    qs = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(base, qs))
    # gestion cas constant
    if len(cuts) <= 2:
        return 0.0
    # histogrammes normalisés
    b = np.histogram(base, bins=cuts)[0].astype(float)
    c = np.histogram(curr, bins=cuts)[0].astype(float)
    b = b / (b.sum() + 1e-12)
    c = c / (c.sum() + 1e-12)
    # éviter log(0)
    b = np.clip(b, 1e-8, None)
    c = np.clip(c, 1e-8, None)
    return float(np.sum((b - c) * np.log(b / c)))

def drift_report(df: pd.DataFrame, time_col="hour_utc",
                 baseline_days=30, current_days=7,
                 feature_cols=None) -> pd.DataFrame:
    """Compare last 7j vs. last 30j (hors 7j) via PSI + stats."""
    d = df.copy()
    d[time_col] = _to_naive_ns(d[time_col])
    tmax = d[time_col].max()
    t_cur_min = tmax - pd.Timedelta(days=current_days)
    t_base_min = tmax - pd.Timedelta(days=(baseline_days + current_days))

    cur = d[(d[time_col] > t_cur_min)]
    base = d[(d[time_col] > t_base_min) & (d[time_col] <= t_cur_min)]

    if feature_cols is None:
        cont = [c for c in ["occ_ratio_hour","temp_C","precip_mm","wind_mps",
                            "bikes_avg","docks_avg"] if c in d.columns]
        feature_cols = cont

    rows = []
    for c in feature_cols:
        psi = _psi_1d(base[c], cur[c]) if c in d.columns else np.nan
        rows.append({
            "feature": c,
            "psi": psi,
            "base_mean": float(pd.to_numeric(base[c], errors="coerce").mean()),
            "curr_mean": float(pd.to_numeric(cur[c],  errors="coerce").mean()),
            "base_std":  float(pd.to_numeric(base[c], errors="coerce").std()),
            "curr_std":  float(pd.to_numeric(cur[c],  errors="coerce").std()),
            "n_base":    int(base[c].notna().sum()),
            "n_curr":    int(cur[c].notna().sum()),
        })
    rep = pd.DataFrame(rows)
    def flag(v):
        if pd.isna(v): return "n/a"
        if v < 0.1:   return "OK"
        if v < 0.25:  return "Attention"
        return "Alerte"
    rep["psi_flag"] = rep["psi"].map(flag)
    return rep.sort_values("psi", ascending=False)
