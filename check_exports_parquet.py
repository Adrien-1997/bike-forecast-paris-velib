#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 200)

# ================== Config ==================
DOWNLOADS = Path(r"H:\Downloads")  # adapte si besoin
DAY = "2025-10-05"                 # change le jour testé
EVENTS = DOWNLOADS / f"velib_exports_events_{DAY}.parquet"
PERF   = DOWNLOADS / f"velib_exports_perf_{DAY}.parquet"

# Colonnes REQUISES (subset obligatoire)
REQ_EVENTS = [
    "tbin_utc","station_id","bikes","capacity","mechanical","ebike",
    "status","status_code","lat","lon","name","temp_C","precip_mm","wind_mps",
    "occ_ratio","is_penury","is_saturation","h","min"
]
REQ_PERF = [
    "tbin_utc","station_id","horizon_bins","y_true","y_baseline_persist",
    "bikes","capacity","occ_ratio","h","min"
]
# Colonnes OPTIONNELLES que l'on check si présentes
OPT_EVENTS = []
OPT_PERF = ["y_pred_int"]

def _ok(msg):   print(f"[OK]  {msg}")
def _warn(msg): print(f"[WARN] {msg}")
def _fail(msg): print(f"[FAIL] {msg}")

def _to_naive_utc(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, utc=True, errors="coerce")
    # tz-aware -> naive UTC for display
    return out.dt.tz_convert("UTC").dt.tz_localize(None) if getattr(out.dt, "tz", None) is not None else out.dt.tz_localize(None)

def _require_columns(df: pd.DataFrame, required: list, label: str) -> bool:
    missing = [c for c in required if c not in df.columns]
    extra   = [c for c in df.columns if c not in required + (OPT_PERF if label=="PERF" else OPT_EVENTS)]
    if missing:
        _fail(f"{label}: colonnes manquantes: {missing}")
        return False
    if extra:
        _warn(f"{label}: colonnes supplémentaires détectées (ok): {extra}")
    else:
        _ok(f"{label}: toutes les colonnes requises sont présentes (les extras sont tolérées)")
    return True

def _range_and_dups(df: pd.DataFrame, label: str):
    if "tbin_utc" in df.columns:
        df["tbin_utc"] = _to_naive_utc(df["tbin_utc"])
        print(f"TIME RANGE ({label}): {df['tbin_utc'].min()} → {df['tbin_utc'].max()} | unique bins = {df['tbin_utc'].nunique()}")
    # Doublons clé
    key = ["station_id","tbin_utc"]
    if all(c in df.columns for c in key):
        dups = df.duplicated(subset=key).sum()
        if dups == 0:
            _ok(f"{label}: aucun doublon ({', '.join(key)})")
        else:
            _warn(f"{label}: {dups} doublon(s) sur ({', '.join(key)})")

def _check_events(df: pd.DataFrame):
    print("="*80)
    print(f"EVENTS: {EVENTS}")
    if not EVENTS.exists():
        _fail(f"Fichier introuvable: {EVENTS}"); return
    df = pd.read_parquet(EVENTS)
    print(f"Rows={len(df):,}  Cols={len(df.columns)}")
    print("Columns:", list(df.columns)); print()

    if not _require_columns(df, REQ_EVENTS, "EVENTS"):
        return

    _range_and_dups(df, "EVENTS")

    # Météo non 100% nulle
    for c in ["temp_C","precip_mm","wind_mps"]:
        if c in df.columns:
            null_rate = df[c].isna().mean()*100
            if null_rate < 100:
                _ok(f"EVENTS: {c} présent (null={null_rate:.1f}%)")
            else:
                _warn(f"EVENTS: {c} = 100% null")

    # Types compacts attendus pour flags
    if str(df["is_penury"].dtype) == "Int8" and str(df["is_saturation"].dtype) == "Int8":
        _ok("EVENTS: is_penury / is_saturation en Int8")
    else:
        _warn(f"EVENTS: types inattendus pour is_penury/is_saturation -> {df['is_penury'].dtype} / {df['is_saturation'].dtype}")

    # Cohérence occ_ratio
    mask = (pd.to_numeric(df["capacity"], errors="coerce") > 0) & df["bikes"].notna()
    if mask.any():
        calc = (df.loc[mask, "bikes"].astype(float) / df.loc[mask, "capacity"].astype(float))
        diff = (df.loc[mask, "occ_ratio"].astype(float) - calc).abs()
        bad = (diff > 1e-6).sum()
        if bad == 0:
            _ok("EVENTS: occ_ratio cohérent avec bikes/capacity")
        else:
            _warn(f"EVENTS: occ_ratio ≠ bikes/capacity sur {bad} lignes (tol=1e-6)")

    # h/min présents et cohérents
    if "h" in df.columns and "min" in df.columns:
        h_bad  = (~df["h"].between(0,23)).sum()
        m_bad  = (~df["min"].isin([0,5,10,15,20,25,30,35,40,45,50,55])).sum()
        if h_bad == 0 and m_bad == 0:
            _ok("EVENTS: colonnes h/min présentes et plausibles (bins 5 min)")
        else:
            _warn(f"EVENTS: incohérences h/min (h_bad={h_bad}, m_bad={m_bad})")

    print("\nHEAD(8):")
    print(df.head(8).to_string(index=False))
    print()

def _check_perf(df: pd.DataFrame):
    print("="*80)
    print(f"PERF:   {PERF}")
    if not PERF.exists():
        _fail(f"Fichier introuvable: {PERF}"); return
    df = pd.read_parquet(PERF)
    print(f"Rows={len(df):,}  Cols={len(df.columns)}")
    print("Columns:", list(df.columns)); print()

    if not _require_columns(df, REQ_PERF, "PERF"):
        return

    _range_and_dups(df, "PERF")

    # Horizons autorisés
    if "horizon_bins" in df.columns:
        hb = sorted(pd.Series(df["horizon_bins"].dropna().unique()).tolist())
        print("horizon_bins unique:", hb)
        if set(hb).issubset({3,12}):
            _ok("PERF: horizon_bins dans {3,12}")
        else:
            _warn("PERF: horizon_bins inattendus (≠ 3/12)")

    # Persistance = bikes
    if {"y_baseline_persist","bikes"}.issubset(df.columns):
        eq = df["y_baseline_persist"].astype("float64") - df["bikes"].astype("float64")
        bad = (eq != 0).sum()
        if bad == 0:
            _ok("PERF: y_baseline_persist == bikes (persistance) ✅")
        else:
            _warn(f"PERF: y_baseline_persist ≠ bikes sur {bad} lignes")

    # Cohérence occ_ratio
    mask = (pd.to_numeric(df["capacity"], errors="coerce") > 0) & df["bikes"].notna()
    if mask.any():
        calc = (df.loc[mask, "bikes"].astype(float) / df.loc[mask, "capacity"].astype(float))
        diff = (df.loc[mask, "occ_ratio"].astype(float) - calc).abs()
        bad = (diff > 1e-6).sum()
        if bad == 0:
            _ok("PERF: occ_ratio cohérent avec bikes/capacity")
        else:
            _warn(f"PERF: occ_ratio ≠ bikes/capacity sur {bad} lignes (tol=1e-6)")

    # h/min présents et plausibles
    if "h" in df.columns and "min" in df.columns:
        h_bad  = (~df["h"].between(0,23)).sum()
        m_bad  = (~df["min"].isin([0,5,10,15,20,25,30,35,40,45,50,55])).sum()
        if h_bad == 0 and m_bad == 0:
            _ok("PERF: colonnes h/min présentes et plausibles (bins 5 min)")
        else:
            _warn(f"PERF: incohérences h/min (h_bad={h_bad}, m_bad={m_bad})")
    else:
        _fail("PERF: colonnes h et/ou min manquantes")

    # y_pred_int (optionnel) — si présent, quelques checks
    if "y_pred_int" in df.columns:
        # doit être entier dans bornes [0, capacity] quand capacity connue
        cap = pd.to_numeric(df["capacity"], errors="coerce")
        ypi = pd.to_numeric(df["y_pred_int"], errors="coerce")
        neg = (ypi < 0).sum()
        over = ((ypi > cap) & cap.notna()).sum()
        if neg == 0 and over == 0:
            _ok("PERF: y_pred_int borné correctement (0..capacity)")
        else:
            _warn(f"PERF: y_pred_int hors bornes (neg={neg}, >cap={over})")
        # petite métrique de sanity si y_true dispo
        if "y_true" in df.columns:
            mt = df[["y_true","y_pred_int"]].dropna()
            if len(mt) > 0:
                mae = (mt["y_true"].astype(float) - mt["y_pred_int"].astype(float)).abs().mean()
                _ok(f"PERF: MAE(y_true, y_pred_int) = {mae:.3f} (sur {len(mt):,} lignes)")
    else:
        _warn("PERF: y_pred_int absent (ok si non souhaité)")

    print("\nHEAD(8):")
    print(df.head(8).to_string(index=False))
    print()

def main():
    _check_events(None)
    _check_perf(None)
    print("="*80)
    print("DONE.")

if __name__ == "__main__":
    sys.exit(main())
