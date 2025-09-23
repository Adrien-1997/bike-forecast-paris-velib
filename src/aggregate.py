# src/aggregate.py
from pathlib import Path
import os
import pandas as pd

# 👇 IMPORTANT : on importe depuis le package src
from src.weather import fetch_history, fetch_forecast

def _floor_hour_naive(s):
    s = pd.to_datetime(s, utc=True, errors="coerce")
    return s.dt.floor("h").dt.tz_localize(None)

def main(input_path: Path, output_path: Path):
    print(f"[aggregate] input={input_path} output={output_path}")

    # 1) Lecture ingest
    df = pd.read_parquet(input_path)
    if "ts_utc" not in df.columns:
        raise ValueError("input parquet must contain 'ts_utc'")
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce").dt.tz_localize(None)
    df["hour_utc"] = _floor_hour_naive(df["ts_utc"])
    start_ts, end_ts = df["ts_utc"].min(), df["ts_utc"].max()
    print(f"[aggregate] snapshot window: {start_ts} → {end_ts} (rows={len(df)})")

    # 2) Météo — historique + éventuellement forecast
    try:
        w = fetch_history(start_ts, end_ts)
    except Exception as e:
        print(f"[aggregate] fetch_history failed: {e}")
        w = pd.DataFrame(columns=["hour_utc","temp_C","precip_mm","wind_mps"])

    hist_n = 0 if w is None or w.empty else len(w)
    print(f"[aggregate] weather hist rows: {hist_n}")

    # Forecast si l’extrémité dépasse “maintenant”
    need_fx = (pd.Timestamp.utcnow().tz_localize(None) < end_ts)
    print(f"[aggregate] need forecast: {need_fx}")

    wf = pd.DataFrame(columns=["hour_utc","temp_C","precip_mm","wind_mps"])
    if need_fx:
        try:
            wf = fetch_forecast(end_ts, horizon_h=24)
        except Exception as e:
            print(f"[aggregate] fetch_forecast failed: {e}")
    fx_n = 0 if wf is None or wf.empty else len(wf)
    print(f"[aggregate] weather forecast rows: {fx_n}")

    # Concat météo et typage
    weather = pd.concat([w, wf], ignore_index=True) if hist_n or fx_n else pd.DataFrame(
        columns=["hour_utc","temp_C","precip_mm","wind_mps"]
    )
    if not weather.empty:
        weather["hour_utc"] = pd.to_datetime(weather["hour_utc"], utc=True, errors="coerce").tz_localize(None)
        for c in ["temp_C","precip_mm","wind_mps"]:
            if c in weather:
                weather[c] = pd.to_numeric(weather[c], errors="coerce")
        weather = weather.drop_duplicates(subset=["hour_utc"]).sort_values("hour_utc")
        print(f"[aggregate] merged weather rows: {len(weather)}")
    else:
        print("[aggregate] WARNING: no weather available (merge will yield NaN)")

    # 3) Jointure sur hour_utc
    if not weather.empty:
        df = df.merge(weather, on="hour_utc", how="left")
    else:
        # Colonnes vides pour cohérence des schémas
        for c in ["temp_C","precip_mm","wind_mps"]:
            if c not in df.columns:
                df[c] = pd.NA

    print("[aggregate] sample after weather merge:")
    print(df[["hour_utc","stationcode","temp_C","precip_mm","wind_mps"]].head(5))

    # 4) Écriture sortie
    df.to_parquet(output_path, index=False)
    print(f"[aggregate] saved {len(df)} rows -> {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # 🔧 aligne avec ce qu’on a convenu dans le Dockerfile
    parser.add_argument("--input",  default="staging_ingest.parquet")
    parser.add_argument("--output", default="velib.parquet")
    args = parser.parse_args()
    main(Path(args.input), Path(args.output))
