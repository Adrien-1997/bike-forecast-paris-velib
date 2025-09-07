import os, pandas as pd
from src.aggregate import hourly_occupancy
from src.forecast import train_and_forecast

if __name__ == "__main__":
    os.makedirs("exports", exist_ok=True)
    hour = hourly_occupancy(with_weather=True)
    if hour.empty:
        print("No hourly data yet — run ingest a bit more.")
    else:
        preds = train_and_forecast(hour, horizon_h=24)
        hour.to_parquet("exports/velib_hourly.parquet", index=False)
        preds.to_parquet("exports/velib_forecast_24h.parquet", index=False)
        preds.to_csv("exports/velib_forecast_24h.csv", index=False)
        print("OK batch: exports/*")
