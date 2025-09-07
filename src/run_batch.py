import pandas as pd
from src.aggregate import hourly_occupancy
from src.forecast import train_forecast

if __name__ == "__main__":
    hour = hourly_occupancy()
    preds = train_forecast(hour, horizon_h=24)
    hour.to_parquet("exports/velib_hourly.parquet", index=False)
    preds.to_parquet("exports/velib_forecast_24h.parquet", index=False)
    # petit CSV pour GitHub Pages
    preds.to_csv("exports/velib_forecast_24h.csv", index=False)
    print("OK batch: exports/*")
