# api/schemas/forecast.py
from __future__ import annotations
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ForecastItem(BaseModel):
    station_id: int
    tbin_latest: datetime
    horizon_min: int
    bikes_pred: float
    bikes_pred_int: int
    capacity_bin: int
    pred_ts_utc: datetime
    model_version: str = Field(default="")

# Et dans la route:
# return [ForecastItem(**rec) for rec in records]
