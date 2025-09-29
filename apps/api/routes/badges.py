from fastapi import APIRouter
router = APIRouter(prefix="/badges", tags=["badges"])

@router.get("")
def badges():
    # TODO: temp_C, precip_mm, wind_mps, parquet_age_min
    return {"temp_C": None, "precip_mm": None, "wind_mps": None, "parquet_age_min": None}
