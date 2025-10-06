from fastapi import APIRouter
from api.core.snapshot_live import fetch_live_snapshot

router = APIRouter(prefix="/snapshot", tags=["snapshot"])

@router.get("")
def get_snapshot():
    """
    Live Vélib snapshot (without weather)
    """
    df = fetch_live_snapshot()
    return df.to_dict(orient="records") if hasattr(df, "to_dict") else []
