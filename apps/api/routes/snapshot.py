# apps/api/routes/snapshot.py
from fastapi import APIRouter
from api.core.snapshot_live import fetch_live_snapshot

router = APIRouter()

@router.get("/snapshot/live")
def snapshot_live():
    df = fetch_live_snapshot()
    return df.to_dict(orient="records")

@router.get("/snapshot/live/head")
def snapshot_live_head(n: int = 10):
    df = fetch_live_snapshot()
    return df.head(max(1, min(n, 100))).to_dict(orient="records")
