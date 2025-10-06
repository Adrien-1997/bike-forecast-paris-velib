# apps/api/routes/history.py
from __future__ import annotations
from fastapi import APIRouter, Response

router = APIRouter(prefix="/history", tags=["history"])

@router.get("")
def history_disabled():
    """
    History endpoint is disabled: features parquet removed from the pipeline.
    """
    return Response(status_code=204)
