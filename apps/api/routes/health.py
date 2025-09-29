from fastapi import APIRouter
import os, datetime as dt

router = APIRouter(prefix="/health", tags=["health"])

@router.get("")
def health():
    return {"status":"ok","image_tag":os.getenv("IMAGE_TAG",""),"utc":dt.datetime.utcnow().isoformat()+"Z"}
