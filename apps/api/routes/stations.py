from fastapi import APIRouter
router = APIRouter(prefix="/stations", tags=["stations"])

@router.get("")
def list_stations():
    # TODO: retourner [{stationcode,name,lat,lon,capacity}]
    return []
