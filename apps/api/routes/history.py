from fastapi import APIRouter
router = APIRouter(prefix="/history", tags=["history"])

@router.get("")
def history(station_id: str, from_: str, to: str):
    # TODO: retourner [{tbin_utc, nb_velos_bin, capacity_bin, occ_ratio_bin}]
    return []
