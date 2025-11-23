# api/routes/history.py

"""History endpoint (disabled).

This router exposes a `/history` endpoint that is intentionally **disabled**:
- the historical "features parquet" has been removed from the pipeline,
- the route always returns HTTP 204 No Content,
- it is kept only for backward compatibility so that old UIs / clients
  calling `/history` do not fail with a 404.

If, in the future, history is reintroduced (e.g. time-travel features),
this module can be extended to proxy the new artefacts.
"""

from __future__ import annotations
from fastapi import APIRouter, Response

router = APIRouter(prefix="/history", tags=["history"])


@router.get("")
def history_disabled():
    """Return 204 No Content: history feature is currently disabled.

    This avoids breaking existing consumers while clearly signalling that
    no historical data is available via this endpoint.
    """
    return Response(status_code=204)
