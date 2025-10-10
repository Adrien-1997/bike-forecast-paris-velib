# apps/api/core/gcs.py
from __future__ import annotations
import json
import time
from typing import Tuple, Dict, Any, Optional
from datetime import datetime, timezone
from email.utils import format_datetime

try:
    from google.cloud import storage  # type: ignore
except Exception:
    storage = None  # vérifié à l'usage


# ───────────────────────── Low-level helpers ─────────────────────────
def _require_client() -> "storage.Client":
    if storage is None:
        raise RuntimeError("google-cloud-storage is not installed")
    return storage.Client()

def _parse_gs(uri: str) -> Tuple[str, str]:
    assert uri.startswith("gs://"), f"invalid GCS uri: {uri}"
    bucket, key = uri[5:].split("/", 1)
    return bucket, key

def _std_headers(blob) -> Dict[str, str]:
    etag = getattr(blob, "etag", None)
    updated = getattr(blob, "updated", None)
    if isinstance(updated, datetime):
        last_mod = format_datetime(updated.astimezone(timezone.utc))
    else:
        last_mod = None

    headers: Dict[str, str] = {"Cache-Control": "public, max-age=60"}
    if etag:
        headers["ETag"] = etag
    if last_mod:
        headers["Last-Modified"] = last_mod
    return headers


# ───────────────────────── Public: direct read ─────────────────────────
def read_blob_bytes(gs_uri: str) -> Tuple[bytes, Dict[str, str]]:
    client = _require_client()
    bkt, key = _parse_gs(gs_uri)
    blob = client.bucket(bkt).blob(key)
    data = blob.download_as_bytes()
    headers = _std_headers(blob)
    return data, headers

def read_blob_text(gs_uri: str, encoding: str = "utf-8") -> Tuple[str, Dict[str, str]]:
    raw, headers = read_blob_bytes(gs_uri)
    return raw.decode(encoding), headers

def read_blob_json(gs_uri: str) -> Tuple[Any, Dict[str, str]]:
    txt, headers = read_blob_text(gs_uri)
    return json.loads(txt), headers

def head_blob(gs_uri: str) -> Dict[str, str]:
    client = _require_client()
    bkt, key = _parse_gs(gs_uri)
    blob = client.bucket(bkt).get_blob(key)
    if blob is None:
        raise FileNotFoundError(gs_uri)
    return _std_headers(blob)


# ───────────────────────── In-memory cache (TTL + ETag) ─────────────────────────
# cache: uri -> { data: bytes, headers: {...}, etag: str|None, expire: float }
_CACHE: Dict[str, Dict[str, Any]] = {}

def _now() -> float:
    return time.time()

def _cache_get(uri: str) -> Optional[Dict[str, Any]]:
    item = _CACHE.get(uri)
    if not item:
        return None
    if item["expire"] < _now():
        # expiré
        _CACHE.pop(uri, None)
        return None
    return item

def _cache_set(uri: str, data: bytes, headers: Dict[str, str], ttl: int) -> None:
    _CACHE[uri] = {
        "data": data,
        "headers": headers,
        "etag": headers.get("ETag"),
        "expire": _now() + max(1, int(ttl)),
    }

def read_blob_bytes_cached(gs_uri: str, ttl_seconds: int = 60) -> Tuple[bytes, Dict[str, str]]:
    """
    Lit un blob avec cache mémoire (TTL) + validation ETag.
    - Si entrée en cache non expirée → renvoie direct.
    - Si expirée, fait un HEAD; si ETag identique → prolonge le TTL et renvoie cache.
      Sinon télécharge à nouveau.
    """
    # 1) cache non expiré
    cached = _cache_get(gs_uri)
    if cached is not None:
        return cached["data"], cached["headers"]

    # 2) cache expiré → HEAD pour comparer ETag (si on en a un)
    client = _require_client()
    bkt, key = _parse_gs(gs_uri)
    blob = client.bucket(bkt).get_blob(key)
    if blob is None:
        raise FileNotFoundError(gs_uri)
    headers_head = _std_headers(blob)
    etag_head = headers_head.get("ETag")

    prev = _CACHE.get(gs_uri)
    if prev and prev.get("etag") and etag_head and prev["etag"] == etag_head:
        # Même ETag → on peut réutiliser les bytes et juste prolonger le TTL
        _cache_set(gs_uri, prev["data"], headers_head, ttl_seconds)
        return prev["data"], headers_head

    # 3) télécharger frais
    data = blob.download_as_bytes()
    headers = _std_headers(blob)
    _cache_set(gs_uri, data, headers, ttl_seconds)
    return data, headers

def read_blob_text_cached(gs_uri: str, ttl_seconds: int = 60, encoding: str = "utf-8") -> Tuple[str, Dict[str, str]]:
    raw, headers = read_blob_bytes_cached(gs_uri, ttl_seconds=ttl_seconds)
    return raw.decode(encoding), headers

def read_blob_json_cached(gs_uri: str, ttl_seconds: int = 60) -> Tuple[Any, Dict[str, str]]:
    txt, headers = read_blob_text_cached(gs_uri, ttl_seconds=ttl_seconds)
    return json.loads(txt), headers
