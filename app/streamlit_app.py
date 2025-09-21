from __future__ import annotations

from pathlib import Path
from typing import Tuple, List
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import folium
from folium.plugins import MarkerCluster
import streamlit as st
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils_io import get_export_path

# -----------------------------------------------------------------------------
# BOOTSTRAP
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Prévisions Vélib’ Paris", page_icon="🚲", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # permet d'importer src.*

from src.forecast import load_model_bundle
from src.cal_features import add_calendar_features
from src.utils_io import get_export_path

MODELS_DIR = ROOT / "models"
DEBUG = False

st.session_state.setdefault("map_center", (48.8566, 2.3522))
st.session_state.setdefault("map_zoom", 15)
st.session_state.setdefault("map_highlight", None)
st.session_state.setdefault("cache_buster", 0)  # pour invalider les @st.cache_data ciblés

# -----------------------------------------------------------------------------
# STYLES
# -----------------------------------------------------------------------------
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: .75rem; max-width: 1300px; }
h2, h3 { margin-top: 1rem !important; margin-bottom: .6rem !important; }
.badges { display:flex; flex-wrap:wrap; gap:.8rem; margin:.5rem 0 1rem; }
.badge { display:inline-block; padding:.35rem .8rem; border-radius:999px; font-size:.80rem;
  border:1px solid var(--border-color); background:var(--badge-bg); color:var(--text-strong); }
.kpi-row { margin: 1rem 0 !important; }
.kpi-card { background: var(--background-color); border: 1px solid var(--border-color); border-radius: 12px;
  padding: 1rem 1.2rem; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
.kpi-title { color: var(--text-muted); font-size: .85rem; margin-bottom: .35rem; }
.kpi-value { font-size: 1.35rem; font-weight: 700; color: var(--text-strong); }
.finder-card [data-testid="column"]{ display:flex; align-items:flex-end; gap:.5rem; }
.finder-card .stButton > button{ height:42px !important; margin-bottom:2px; }
.leaflet-container { max-width: 100% !important; }
.monitoring-fig { max-width: 900px; margin: 0 auto; width: 100%; }
.monitoring-fig img { width: 100% !important; height: auto !important; display: block; }
div[data-testid="stDataFrameResizable"] { overflow-x: auto; }
:root { --background-color:#ffffff; --border-color:#e5e7eb; --text-muted:#6b7280; --text-strong:#111827; --badge-bg:#f9fafb; }
@media (prefers-color-scheme: dark) {
  :root { --background-color:#1f2937; --border-color:#374151; --text-muted:#9ca3af; --text-strong:#f3f4f6; --badge-bg:#374151; }
}
@media (max-width: 768px) {
  .block-container { padding-left:.5rem !important; padding-right:.5rem !important; }
  h2, h3 { margin-top:.6rem !important; margin-bottom:.5rem !important; }
  .badges { gap:.6rem; margin:.4rem 0 .8rem; }
  .finder-card [data-testid="column"] { width: 100% !important; align-items: stretch !important; }
  .finder-card .stTextInput, .finder-card .stNumberInput, .finder-card .stButton { width: 100% !important; }
  div[data-testid="stIFrame"] iframe, iframe[title="st.iframe"], .map-wrap iframe { height: 60vh !important; min-height: 360px !important; }
  .monitoring-fig { max-width: 100%; }
}
</style>
""", unsafe_allow_html=True)

TITLE = "🚲 Prévisions Vélib’ Paris"

# -----------------------------------------------------------------------------
# RÉSOLUTIONS (HF + OpenData)
# -----------------------------------------------------------------------------
# --- remplace ta fonction _velib_local_path par celle-ci
def _velib_local_path(force_hf: bool = False) -> str:
    """
    Retourne le chemin local du parquet.
    - Appel simple sans mot-clé si force_hf=False
    - Si force_hf=True, tente plusieurs signatures courantes : force, force_download, refresh
    """
    try:
        if not force_hf:
            return str(get_export_path("velib.parquet"))  # appel simple
        # force_hf=True : tester différentes signatures
        try:
            return str(get_export_path("velib.parquet", force=True))
        except TypeError:
            pass
        try:
            return str(get_export_path("velib.parquet", force_download=True))
        except TypeError:
            pass
        try:
            return str(get_export_path("velib.parquet", refresh=True))
        except TypeError:
            pass
        # Dernier recours : appel simple (sans forcer)
        return str(get_export_path("velib.parquet"))
    except Exception as e:
        st.error(f"get_export_path a échoué : {e}")
        # Fallback très conservateur si ta fonction n'est pas dispo : chemin attendu en repo
        return "docs/exports/velib.parquet"

def _gentle_refresh_from_hf() -> str | None:
    try:
        path = _velib_local_path(force_hf=True)   # ← force via multi-signatures
        st.session_state["cache_buster"] += 1
        st.toast("Parquet rafraîchi depuis Hugging Face ✅")
        return path
    except Exception as e:
        st.error(f"Échec du rafraîchissement HF : {e}")
        return None

def _format_badge(val, unit, icon):
    try:
        v = float(val);  return f"<span class='badge'>{icon} {v:.1f} {unit}</span>"
    except Exception:
        return ""

def weather_badges_from_parquet(parquet_path: str) -> str:
    try:
        dfp = pd.read_parquet(parquet_path)
        if dfp.empty: return ""
        dfp["hour_utc"] = pd.to_datetime(dfp["hour_utc"], utc=True).dt.tz_localize(None)
        last_h = dfp["hour_utc"].max()
        sub = dfp[dfp["hour_utc"] == last_h][["temp_C", "precip_mm", "wind_mps"]].dropna(how="all")
        if sub.empty: return ""
        temp = sub.get("temp_C", pd.Series([np.nan])).mean()
        rain = sub.get("precip_mm", pd.Series([np.nan])).mean()
        wind = sub.get("wind_mps", pd.Series([np.nan])).mean()
        # NB: wind_mps → km/h visuel
        wind_kmh = None if pd.isna(wind) else float(wind) * 3.6
        parts = [
            _format_badge(temp, "°C", "🌡️"),
            _format_badge(rain, "mm", "🌧️"),
            _format_badge(wind_kmh if wind_kmh is not None else np.nan, "km/h", "💨"),
        ]
        return "".join([p for p in parts if p])
    except Exception:
        return ""

def _extract_latlon(coord):
    if isinstance(coord, dict):
        return coord.get("lat"), coord.get("lon")
    if isinstance(coord, (list, tuple)) and len(coord) == 2:
        a, b = coord
        try:
            af, bf = float(a), float(b)
            if abs(af) < 10 and abs(bf) > 10:  # heuristique: [lon,lat] -> swap
                return bf, af
            return af, bf
        except Exception:
            return None, None
    return None, None

def _make_session() -> requests.Session:
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter
    s = requests.Session()
    retry = Retry(total=3, read=3, connect=3, backoff_factor=0.5,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET", "HEAD", "OPTIONS"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "velib-app/1.0 (+github.com/Adrien-1997)"})
    return s

def fetch_opendata_v1() -> pd.DataFrame:
    URL = "https://opendata.paris.fr/api/records/1.0/search/?dataset=velib-disponibilite-en-temps-reel&rows=2000"
    r = _make_session().get(URL, timeout=25)
    r.raise_for_status()
    rows = []
    for rec in r.json().get("records", []):
        f = rec.get("fields", {})
        lat, lon = _extract_latlon(f.get("coordonnees_geo"))
        rows.append({
            "stationcode": f.get("stationcode") or f.get("station_id"),
            "name": f.get("name"),
            "lat": lat, "lon": lon,
            "capacity": f.get("capacity"),
            "numbikesavailable": f.get("numbikesavailable"),
            "numdocksavailable": f.get("numdocksavailable"),
        })
    df = pd.DataFrame(rows).dropna(subset=["stationcode","lat","lon"])
    for c in ["capacity","numbikesavailable","numdocksavailable","lat","lon"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["fetched_at_utc"] = pd.Timestamp.utcnow()
    return df[["stationcode","name","lat","lon","capacity","numbikesavailable","numdocksavailable","fetched_at_utc"]]

def fetch_synthetic(n: int = 60) -> pd.DataFrame:
    now = pd.Timestamp.utcnow()
    rng = np.random.default_rng(42)
    rows = []
    for k in range(n):
        cap = int(rng.integers(20, 45))
        bikes = int(rng.integers(0, cap))
        rows.append({
            "stationcode": f"999{k:03d}",
            "name": f"Station {k:03d}",
            "lat": 48.85 + rng.normal(0, 0.01),
            "lon": 2.35 + rng.normal(0, 0.01),
            "capacity": cap,
            "numbikesavailable": bikes,
            "numdocksavailable": cap - bikes,
            "fetched_at_utc": now,
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=300)
def load_live_data_cached() -> Tuple[pd.DataFrame, str, str, pd.Timestamp]:
    try:
        df = fetch_opendata_v1()
        return df, "OpenData", "", df["fetched_at_utc"].iloc[0]
    except Exception as e2:
        df = fetch_synthetic(60)
        return df, "fallback", f"OpenData failed: {e2}", df["fetched_at_utc"].iloc[0]

# -----------------------------------------------------------------------------
# PARQUET (HF) — gentle refresh
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _read_parquet_cached(path: str, buster: int) -> pd.DataFrame:
    # buster n'est pas utilisé, juste là pour invalider proprement
    return pd.read_parquet(path)

def _parquet_age_minutes(path: str) -> int | None:
    try:
        dfp = _read_parquet_cached(path, st.session_state["cache_buster"])
        if dfp.empty: return None
        ts_col = dfp["tbin_utc"] if "tbin_utc" in dfp.columns else dfp.get("ts")
        if ts_col is None: return None
        ts = pd.to_datetime(ts_col, errors="coerce").max()
        if pd.isna(ts): return None
        if ts.tzinfo is None: ts = ts.tz_localize("UTC")
        now_paris = pd.Timestamp.now(tz="Europe/Paris").floor("15min")
        ts_paris = ts.tz_convert("Europe/Paris").floor("15min")
        return int((now_paris - ts_paris).total_seconds() // 60)
    except Exception:
        return None

def _gentle_refresh_from_hf() -> str | None:
    """Télécharge depuis HF (force=True), invalide cache ciblé, retourne le chemin local."""
    try:
        path = _velib_local_path(force_hf=True)  # HF → local (écrase)
        st.session_state["cache_buster"] += 1    # invalider _read_parquet_cached
        st.toast("Parquet rafraîchi depuis Hugging Face ✅")
        return path
    except Exception as e:
        st.error(f"Échec du rafraîchissement HF : {e}")
        return None

# -----------------------------------------------------------------------------
# FEATURES LIVE (depuis velib.parquet)
# -----------------------------------------------------------------------------
def _load_recent_bins(parquet_path: str) -> pd.DataFrame:
    df = _read_parquet_cached(parquet_path, st.session_state["cache_buster"])
    if df.empty: return df
    cutoff = pd.Timestamp.utcnow().floor("15min") - pd.Timedelta(days=2)
    return df[pd.to_datetime(df["tbin_utc"]) >= cutoff.tz_localize(None)].copy()

def _add_lags_rollings(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    req_cols = ["tbin_utc", "nb_velos_bin", "occ_ratio_bin"]
    for c in req_cols:
        if c not in df.columns: df[c] = np.nan
    gdf = df.copy()
    gdf["tbin_utc"] = pd.to_datetime(gdf["tbin_utc"], utc=True, errors="coerce").dt.tz_localize(None)
    gdf["nb_velos_bin"] = pd.to_numeric(gdf["nb_velos_bin"], errors="coerce")
    gdf["occ_ratio_bin"] = pd.to_numeric(gdf["occ_ratio_bin"], errors="coerce")

    def _per_station(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("tbin_utc").copy()
        for b in (1, 2, 3, 4, 8, 16):
            g[f"lag_nb_{b}b"]  = g["nb_velos_bin"].shift(b)
            g[f"lag_occ_{b}b"] = g["occ_ratio_bin"].shift(b)
        g["roll_nb_4b"]  = g["nb_velos_bin"].rolling(window=4,  min_periods=1).mean()
        g["roll_nb_8b"]  = g["nb_velos_bin"].rolling(window=8,  min_periods=1).mean()
        g["roll_occ_4b"] = g["occ_ratio_bin"].rolling(window=4,  min_periods=1).mean()
        g["roll_occ_8b"] = g["occ_ratio_bin"].rolling(window=8,  min_periods=1).mean()
        g["trend_nb_4b"]  = (g["nb_velos_bin"]  - g["nb_velos_bin"].shift(4))  / 4.0
        g["trend_occ_4b"] = (g["occ_ratio_bin"] - g["occ_ratio_bin"].shift(4)) / 4.0
        return g

    try:
        out = gdf.groupby("stationcode", group_keys=False).apply(_per_station, include_groups=False)
    except TypeError:
        out = gdf.groupby("stationcode", group_keys=False).apply(_per_station)

    if "stationcode" not in out.columns and "stationcode" in gdf.columns:
        out = out.join(gdf[["stationcode"]])

    float_cols = out.select_dtypes(include=["float64"]).columns
    if len(float_cols): out[float_cols] = out[float_cols].astype("float32", copy=False)
    return out

def build_live_feature_frame(parquet_path: str, feat_cols: List[str]) -> pd.DataFrame:
    base = _load_recent_bins(parquet_path)
    if base is None or base.empty:
        return pd.DataFrame(columns=feat_cols)
    base["tbin_utc"] = pd.to_datetime(base["tbin_utc"], utc=True).dt.tz_localize(None)
    base["hour_utc"] = pd.to_datetime(base["hour_utc"], utc=True).dt.tz_localize(None)
    base = _add_lags_rollings(base)
    base = add_calendar_features(base, tz="Europe/Paris")
    base = base.sort_values(["stationcode","tbin_utc"]).groupby("stationcode", as_index=False).tail(1)

    X = base.copy()
    for c in feat_cols:
        if c not in X.columns: X[c] = 0.0
    X = X[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
    X["stationcode"] = base["stationcode"].astype(str).values
    return X

# -----------------------------------------------------------------------------
# PRÉDICTIONS
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_model_cached_minutes(h_minutes: int):
    model, feats = load_model_bundle(horizon_minutes=h_minutes, model_dir=str(MODELS_DIR))
    return model, feats

def predict_nbvelos_t1h(df_now: pd.DataFrame, parquet_path: str) -> pd.DataFrame:
    try:
        model, feat_cols = _load_model_cached_minutes(60)
        X = build_live_feature_frame(parquet_path, feat_cols)
        if X.empty:
            raise RuntimeError("No live features available (empty X)")
        stationcode = X["stationcode"].astype(str).values
        Xmat = X.drop(columns=["stationcode"], errors="ignore")
        best_it = getattr(model, "best_iteration", None)
        y = model.predict(Xmat, num_iteration=best_it) if best_it else model.predict(Xmat)
        y = np.maximum(y, 0.0)
        return pd.DataFrame({"stationcode": stationcode, "y_nb_pred": y})
    except Exception as e:
        st.warning(f"⚠️ Fallback baseline activé : {type(e).__name__}: {e}")
        y = pd.to_numeric(df_now.get("numbikesavailable"), errors="coerce").fillna(0).astype(int)
        return pd.DataFrame({"stationcode": df_now["stationcode"].astype(str), "y_nb_pred": y})

# -----------------------------------------------------------------------------
# ROUTER & ACTIONS
# -----------------------------------------------------------------------------
PAGE = st.sidebar.radio("Navigation", ["Carte", "Monitoring réseau"], index=0)
if st.sidebar.button("Actualiser données OpenData"):
    load_live_data_cached.clear()
if st.sidebar.button("🔄 Recharger le parquet depuis HF", use_container_width=True):
    _gentle_refresh_from_hf()

# -----------------------------------------------------------------------------
# DONNÉES + PRÉDICTIONS (différées après démarrage)
# -----------------------------------------------------------------------------
with st.spinner("Initialisation des données…"):
    df_now, source_label, debug_reason, fetched_utc = load_live_data_cached()
try:
    fetched_local = pd.to_datetime(fetched_utc, utc=True).tz_convert("Europe/Paris")
except Exception:
    fetched_local = pd.Timestamp.utcnow().tz_localize("UTC").tz_convert("Europe/Paris")

# --- contrôle fraicheur parquet & gentle refresh (sans reboot global)
parquet_path = _velib_local_path(force_hf=False)
_age = _parquet_age_minutes(parquet_path)
if _age is not None and _age > 15:
    st.warning(f"⏱️ Parquet âgé de {_age} min → rafraîchissement depuis HF…")
    p2 = _gentle_refresh_from_hf()
    if p2: parquet_path = p2

# --- prédiction
pred_df = predict_nbvelos_t1h(df_now, parquet_path)
df_with_pred = df_now.merge(pred_df, on="stationcode", how="left")
if "capacity" in df_with_pred:
    cap = pd.to_numeric(df_with_pred["capacity"], errors="coerce").fillna(0).astype(int)
    df_with_pred["y_nb_pred"] = np.clip(pd.to_numeric(df_with_pred["y_nb_pred"], errors="coerce").fillna(0), 0, cap).astype(int)

# -----------------------------------------------------------------------------
# BADGES & UTILITAIRES
# -----------------------------------------------------------------------------
def render_badges(extra: str = ""):
    st.markdown(
        f"<div class='badges'>"
        f"<span class='badge'>Source : <b>{source_label}</b></span>"
        f"<span class='badge'>Requête : {fetched_local.strftime('%d/%m %H:%M')}</span>"
        f"{extra}"
        f"</div>",
        unsafe_allow_html=True,
    )

def parquet_freshness_badge(path: str) -> str:
    try:
        dfp = _read_parquet_cached(path, st.session_state['cache_buster'])
        if dfp.empty:
            return "<span class='badge'>Parquet: vide</span>"
        ts_col = dfp["tbin_utc"] if "tbin_utc" in dfp.columns else dfp.get("ts", None)
        if ts_col is None:
            return "<span class='badge'>Parquet: timestamp absent</span>"
        ts = pd.to_datetime(ts_col, errors="coerce").max()
        if pd.isna(ts): return "<span class='badge'>Parquet: invalide</span>"
        if ts.tzinfo is None: ts = ts.tz_localize("UTC")
        now_paris = pd.Timestamp.now(tz="Europe/Paris").floor("15min")
        ts_paris = ts.tz_convert("Europe/Paris").floor("15min")
        delta_min = int((now_paris - ts_paris).total_seconds() // 60)
        if delta_min <= 15:
            color = "#dcfce7; color:#166534"
        elif delta_min <= 60:
            color = "#fef9c3; color:#854d0e"
        else:
            color = "#fee2e2; color:#991b1b"
        return f"<span class='badge' style='background:{color}'>Parquet: {delta_min} min</span>"
    except Exception:
        return "<span class='badge'>Parquet: erreur</span>"

def geocode_address(q: str) -> tuple[float, float] | None:
    q = (q or "").strip()
    if not q: return None
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search",
                         params={"q": q, "format": "json", "limit": 1, "accept-language": "fr"},
                         headers={"User-Agent": "velib-app/1.0"}, timeout=10)
        r.raise_for_status()
        js = r.json()
        return (float(js[0]["lat"]), float(js[0]["lon"])) if js else None
    except Exception:
        return None

# -----------------------------------------------------------------------------
# PAGES
# -----------------------------------------------------------------------------
if PAGE == "Carte":
    st.markdown(f"### {TITLE}")

    try:
        display_mode = st.segmented_control("Mode d'affichage",
            options=["Actuel", "Prévision T+1h"],
            selection_mode="single", default="Actuel", label_visibility="collapsed")
    except Exception:
        display_mode = st.radio("Mode d'affichage", ["Actuel", "Prévision T+1h"], index=0, horizontal=True, label_visibility="collapsed")
    display_pred = (display_mode == "Prévision T+1h")

    wx_badges = weather_badges_from_parquet(parquet_path)
    pf_badge = parquet_freshness_badge(parquet_path)
    render_badges(
        f"<span class='badge'>Mode : <b>{'Prévision T+1h' if display_pred else 'Actuel'}</b></span>"
        + wx_badges
        + pf_badge
    )

    st.markdown("<div class='finder-card'>", unsafe_allow_html=True)
    st.markdown("#### Trouver une station (≥ vélos)")
    c1, c2 = st.columns([2.6, 1.0])
    with c1:
        addr_text = st.text_input("Adresse / lieu", placeholder="Ex: 10 Rue de Rivoli, Paris", key="addr_text")
    with c2:
        min_bikes = st.number_input("Vélos min.", min_value=1, max_value=50, value=5, step=1)
    go = st.button("Trouver une station", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    origin = None
    if (addr_text or "").strip():
        geoc = geocode_address(addr_text)
        if geoc:
            origin = geoc
            st.session_state["map_center"] = geoc
            st.session_state["map_zoom"] = 18
            st.session_state["map_highlight"] = None

    user_lat, user_lon = origin if origin else (48.8566, 2.3522)

    def find_nearest_station(df: pd.DataFrame, user_lat: float, user_lon: float,
                             min_bikes: int, use_prediction: bool) -> dict | None:
        dfx = df.copy()
        for c in ["lat","lon","capacity","numbikesavailable","y_nb_pred"]:
            dfx[c] = pd.to_numeric(dfx.get(c), errors="coerce")
        if use_prediction:
            dfx = dfx[dfx["y_nb_pred"] >= int(min_bikes)]
        else:
            dfx = dfx[dfx["numbikesavailable"] >= int(min_bikes)]
        if dfx.empty:
            return None
        R = 6371000.0
        lat1 = np.radians(float(user_lat)); lon1 = np.radians(float(user_lon))
        lat2 = np.radians(dfx["lat"].astype(float)); lon2 = np.radians(dfx["lon"].astype(float))
        dlat = lat2 - lat1; dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        dfx["dist_m"] = 2 * R * np.arcsin(np.sqrt(a))
        return dfx.sort_values("dist_m").iloc[0].to_dict()

    target = None
    if go:
        target = find_nearest_station(
            df_with_pred, user_lat=user_lat, user_lon=user_lon,
            min_bikes=int(min_bikes), use_prediction=bool(display_pred),
        )
        if target is None:
            st.warning("Aucune station trouvée avec ce seuil.")
        else:
            st.success(f"✅ Station la plus proche : {target['name']} (#{target['stationcode']}) à ~{int(target['dist_m'])} m")
            st.session_state["map_highlight"] = str(target["stationcode"])
            st.session_state["map_center"] = (float(target["lat"]), float(target["lon"]))
            st.session_state["map_zoom"] = 17

    def _ring_color(ratio: float) -> str:
        if pd.isna(ratio): return "#9aa0a6"
        if ratio >= 0.75:  return "#1a9641"
        if ratio >= 0.50:  return "#a6d96a"
        if ratio >= 0.25:  return "#fdae61"
        return "#d7191c"

    def build_map(
        df: pd.DataFrame,
        center=(48.8566, 2.3522),
        zoom=12,
        highlight_stationcode: str | None = None,
        open_popup: bool = False,
        display_prediction: bool = False,
        bubbles_start_zoom: int = 19,
        cluster_radius: int = 120,
    ) -> folium.Map:
        m = folium.Map(location=center, zoom_start=zoom, control_scale=True,
                       tiles="cartodbpositron", prefer_canvas=True)
        mc = MarkerCluster(name="Stations",
                           options={"disableClusteringAtZoom": int(bubbles_start_zoom),
                                    "maxClusterRadius": int(cluster_radius),
                                    "showCoverageOnHover": False}).add_to(m)
        df = df.copy()
        for c in ["capacity","numbikesavailable","y_nb_pred","lat","lon"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

        def _to_int_safe(v):
            v = pd.to_numeric(v, errors="coerce"); return int(0 if pd.isna(v) else v)

        def _delta_badge(nb_act: int, nb_pred: int, cap: int) -> str:
            if cap <= 0: return "<span class='badge' style='background:#f3f4f6'>▬ 0%</span>"
            delta = 100.0 * (nb_pred - nb_act) / cap
            if abs(delta) < 0.5:  symbol, bg, fg = "▬", "#e5e7eb", "#111827"
            elif delta > 0:       symbol, bg, fg = "▲", "#dcfce7", "#166534"
            else:                 symbol, bg, fg = "▼", "#fee2e2", "#991b1b"
            return f"<span class='badge' style='background:{bg}; color:{fg}; border-color:rgba(0,0,0,.08)'>{symbol} {delta:+.0f}%</span>"

        for _, r in df.iterrows():
            name = (r.get("name") or "—").strip()
            code = str(r.get("stationcode", ""))
            lat  = float(r.get("lat")) if pd.notna(r.get("lat")) else 0.0
            lon  = float(r.get("lon")) if pd.notna(r.get("lon")) else 0.0
            cap     = _to_int_safe(r.get("capacity"))
            nb_act  = _to_int_safe(r.get("numbikesavailable"))
            nb_pred = _to_int_safe(r.get("y_nb_pred"))
            nb_show = nb_pred if display_prediction else nb_act
            ratio   = (nb_show / cap) if cap > 0 else np.nan
            col_ring = _ring_color(ratio)
            deg = int(max(0, min(1, 0 if pd.isna(ratio) else ratio)) * 360)
            size = 52 if highlight_stationcode == code else 44

            icon_html = f"""
            <div style="position:relative; width:{size}px; height:{size}px; filter: drop-shadow(0 1px 2px rgba(0,0,0,.25));">
              <div style="width:100%; height:100%; border-radius:50%;
                          background: conic-gradient({col_ring} {deg}deg, #e5e7eb 0deg);
                          display:flex; align-items:center; justify-content:center;
                          border:2px solid #263238; box-sizing:border-box;">
                <div style="width:{size-12}px; height:{size-12}px; border-radius:50%; background:#fff;
                            display:flex; align-items:center; justify-content:center;
                            font: 700 {int(size*0.42)}px/1 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif; color:#263238;">
                  {nb_show}
                </div>
              </div>
              <div style="position:absolute; left:50%; bottom:-12px; transform:translateX(-50%);
                          width:0; height:0; border-left:8px solid transparent; border-right:8px solid transparent;
                          border-top:12px solid #263238; opacity:.9;"></div>
            </div>
            """
            icon = folium.DivIcon(html=icon_html, icon_size=(size, size + 12), icon_anchor=(size/2, size + 12))
            popup_html = (
                f"<div style='font-family:Inter,system-ui,Segoe UI,Roboto,sans-serif;'>"
                f"<div style='font-weight:700;margin-bottom:.15rem'>{name}</div>"
                f"<div style='color:#6b7280;margin-bottom:.35rem'>#{code}</div>"
                f"<div style='display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:.35rem'>"
                f"<span style='padding:.2rem .5rem;border-radius:8px;background:#f3f4f6'>Actuel&nbsp;: <b>{nb_act}</b> / {cap}</span>"
                f"<span style='padding:.2rem .5rem;border-radius:8px;background:#eef2ff'>Prévision T+1h&nbsp;: <b>{nb_pred}</b></span>"
                f"{_delta_badge(nb_act, nb_pred, cap)}</div></div>"
            )
            folium.Marker((lat, lon), icon=icon, tooltip=f"{name} (#{code})",
                          popup=folium.Popup(popup_html, max_width=300,
                          show=(highlight_stationcode is not None and highlight_stationcode == code))).add_to(mc)
        folium.LayerControl(collapsed=True).add_to(m)
        return m

    center = st.session_state.get("map_center", (48.8566, 2.3522))
    zoom   = st.session_state.get("map_zoom", 12)
    highlight_code = st.session_state.get("map_highlight")

    m = build_map(
        df_with_pred, center=center, zoom=zoom,
        highlight_stationcode=str(highlight_code) if highlight_code else None,
        open_popup=bool(highlight_code), display_prediction=bool(display_pred),
    )
    st.components.v1.html(m.get_root().render(), height=560)
    st.markdown("<div style='color:#999;font-size:12px;margin-top:8px;'>OpenData en cache 5 min • Parquet rafraîchi automatiquement si >15 min (HF).</div>", unsafe_allow_html=True)

else:
    st.markdown("### 📊 Monitoring réseau")

    wx_badges = weather_badges_from_parquet(parquet_path)
    pf_badge = parquet_freshness_badge(parquet_path)

    render_badges(
        "<span class='badge'>Prévision : <b>T+1h</b></span>"
        + wx_badges
        + pf_badge
    )

    # ------------------------ KPIs réseau (actuel) ------------------------
    def compute_network_kpis(df_now: pd.DataFrame) -> dict:
        try:
            stations = int(df_now["stationcode"].nunique())
            bikes_total = int(pd.to_numeric(df_now["numbikesavailable"], errors="coerce").fillna(0).sum())
            docks_total = int(pd.to_numeric(df_now["numdocksavailable"], errors="coerce").fillna(0).sum())
            cap_total   = int(pd.to_numeric(df_now["capacity"], errors="coerce").fillna(0).sum())
            occ_pct     = round(100 * (bikes_total / cap_total), 1) if cap_total > 0 else 0.0
        except Exception:
            stations, bikes_total, docks_total, occ_pct = 0, 0, 0, 0.0
        return dict(stations=stations, bikes_total=bikes_total, docks_total=docks_total, occ_pct=occ_pct)

    kpis = compute_network_kpis(df_now)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f"<div class='kpi-row kpi-card'><div class='kpi-title'>Stations</div><div class='kpi-value'>{kpis['stations']}</div></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='kpi-row kpi-card'><div class='kpi-title'>Vélos dispo (actuel)</div><div class='kpi-value'>{kpis['bikes_total']}</div></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='kpi-row kpi-card'><div class='kpi-title'>Bornes libres (actuel)</div><div class='kpi-value'>{kpis['docks_total']}</div></div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='kpi-row kpi-card'><div class='kpi-title'>Occupation réseau</div><div class='kpi-value'>{kpis['occ_pct']} %</div></div>", unsafe_allow_html=True)

    # ------------------------ Préparation données ------------------------
    dfm = df_with_pred.copy()
    for c in ["capacity","numbikesavailable","numdocksavailable","y_nb_pred"]:
        dfm[c] = pd.to_numeric(dfm.get(c), errors="coerce").fillna(0)

    dfm["cap_eff"] = dfm[["capacity","numbikesavailable","numdocksavailable"]].max(axis=1)
    dfm = dfm[dfm["cap_eff"] > 0].copy()
    dfm["pct_now"]  = 100 * dfm["numbikesavailable"] / dfm["cap_eff"]
    dfm["pct_pred"] = 100 * dfm["y_nb_pred"]          / dfm["cap_eff"]
    dfm["delta_pts"] = dfm["pct_pred"] - dfm["pct_now"]

    # ------------------------ Contrôles ------------------------
    st.divider()
    st.subheader("Stations critiques (T+1h)")

    cc1, cc2, cc3, cc4 = st.columns([1.1, 1, 1, 1.2])
    with cc1:
        top_n = st.slider("Top N", min_value=5, max_value=30, value=10, step=1)
    with cc2:
        cap_min = st.number_input("Capacité min.", min_value=0, max_value=80, value=10, step=5)
    with cc3:
        metric = st.selectbox("Classement par", ["% prévu", "Δ points (prévu-actuel)"], index=0)
    with cc4:
        mode_val = st.radio("Afficher valeurs", ["% remplissage", "Nb vélos"], index=0, horizontal=True)

    dfv = dfm[dfm["cap_eff"] >= cap_min].copy()

    if metric == "% prévu":
        top_sat  = dfv.sort_values("pct_pred", ascending=False).head(top_n).copy()
        top_lack = dfv.sort_values("pct_pred", ascending=True ).head(top_n).copy()
        y_now, y_pred = "pct_now", "pct_pred"
    else:
        top_sat  = dfv.sort_values("delta_pts", ascending=False).head(top_n).copy()
        top_lack = dfv.sort_values("delta_pts", ascending=True ).head(top_n).copy()
        y_now, y_pred = "pct_now", "pct_pred"

    if mode_val == "Nb vélos":
        y_now, y_pred = "numbikesavailable", "y_nb_pred"

    def _shorten(s: str, max_chars=32) -> str:
        s = (s or "").strip()
        return (s[: max_chars - 1] + "…") if len(s) > max_chars else s

    def _plot_dumbbell(df: pd.DataFrame, title: str):
        n = len(df)
        h = max(0.55 * n, 3.8)
        fig, ax = plt.subplots(figsize=(8.5, h))
        ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
        ax.set_axisbelow(True)

        y = np.arange(n)
        names = df["name"].fillna("—").map(_shorten).tolist()
        x_now = pd.to_numeric(df[y_now], errors="coerce").fillna(0).values
        x_pred = pd.to_numeric(df[y_pred], errors="coerce").fillna(0).values
        deltas = x_pred - x_now

        if mode_val == "% remplissage":
            ax.set_xlim(0, 100)
        else:
            xmax = float(np.nanmax([x_now.max(), x_pred.max(), 1]))
            ax.set_xlim(0, np.ceil(xmax / 5) * 5)

        for i, (a, b, d) in enumerate(zip(x_now, x_pred, deltas)):
            ax.hlines(y=i, xmin=min(a, b), xmax=max(a, b), linewidth=3, alpha=0.8)
            ax.scatter(a, i, s=36, marker="o", zorder=3, label="Actuel" if i == 0 else None)
            ax.scatter(b, i, s=48, marker="D", zorder=3, label="Prévu"  if i == 0 else None)
            label = f"{b:.0f}% ({d:+.0f})" if mode_val == "% remplissage" else f"{int(b)} ({int(d):+d})"
            ax.text(b + 1.5, i, label, va="center", fontsize=8)

        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xlabel("% remplissage (vélos / capacité)" if mode_val == "% remplissage" else "Vélos disponibles (actuel/prévu)")
        ax.legend(loc="lower right", frameon=False, fontsize=9)
        plt.tight_layout()
        return fig

    tab1, tab2 = st.tabs(["Risque de saturation", "Risque de manque"])
    with tab1:
        st.markdown("**Top — proches de 100% (pleines)**")
        st.pyplot(_plot_dumbbell(top_sat, "Stations proches de 100% (pleines)"))
    with tab2:
        st.markdown("**Top — proches de 0% (vides)**")
        st.pyplot(_plot_dumbbell(top_lack, "Stations proches de 0% (vides)"))
