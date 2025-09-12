# app/streamlit_app.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import requests
import math
import traceback, json, time
import folium
from folium.plugins import Search

import streamlit as st
from streamlit.components.v1 import html as st_html
from streamlit_js_eval import get_geolocation
#from streamlit_geolocation import streamlit_geolocation
# --- Repo paths
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

# Import interne (assure-toi de lancer avec PYTHONPATH=".")
import sys
sys.path.insert(0, str(ROOT))

from src.features import prepare_live_features  # aligne les features live -> artefact
from src.forecast import load_model_bundle      # charge (model, features) depuis joblib


# --- helpers proximité / géocodage ------------------------------------------


# --- Géoloc : garde seulement des positions plausibles ---
def _in_france(lat: float, lon: float) -> bool:
    # approx. métropole
    return (41.0 <= lat <= 51.5) and (-5.5 <= lon <= 9.8)

def _in_idf(lat: float, lon: float) -> bool:
    # gros rectangle Île-de-France pour recentrer confortablement
    return (48.0 <= lat <= 49.2) and (1.5 <= lon <= 3.8)




def _geoloc_html_fallback(timeout_ms: int = 6000):
    """Fallback HTML5 : renvoie (lat, lon, acc) ou None."""
    code = f"""
    <script>
      const opts = {{ enableHighAccuracy: true, timeout: {timeout_ms}, maximumAge: 0 }};
      function send(value) {{
        window.parent.postMessage({{ isStreamlitMessage: true, type: "streamlit:setComponentValue", value }}, "*");
      }}
      navigator.geolocation.getCurrentPosition(
        pos => {{
          const c = pos.coords;
          send({{ lat: c.latitude, lon: c.longitude, acc: c.accuracy }});
        }},
        err => {{
          send({{ error: String(err.code) + ":" + err.message }});
        }},
        opts
      );
    </script>
    """
    val = st_html(code, height=0)
    if val and isinstance(val, dict) and "lat" in val and "lon" in val:
        return float(val["lat"]), float(val["lon"]), float(val.get("acc") or 20.0)
    return None


def get_browser_geolocation(timeout_ms: int = 6000):
    """Retourne (lat, lon, acc) ou None. Tente JS, sinon None (pas d'HTML fallback ici)."""
    try:
        # versions récentes
        loc = get_geolocation(enable_high_accuracy=True, timeout=timeout_ms)
    except TypeError:
        # anciennes signatures
        try:
            loc = get_geolocation()
        except Exception:
            loc = None
    except Exception:
        loc = None

    if not loc:
        return None

    # normaliser structure
    c = loc.get("coords", loc)
    lat = c.get("latitude")
    lon = c.get("longitude")
    acc = c.get("accuracy", 20.0)
    if lat is None or lon is None:
        return None
    return float(lat), float(lon), float(acc)

# --- capture robuste + diagnostics ---

def capture_user_location():
    """Demande la position au navigateur et la stocke dans la session, mais rejette les positions douteuses (IP/US)."""
    if get_geolocation is None:
        st.warning("La géolocalisation navigateur n'est pas disponible.")
        return

    loc = get_geolocation()
    if not loc:
        st.error("Impossible d’obtenir votre position (aucune donnée renvoyée).")
        return

    c   = loc.get("coords", loc)
    lat = c.get("latitude")
    lon = c.get("longitude")
    acc = float(c.get("accuracy") or 999999)  # m

    if lat is None or lon is None:
        st.error("Position renvoyée sans latitude/longitude.")
        return

    lat = float(lat); lon = float(lon)

    # 1) coupe les positions manifestement IP (ex: centre USA) ou hors métropole
    if not _in_france(lat, lon):
        st.warning(f"Position hors France détectée ({lat:.4f}, {lon:.4f}) – ignorée.")
        return

    # 2) filtre la précision trop mauvaise (ex: > 50 km ⇒ souvent IP)
    if acc > 50000:
        st.warning(f"Position trop imprécise (~{int(acc)} m) – ignorée. Désactive VPN / active GPS.")
        return

    st.session_state["user_loc"] = (lat, lon)
    st.session_state["user_acc"] = acc
    st.toast(f"✅ Position captée : {lat:.5f}, {lon:.5f} (±{int(acc)} m)")

    # Recentrage doux : si on est en IDF, zoom fort; sinon zoom standard France
    if _in_idf(lat, lon):
        st.session_state["map_center"] = (lat, lon)
        st.session_state["map_zoom"] = 16
    else:
        st.session_state["map_center"] = (lat, lon)
        st.session_state["map_zoom"] = 12


def find_nearest_station(
    df: pd.DataFrame,
    user_lat: float,
    user_lon: float,
    min_bikes: int = 5,
    use_prediction: bool = False,
) -> dict | None:
    if df.empty or pd.isna(user_lat) or pd.isna(user_lon):
        return None

    dff = df.copy()
    # colonnes numériques propres
    dff["lat"] = pd.to_numeric(dff["lat"], errors="coerce")
    dff["lon"] = pd.to_numeric(dff["lon"], errors="coerce")
    dff["capacity"] = pd.to_numeric(dff.get("capacity"), errors="coerce").fillna(0).astype(int)
    dff["numbikesavailable"] = pd.to_numeric(dff.get("numbikesavailable"), errors="coerce").fillna(0).astype(int)
    if "y_nb_pred" in dff.columns:
        dff["y_nb_pred"] = pd.to_numeric(dff["y_nb_pred"], errors="coerce").fillna(0).astype(int)
    dff = dff.dropna(subset=["lat","lon"])
    if dff.empty:
        return None

    # colonne quantité (actuel ou prévision)
    qty_col = "y_nb_pred" if (use_prediction and "y_nb_pred" in dff.columns) else "numbikesavailable"

    # distances (haversine vectorisée)
    R  = 6371000.0
    φ1 = np.radians(float(user_lat))
    φ2 = np.radians(dff["lat"].to_numpy(float))
    dφ = np.radians(dff["lat"].to_numpy(float) - float(user_lat))
    dλ = np.radians(dff["lon"].to_numpy(float) - float(user_lon))
    a  = (np.sin(dφ/2.0)**2) + np.cos(φ1) * np.cos(φ2) * (np.sin(dλ/2.0)**2)
    a  = np.clip(a, 0.0, 1.0)
    dff["dist_m"] = 2.0 * R * np.arcsin(np.sqrt(a))

    # 1) rayon progressif + filtre min_bikes strict
    for radius in (250, 500, 800, 1200, 2000):
        cand = dff[(dff[qty_col] >= int(min_bikes)) & (dff["dist_m"] <= radius)].copy()
        if not cand.empty:
            cand.sort_values(["dist_m", qty_col], ascending=[True, False], inplace=True)
            return cand.iloc[0].to_dict()

    # 2) si rien : on relâche progressivement min_bikes (jusqu’à 1)
    for v in range(int(min_bikes) - 1, 0, -1):
        cand = dff[dff[qty_col] >= v].copy()
        if not cand.empty:
            cand.sort_values(["dist_m", qty_col], ascending=[True, False], inplace=True)
            return cand.iloc[0].to_dict()

    # 3) dernier recours : score distance - poids*vélo
    score = dff["dist_m"] - 30.0 * dff[qty_col]  # 30 m “bonus” par vélo
    return dff.iloc[int(np.argmin(score))].to_dict()

# ---------- UI CONFIG ----------
st.set_page_config(
    page_title="Prévisions Vélib’ Paris",
    page_icon="🚲",
    layout="wide",
)

# --- DEBUG switch (visible UI) ---
DEBUG = st.toggle("Mode debug", value=False, help="Affiche les diagnostics détaillés")

if DEBUG:
    st.markdown("##### Contexte navigateur")
    st_html("""
    <div id="probe" style="font:13px system-ui"></div>
    <script>
      (function(){
        const secure = window.isSecureContext;
        const inIframe = (function(){ try{ return window.self !== window.top; }catch(e){ return true; }})();
        const proto = window.location.protocol;
        const host = window.location.host;
        const msg = [
          "Protocol: " + proto,
          "Host: " + host,
          "Secure context: " + secure,
          "In iframe: " + inIframe
        ].join("<br>");
        document.getElementById("probe").innerHTML = msg;
      })();
    </script>
    """, height=0)


# --- Session defaults pour la carte ---
if "map_center" not in st.session_state:
    st.session_state["map_center"] = (48.8566, 2.3522)
if "map_zoom" not in st.session_state:
    st.session_state["map_zoom"] = 15
if "map_highlight" not in st.session_state:
    st.session_state["map_highlight"] = None

# --- Session defaults (évite KeyError) ---
if "user_loc" not in st.session_state:
    st.session_state["user_loc"] = None
if "user_acc" not in st.session_state:
    st.session_state["user_acc"] = None

# --- Derivés position utilisateur ---
_uloc = st.session_state.get("user_loc")
_uacc = st.session_state.get("user_acc")
have_user = bool(_uloc)
if have_user:
    u_lat, u_lon = _uloc


# --- UI tweaks (CSS) ---

st.markdown("""
<style>
/* Conteneur + titres */
.block-container {padding-top: 1.2rem; padding-bottom: .75rem; max-width: 1300px;}
h1, h2, h3 {letter-spacing:.2px; margin-top: .2rem;}

/* KPI cards */
.kpi-row {margin-bottom:.6rem;}
.kpi-card{background:#fff;border:1px solid #e7e7e7;border-radius:10px;padding:.75rem 1rem;text-align:center;}
.kpi-title{color:#6b7280;font-size:.80rem;margin-bottom:.25rem;}
.kpi-value{font-size:1.25rem;font-weight:700;}

/* Ligne de contrôles alignée sur la grille KPI */
.controls-row { margin:.25rem 0 1rem 0; }
.controls-row [data-testid="column"]{ display:flex; align-items:center; justify-content:center; gap:.5rem; }
.controls-row label { margin-bottom:0; }

/* Tailles homogènes */
.stButton > button{ height:42px !important; padding:0 .9rem; border-radius:.55rem; }
.stNumberInput input{ height:42px !important; text-align:center; width:88px !important; }
.stNumberInput div[data-baseweb="input"]{ min-height:42px; }
.stToggle{ transform:scale(1.0); }

/* Badges “Source / Heure / Horizon” */
.badges { display:flex; flex-wrap:wrap; gap:.5rem; margin:.25rem 0 0 0; }
.badge { display:inline-block; padding:.25rem .6rem; border-radius:999px; font-size:.80rem;
         border:1px solid #e5e7eb; background:#f9fafb; color:#374151; }

/* Carte : légende discrète */
.leaflet-control { font-size: 12px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
button[kind="secondaryFormSubmit"], button[kind="secondary"] {
  border-radius: 999px !important;
  padding: .25rem .6rem !important;
  font-size: .85rem !important;
}
</style>
""", unsafe_allow_html=True)


TITLE = "🚲 Prévisions Vélib’ Paris"
SUBTITLE = "Cartographie temps réel + prévisions ML (T+1h/T+3h/T+6h)"

# ---------- HELPERS DATA SOURCES ----------

# --- Geocoding (OpenStreetMap Nominatim) -------------------------------------
@st.cache_data(ttl=3600)
def geocode_address(query: str) -> tuple[float, float] | None:
    """
    Géocode une adresse via Nominatim. Renvoie (lat, lon) ou None si échec.
    """
    q = (query or "").strip()
    if not q:
        return None
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": q, "format": "json", "limit": 1, "accept-language": "fr"},
            headers={"User-Agent": "velib-app/1.0"},
            timeout=10,
        )
        r.raise_for_status()
        js = r.json()
        if not js:
            return None
        return float(js[0]["lat"]), float(js[0]["lon"])
    except Exception:
        return None
    
@st.cache_data(ttl=30)
def autocomplete_nominatim(q: str, limit: int = 6) -> list[dict]:
    q = (q or "").strip()
    if len(q) < 3:
        return []
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": q, "format": "json", "addressdetails": 1, "limit": limit, "accept-language": "fr"},
            headers={"User-Agent": "velib-app/1.0"},
            timeout=8,
        )
        r.raise_for_status()
        items = r.json() or []
        out = []
        for it in items:
            label = it.get("display_name") or it.get("name") or "Adresse"
            out.append({
                "label": label,
                "lat": float(it["lat"]),
                "lon": float(it["lon"]),
            })
        return out
    except Exception:
        return []
    

def _make_session() -> requests.Session:
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter
    s = requests.Session()
    retry = Retry(
        total=3, read=3, connect=3, backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD", "OPTIONS"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "velib-app/1.0 (+github.com/Adrien-1997)"})
    return s

def fetch_gbfs() -> pd.DataFrame:
    """
    Source principale (GBFS Smoove): station_status + station_information
    Retourne: df[stationcode, name, lat, lon, capacity, numbikesavailable, numdocksavailable]
    """
    URL_STATUS = "https://velib-metropole-opendata.smoove.pro/opendata/Velib_Metropole/station_status.json"
    URL_INFO   = "https://velib-metropole-opendata.smoove.pro/opendata/Velib_Metropole/station_information.json"
    s = _make_session()
    rs = s.get(URL_STATUS, timeout=20)
    ri = s.get(URL_INFO, timeout=20)
    rs.raise_for_status(); ri.raise_for_status()
    js_status = rs.json(); js_info = ri.json()
    stas = (js_status.get("data") or {}).get("stations") or []
    infs = (js_info.get("data") or {}).get("stations") or []

    df_st = pd.DataFrame([{
        "station_id": s.get("station_id"),
        "numbikesavailable": s.get("num_bikes_available"),
        "numdocksavailable": s.get("num_docks_available"),
        "last_reported": s.get("last_reported"),
    } for s in stas if s.get("station_id")])

    df_in = pd.DataFrame([{
        "station_id": i.get("station_id"),
        "stationcode": i.get("station_code") or i.get("station_id"),
        "name": i.get("name"),
        "lat": i.get("lat"),
        "lon": i.get("lon"),
        "capacity": i.get("capacity"),
    } for i in infs if i.get("station_id")])

    df = df_st.merge(df_in, on="station_id", how="inner")
    for c in ["capacity","numbikesavailable","numdocksavailable","lat","lon"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["stationcode","lat","lon"])
    df["fetched_at_utc"] = pd.Timestamp.utcnow()
    return df[["stationcode","name","lat","lon","capacity","numbikesavailable","numdocksavailable","fetched_at_utc"]]

def fetch_opendata_v1() -> pd.DataFrame:
    """
    Fallback (OpenData Paris v1)
    """
    URL = "https://opendata.paris.fr/api/records/1.0/search/?dataset=velib-disponibilite-en-temps-reel&rows=2000"
    s = _make_session()
    r = s.get(URL, timeout=25)
    r.raise_for_status()
    j = r.json()
    rows = []
    for rec in j.get("records", []):
        f = rec.get("fields", {})
        coord = f.get("coordonnees_geo")
        lat, lon = (coord or [None, None])
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

def fetch_synthetic(n: int = 30) -> pd.DataFrame:
    """
    Fallback ultime pour ne pas casser l'app en local offline.
    """
    import numpy as np
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

@st.cache_data(ttl=300)  # 5 minutes
def load_live_data_cached() -> Tuple[pd.DataFrame, str, str, pd.Timestamp]:
    """
    Renvoie (df, source_label, debug_reason, fetched_at_utc)
    """
    try:
        df = fetch_gbfs()
        return df, "GBFS", "", df["fetched_at_utc"].iloc[0]
    except Exception as e1:
        try:
            df = fetch_opendata_v1()
            return df, "OpenData", f"GBFS failed: {e1}", df["fetched_at_utc"].iloc[0]
        except Exception as e2:
            df = fetch_synthetic(60)
            return df, "fallback", f"GBFS failed: {e1} | OpenData failed: {e2}", df["fetched_at_utc"].iloc[0]

# ---------- MODEL ----------
@st.cache_resource
def load_model_cached(h: int):
    model_dir = str(MODELS_DIR)
    return load_model_bundle(h, model_dir=model_dir)

def predict_nbvelos(df_live: pd.DataFrame, horizon_hours: int = 1) -> np.ndarray:
    """
    Prédit nb vélos T+H. Fallback: persistance.
    """
    try:
        model, feats = load_model_cached(horizon_hours)
        if not feats:
            raise ValueError("features missing in artefact")
        X = prepare_live_features(df_live, feats)
        best_it = getattr(model, "best_iteration", None)
        y = model.predict(X, num_iteration=best_it)
        cap = pd.to_numeric(df_live.get("capacity"), errors="coerce").fillna(0).to_numpy(float)
        y = np.clip(np.round(y), 0, cap).astype(int)
        return y
    except Exception:
        # fallback: persistance (= vélos actuels)
        return pd.to_numeric(df_live.get("numbikesavailable"), errors="coerce").fillna(0).astype(int).to_numpy()

# ---------- MAP ----------


# --- helpers pour l’icône circulaire ---------------------------------
def _ring_color(ratio: float) -> str:
    # couleur de l’anneau selon le % de vélos (0..1)
    if pd.isna(ratio):
        return "#9aa0a6"    # gris
    if ratio >= 0.75:
        return "#1a9641"    # vert
    if ratio >= 0.50:
        return "#a6d96a"    # vert clair
    if ratio >= 0.25:
        return "#fdae61"    # orange
    return "#d7191c"        # rouge

def _make_divicon(nb: int, cap: int, size: int = 44) -> folium.DivIcon:
    # ratio pour l’anneau
    ratio = (nb / cap) if cap and cap > 0 else np.nan
    col   = _ring_color(ratio)
    deg   = int(max(0, min(1, 0 if pd.isna(ratio) else ratio)) * 360)

    # style : anneau + disque + nombre
    html = f"""
    <div class="velib-pin" style="
        position:relative; width:{size}px; height:{size}px;
        filter: drop-shadow(0 1px 2px rgba(0,0,0,.25));
        transform: translate(-50%, -100%);
    ">
      <div style="
        width:100%; height:100%; border-radius:50%;
        background:
          conic-gradient({col} {deg}deg, #e5e7eb 0deg);
        display:flex; align-items:center; justify-content:center;
        border:2px solid #263238;
        box-sizing:border-box;
      ">
        <div style="
          width:{size-12}px; height:{size-12}px; border-radius:50%;
          background:#ffffff; display:flex; align-items:center; justify-content:center;
          font: 700 {int(size*0.42)}px/1 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
          color:#263238;
        ">{nb}</div>
      </div>
      <!-- petite pointe -->
      <div style="
        position:absolute; left:50%; bottom:-8px; transform:translateX(-50%);
        width:0; height:0; border-left:8px solid transparent; border-right:8px solid transparent;
        border-top:12px solid #263238; opacity:.9;
      "></div>
    </div>
    """
    return folium.DivIcon(html=html, icon_size=(size, size), icon_anchor=(size/2, size))

# --- remplace le rendu des points dans build_map ---------------------
def build_map(df: pd.DataFrame, center=(48.8566, 2.3522), zoom=12,
              highlight_stationcode: str | None = None,
              open_popup: bool = False,
              user_location: tuple[float, float] | None = None,
              user_accuracy_m: float | None = None) -> folium.Map:
    m = folium.Map(location=center, zoom_start=zoom, control_scale=True, tiles="cartodbpositron")
    fg = folium.FeatureGroup(name="Stations").add_to(m)

    # Marqueurs stations (DivIcon avec anneau)
    for _, r in df.iterrows():
        name = (r.get("name") or "—").strip()
        code = str(r.get("stationcode", ""))
        lat, lon = float(r["lat"]), float(r["lon"])
        cap = int(pd.to_numeric(r.get("capacity"), errors="coerce") or 0)
        nb  = int(pd.to_numeric(r.get("numbikesavailable"), errors="coerce") or 0)
        y   = int(pd.to_numeric(r.get("y_nb_pred"), errors="coerce") or 0)

        popup = folium.Popup(
            f"<b>{name}</b> (#{code})<br>"
            f"Actuel : {nb} / {cap}<br>"
            f"Prévision : <b>{y}</b> (T+{int(r.get('horizon',1))}h)",
            max_width=260,
            show=(open_popup and highlight_stationcode == code),
        )

        icon = _make_divicon(nb=nb, cap=cap, size=48 if highlight_stationcode == code else 44)

        mkr = folium.Marker(
            location=(lat, lon),
            icon=icon,
            tooltip=f"{name} (#{code})",
            popup=popup,
        )
        # exposer un 'name' pour la barre de recherche
        mkr.options['name'] = name
        mkr.add_to(fg)

    # Barre de recherche (par nom)
    try:
        Search(
            layer=fg,
            search_label="name",
            placeholder="Rechercher une station…",
            collapsed=False,
            search_zoom=16
        ).add_to(m)
    except Exception:
        pass

    # --- Point bleu : position utilisateur ---
    if user_location:
        u_lat, u_lon = user_location
        acc = float(user_accuracy_m or 20.0)
        acc = max(20.0, acc)          # mini 20 m
        col = "#2563eb" if acc <= 50 else "#9aa0a6"  # bleu si précis, gris si large

        # point central
        folium.CircleMarker(
            location=(u_lat, u_lon),
            radius=4,
            color=col,
            fill=True,
            fill_opacity=1.0,
            weight=2,
            tooltip=f"Vous êtes ici (~{int(acc)} m)",
        ).add_to(m)

        # cercle de précision
        folium.Circle(
            location=(u_lat, u_lon),
            radius=acc,
            color=col,
            fill=True,
            fill_opacity=0.10 if acc <= 50 else 0.05,
            weight=1,
        ).add_to(m)

    return m


# ---------- UI LAYOUT ----------
# --- Header propre ---
st.markdown(f"### {TITLE}")
st.caption(SUBTITLE)

# --- Barre de commandes (horizon + refresh + badge source/heure plus bas) ---
# --- Contrôles (alignés sur 2 colonnes comme les KPI) ---
c1, c2, c3 = st.columns([1.2, 1.1, 1.1], gap="small")
with c1:
    try:
        horizon = st.segmented_control("Horizon", options=[1,3,6], selection_mode="single",
                                       default=1, label_visibility="collapsed")
    except Exception:
        horizon = st.selectbox("Horizon", options=[1,3,6], index=0, label_visibility="collapsed")
with c2:
    refresh = st.button("Actualiser")
with c3:
    st.write("")  # spacer

if refresh:
    load_live_data_cached.clear()

# Données live
df_raw, source_label, debug_reason, fetched_utc = load_live_data_cached()
try:
    fetched_local = pd.to_datetime(fetched_utc, utc=True).tz_convert("Europe/Paris")
except Exception:
    fetched_local = pd.Timestamp.utcnow().tz_localize("UTC").tz_convert("Europe/Paris")

user_acc = st.session_state.get("user_acc")
acc_badge = f"<span class='badge'>Précision : ~{int(user_acc)} m</span>" if user_acc else ""
st.markdown(
    f"<div class='badges'>"
    f"<span class='badge'>Source : <b>{source_label}</b></span>"
    f"<span class='badge'>Requête : {fetched_local.strftime('%d/%m %H:%M')}</span>"
    f"<span class='badge'>Horizon : T+{int(horizon)}h</span>"
    f"{acc_badge}"
    f"</div>",
    unsafe_allow_html=True,
)

# Timestamp joliment rendu (Europe/Paris)
try:
    fetched_local = pd.to_datetime(fetched_utc, utc=True).tz_convert("Europe/Paris")
except Exception:
    fetched_local = pd.Timestamp.utcnow().tz_localize("UTC").tz_convert("Europe/Paris")
st.markdown("")


# KPIs
try:
    stations = int(df_raw["stationcode"].nunique())
    bikes_total = int(pd.to_numeric(df_raw["numbikesavailable"], errors="coerce").fillna(0).sum())
    docks_total = int(pd.to_numeric(df_raw["numdocksavailable"], errors="coerce").fillna(0).sum())
    cap_total   = int(pd.to_numeric(df_raw["capacity"], errors="coerce").fillna(0).sum())
    occ_pct     = round(100 * (bikes_total / cap_total), 1) if cap_total > 0 else 0.0
except Exception:
    stations, bikes_total, docks_total, occ_pct = 0, 0, 0, 0.0


# Prédictions
y_pred = predict_nbvelos(df_raw, horizon_hours=int(horizon))
df_map = df_raw.copy()
df_map["y_nb_pred"] = y_pred
df_map["horizon"] = int(horizon)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"<div class='kpi-row kpi-card'><div class='kpi-title'>Stations</div><div class='kpi-value'>{stations}</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='kpi-row kpi-card'><div class='kpi-title'>Vélos dispo</div><div class='kpi-value'>{bikes_total}</div></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='kpi-row kpi-card'><div class='kpi-title'>Bornes libres</div><div class='kpi-value'>{docks_total}</div></div>", unsafe_allow_html=True)
with k4:
    st.markdown(f"<div class='kpi-row kpi-card'><div class='kpi-title'>Occupation réseau</div><div class='kpi-value'>{occ_pct} %</div></div>", unsafe_allow_html=True)



# --- Trouver une station la plus proche ---
st.markdown("<div class='finder-card'>", unsafe_allow_html=True)
st.markdown("### Trouver une station (≥ vélos)")

c1, c2, c3, c4, c5 = st.columns([2.4, 1.0, 1.0, 1.4, 1.2])
with c1:
    addr_text = st.text_input(
        "Adresse / lieu", 
        placeholder="Ex: 10 Rue de Rivoli, Paris", 
        key="addr_text"
    )

    # récupérer une éventuelle sélection précédente
    addr_pick = st.session_state.get("addr_pick")

    suggestions = autocomplete_nominatim(addr_text)
    if suggestions:
        st.markdown("<div style='margin-top:.25rem'></div>", unsafe_allow_html=True)
        for i, s in enumerate(suggestions):
            lab = s["label"]
            if st.button(lab, key=f"sugg_{i}"):
                st.session_state["addr_pick"] = (s["lat"], s["lon"])
                st.session_state["addr_label"] = lab
                # <-- centre la carte sur l'adresse choisie, même sans cliquer "M'y emmener"
                st.session_state["map_center"] = (s["lat"], s["lon"])
                st.session_state["map_zoom"] = 15
                st.session_state["map_highlight"] = None
                st.rerun()


    # optionnel : afficher le choix courant
    if st.session_state.get("addr_label"):
        st.caption(f"Sélection : {st.session_state['addr_label']}")

with c2:
    min_bikes = st.number_input("Vélos min.", min_value=1, max_value=50, value=5, step=1)
with c3:
    use_pred = st.toggle("Prévision", value=False, help="Filtrer sur la prévision T+H")
with c4:
    use_loc = st.button("📍 Ma position", key="btn_loc_finder")
    if use_loc:
        capture_user_location()
        if st.session_state.get("user_loc"):
            st.session_state["map_center"] = st.session_state["user_loc"]
            st.session_state["map_zoom"] = 16
            st.session_state["map_highlight"] = None
            st.rerun()

with c5:
    go = st.button("🗺️ M’y emmener", type="primary")  # <-- go EST défini ici

st.markdown("</div>", unsafe_allow_html=True)

# Déterminer le point de départ (adresse > position navigateur > centre Paris)
origin = None
addr_pick = st.session_state.get("addr_pick")

if addr_pick is not None:
    origin = addr_pick
    st.session_state["map_center"] = addr_pick
    st.session_state["map_zoom"] = 18   # <--- zoom plus fort
    st.session_state["map_highlight"] = None

elif addr_text.strip():
    geoc = geocode_address(addr_text)
    # après geocode_address(addr_text)
    if geoc:
        origin = geoc
        st.session_state["map_center"] = geoc
        st.session_state["map_zoom"] = 18   # <--- zoom plus fort
        st.session_state["map_highlight"] = None


user_lat, user_lon = origin if origin else (48.8566, 2.3522)

target = None
if go:
    if DEBUG:
        st.info(f"Recherche: origin={origin or 'PARIS centre'} • min_bikes={int(min_bikes)} • use_pred={bool(use_pred)}")

    target = find_nearest_station(
        df_map,
        user_lat=user_lat,
        user_lon=user_lon,
        min_bikes=int(min_bikes),
        use_prediction=bool(use_pred),
    )

    if target is None:
        st.warning("Aucune station trouvée avec ce seuil.")
    else:
        if DEBUG:
            st.json({
                "target": {
                    "name": target.get("name"),
                    "stationcode": target.get("stationcode"),
                    "dist_m": int(target.get("dist_m", -1)),
                    "numbikesavailable": int(target.get("numbikesavailable", -1)),
                    "y_nb_pred": int(target.get("y_nb_pred", -1)),
                }
            })
        st.success(
            f"✅ Station la plus proche : {target['name']} (#{target['stationcode']}) à ~{int(target['dist_m'])} m"
        )



# Base : les valeurs mémorisées (si disponibles), sinon Paris
user_loc = st.session_state.get("user_loc")
user_acc = st.session_state.get("user_acc")

center = st.session_state.get("map_center", (48.8566, 2.3522))
zoom   = st.session_state.get("map_zoom", 12)

m = build_map(
    df_map,
    center=center,
    zoom=zoom,
    highlight_stationcode=str(target["stationcode"]) if ('target' in locals() and target) else None,
    open_popup=('target' in locals() and target),
    user_location=user_loc,
    user_accuracy_m=user_acc,
)




st.components.v1.html(m.get_root().render(), height=680)



# Footer léger
st.markdown(
    "<div style='color:#999;font-size:12px;margin-top:8px;'>"
    "Astuce : utilisez le bouton « Actualiser maintenant » pour recharger les données (cache 5 min)."
    "</div>",
    unsafe_allow_html=True,
)