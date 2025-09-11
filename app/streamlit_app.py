# app/streamlit_app.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import requests
import folium
from folium.plugins import Search, MarkerCluster
import math, requests

import streamlit as st
from streamlit.components.v1 import html as st_html
from streamlit_js_eval import get_geolocation
from streamlit_geolocation import streamlit_geolocation
# --- Repo paths
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

# Import interne (assure-toi de lancer avec PYTHONPATH=".")
import sys
sys.path.insert(0, str(ROOT))

from src.features import prepare_live_features  # aligne les features live -> artefact
from src.forecast import load_model_bundle      # charge (model, features) depuis joblib

# --- helpers proximité / géocodage ------------------------------------------

def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    φ1 = math.radians(lat1); φ2 = math.radians(lat2)
    dφ = math.radians(lat2 - lat1); dλ = math.radians(lon2 - lon1)
    a = (math.sin(dφ/2)*2) + math.cos(φ1)*math.cos(φ2)(math.sin(dλ/2)**2)
    a = min(1.0, max(0.0, a))
    return 2*R*math.asin(math.sqrt(a))

def geocode_address(q: str) -> tuple[float, float] | None:
    """Adresse → (lat, lon) via Nominatim."""
    url = "https://nominatim.openstreetmap.org/search"
    r = requests.get(url, params={"q": q, "format": "json", "limit": 1},
                     headers={"User-Agent": "velib-app/1.0"})
    r.raise_for_status()
    items = r.json()
    if not items:
        return None
    return float(items[0]["lat"]), float(items[0]["lon"])

# --- Géoloc navigateur (avec fallback) ---
try:
    from streamlit_geolocation import get_geolocation
except Exception:
    get_geolocation = None  # on gérera le fallback

def capture_user_location():
    """Demande la position au navigateur et la stocke dans la session."""
    if get_geolocation is None:
        st.warning("La géolocalisation navigateur n'est pas disponible. Installe streamlit-geolocation.")
        return
    loc = get_geolocation()  # pas d'arguments, et pas de timeout
    if loc and loc.get("latitude") is not None and loc.get("longitude") is not None:
        st.session_state["user_loc"] = (float(loc["latitude"]), float(loc["longitude"]))
        st.toast("✅ Position captée")
    else:
        st.info("Autorisez la géolocalisation dans votre navigateur.")

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
    dff["lat"] = pd.to_numeric(dff["lat"], errors="coerce")
    dff["lon"] = pd.to_numeric(dff["lon"], errors="coerce")
    dff["numbikesavailable"] = pd.to_numeric(dff.get("numbikesavailable"), errors="coerce").fillna(0).astype(int)
    if "y_nb_pred" in dff.columns:
        dff["y_nb_pred"] = pd.to_numeric(dff["y_nb_pred"], errors="coerce").fillna(0).astype(int)
    dff = dff.dropna(subset=["lat","lon"])
    if dff.empty:
        return None

    # Pick the quantity column
    qty_col = "y_nb_pred" if (use_prediction and "y_nb_pred" in dff.columns) else "numbikesavailable"

    # Vectorized distances (correct haversine)
    lat2 = dff["lat"].to_numpy(float)
    lon2 = dff["lon"].to_numpy(float)

    R  = 6371000.0
    φ1 = np.radians(float(user_lat))
    φ2 = np.radians(lat2)
    dφ = np.radians(lat2 - float(user_lat))
    dλ = np.radians(lon2 - float(user_lon))

    a = (np.sin(dφ / 2.0) * 2.0) + np.cos(φ1) * np.cos(φ2) * (np.sin(dλ / 2.0) * 2.0)
    a = np.clip(a, 0.0, 1.0)  # numeric safety
    dist = 2.0 * R * np.arcsin(np.sqrt(a))
    dff["dist_m"] = dist

    # Progressive radii, then global fallback
    for radius in (300, 600, 900, 1500):
        cand = dff[(dff[qty_col] >= int(min_bikes)) & (dff["dist_m"] <= radius)].copy()
        if not cand.empty:
            # rank by distance THEN by more bikes
            cand.sort_values(["dist_m", qty_col], ascending=[True, False], inplace=True)
            return cand.iloc[0].to_dict()

    # Last resort: best tradeoff score everywhere
    # score = distance(m) - 25 * bikes (i.e., 25 m “bonus” per available bike)
    score = dff["dist_m"] - 25.0 * dff[qty_col]
    idx = int(np.argmin(score))
    return dff.iloc[idx].to_dict()

# ---------- UI CONFIG ----------
st.set_page_config(
    page_title="Prévisions Vélib’ Paris",
    page_icon="🚲",
    layout="wide",
)

# --- UI tweaks (CSS) ---


st.markdown("""
<style>
/* ---------- Layout global ---------- */
.block-container {padding-top: 2rem; padding-bottom: .75rem; max-width: 1300px;}
h1, h2, h3 {letter-spacing:.2px;}

/* ---------- KPI cards ---------- */
.kpi-row {margin-bottom: .6rem;}
.kpi-card{
  background:#fff; border:1px solid #e7e7e7; border-radius:10px;
  padding:.75rem 1rem; text-align:center;
}
.kpi-title{color:#6b7280; font-size:.80rem; margin-bottom:.25rem;}
.kpi-value{font-size:1.25rem; font-weight:700;}

/* ---------- Ligne de contrôles (alignée et centrée) ---------- */
.controls-row { margin:.25rem 0 1rem 0; }
.controls-row [data-testid="column"],
[data-testid="column"]{           /* colonnes streamlit : centrage H/V */
  display:flex; align-items:center; justify-content:center;
}
.controls-row label{ margin-bottom:.25rem; }

/* Tailles homogènes */
.stButton > button{ height:46px !important; padding:0 .9rem; border-radius:.55rem; }
.stNumberInput input{ height:46px !important; text-align:center; width:80px !important; }
.stNumberInput div[data-baseweb="input"]{ min-height:46px; }
.stToggle{ transform:scale(1.0); }

/* ---------- Nettoyage styles anciens ---------- */
div.toolbar{ display:none !important; }   /* ancienne barre si encore présente */
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
.stToggle label { white-space: nowrap; margin-left: 6px; font-size: 0.9rem; vertical-align: middle; }
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
def color_for_ratio(r: float) -> str:
    # r = taux d’occupation (nb vélos / capacité)
    if pd.isna(r):
        return "#808080"  # inconnu
    if r >= 0.8:  # très plein -> vert foncé
        return "#1a9641"
    if r >= 0.6:
        return "#a6d96a"
    if r >= 0.4:
        return "#ffffbf"
    if r >= 0.2:
        return "#fdae61"
    return "#d7191c"      # presque vide -> rouge


def build_map(df: pd.DataFrame, center=(48.8566, 2.3522), zoom=12,
              highlight_stationcode: str | None = None,
              open_popup: bool = False) -> folium.Map:
    m = folium.Map(location=center, zoom_start=zoom, control_scale=True, tiles="cartodbpositron")
    fg = folium.FeatureGroup(name="Stations").add_to(m)

    for _, r in df.iterrows():
        name = (r.get("name") or "—").strip() if pd.notna(r.get("name")) else "—"
        code = str(r.get("stationcode", ""))
        lat, lon = float(r["lat"]), float(r["lon"])
        cap = int(pd.to_numeric(r.get("capacity"), errors="coerce") or 0)
        nb  = int(pd.to_numeric(r.get("numbikesavailable"), errors="coerce") or 0)
        y   = int(pd.to_numeric(r.get("y_nb_pred"), errors="coerce") or 0)
        occ = (nb / cap) if cap > 0 else np.nan
        col = color_for_ratio(occ)

        is_highlight = (highlight_stationcode is not None and code == str(highlight_stationcode))
        radius = 9 if is_highlight else 7
        opacity = 1.0 if is_highlight else 0.9
        border = 2 if is_highlight else 0

        popup = folium.Popup(
            f"<b>{name}</b> (#{code})<br>"
            f"Actuel : {nb} / {cap}<br>"
            f"Prévision : <b>{y}</b> (T+{int(r.get('horizon',1))}h)",
            max_width=260, show=(open_popup and is_highlight)
        )
        cm = folium.CircleMarker(
            location=(lat, lon),
            radius=radius,
            fill=True,
            fill_opacity=opacity,
            fill_color=col,
            color="#1f2937" if is_highlight else col,  # petit liseré si highlight
            weight=border,
            tooltip=f"{name} (#{code})",
            popup=popup,
        )
        # propriété 'name' pour la recherche
        cm.options.setdefault('name', name)
        cm.add_to(fg)

    # Barre de recherche par nom
    Search(layer=fg, search_label="name",
           placeholder="Rechercher une station…", collapsed=False, search_zoom=16).add_to(m)

    # Légende
    legend_html = """
    <div style="position: fixed; bottom: 25px; left: 25px; z-index: 9999;
      background: white; padding: 8px 10px; border-radius: 6px; border: 1px solid #ddd;
      font-size: 13px; box-shadow: 0 1px 4px rgba(0,0,0,0.1);">
      <b>Légende — Occupation actuelle</b><br>
      <span style="display:inline-block;width:12px;height:12px;background:#1a9641;margin-right:6px;border:1px solid #999"></span>0–20%<br>
      <span style="display:inline-block;width:12px;height:12px;background:#a6d96a;margin-right:6px;border:1px solid #999"></span>20–40%<br>
      <span style="display:inline-block;width:12px;height:12px;background:#ffffbf;margin-right:6px;border:1px solid #999"></span>40–60%<br>
      <span style="display:inline-block;width:12px;height:12px;background:#fdae61;margin-right:6px;border:1px solid #999"></span>60–80%<br>
      <span style="display:inline-block;width:12px;height:12px;background:#d7191c;margin-right:6px;border:1px solid #999"></span>80–100%<br>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    return m

# ---------- UI LAYOUT ----------
# --- Header propre ---
st.markdown(f"### {TITLE}")
st.caption(SUBTITLE)

# --- Barre de commandes (horizon + refresh + badge source/heure plus bas) ---
c1, c2, c_sp = st.columns([1.2, 1.0, 6.0])
with c1:
    try:
        horizon = st.segmented_control(
            "Horizon", options=[1, 3, 6], selection_mode="single", default=1,
            label_visibility="collapsed",
        )
    except Exception:
        horizon = st.selectbox("Horizon", options=[1,3,6], index=0, label_visibility="collapsed")
with c2:
    if st.button("Actualiser", type="secondary"):
        load_live_data_cached.clear()

# Données live (avec heure de requête)
df_raw, source_label, debug_reason, fetched_utc = load_live_data_cached()

# Affichage heure FR (jj/mm HH:MM)
try:
    fetched_local = pd.to_datetime(fetched_utc, utc=True).tz_convert("Europe/Paris")
except Exception:
    fetched_local = pd.Timestamp.utcnow().tz_localize("UTC").tz_convert("Europe/Paris")

st.markdown(
    f"<div class='toolbar'>"
    f"  <span class='badge'>Source : <b>{source_label}</b></span>"
    f"  <span class='badge'>Requête : {fetched_local.strftime('%d/%m %H:%M %Z')}</span>"
    f"  <span class='badge'>Horizon : T+{int(horizon)}h</span>"
    f"</div>",
    unsafe_allow_html=True,
)

# Timestamp joliment rendu (Europe/Paris)
try:
    fetched_local = pd.to_datetime(fetched_utc, utc=True).tz_convert("Europe/Paris")
except Exception:
    fetched_local = pd.Timestamp.utcnow().tz_localize("UTC").tz_convert("Europe/Paris")
st.markdown(
    f"*Source :* {source_label} &nbsp;•&nbsp; *Heure de requête :* {fetched_local.strftime('%Y-%m-%d %H:%M:%S %Z')}"
)

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
    addr = st.text_input("Adresse / lieu (optionnel)", placeholder="Ex: 10 Rue de Rivoli, Paris")
with c2:
    min_bikes = st.number_input("Vélos min.", min_value=1, max_value=50, value=5, step=1)
with c3:
    use_pred = st.toggle("Prévision", value=False, help="Filtrer sur la prévision T+H")
with c4:
    use_loc = st.button("📍 Ma position")
with c5:
    go = st.button("🗺️ M’y emmener", type="primary")  # <-- go EST défini ici

st.markdown("</div>", unsafe_allow_html=True)

# Déterminer le point de départ (adresse > position navigateur > centre Paris)
origin = None
if addr.strip():
    origin = geocode_address(addr)

if origin is None and use_loc:
    try:
        loc = streamlit_geolocation()
        if loc and "latitude" in loc and "longitude" in loc:
            origin = (float(loc["latitude"]), float(loc["longitude"]))
    except Exception:
        origin = None

user_lat, user_lon = origin if origin else (48.8566, 2.3522)

target = None
if go:  # <-- plus d'erreur ici
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
        st.success(f"✅ Station la plus proche : {target['name']} (#{target['stationcode']}) à ~{int(target['dist_m'])} m")

# Centrer la carte sur la cible si trouvée
if target:
    m = build_map(
        df_map,
        center=(float(target["lat"]), float(target["lon"])),
        zoom=16,
        highlight_stationcode=str(target["stationcode"]),
        open_popup=True,
    )
else:
    m = build_map(df_map, center=(48.8566, 2.3522), zoom=12)

st.components.v1.html(m.get_root().render(), height=680)



# Footer léger
st.markdown(
    "<div style='color:#999;font-size:12px;margin-top:8px;'>"
    "Astuce : utilisez le bouton « Actualiser maintenant » pour recharger les données (cache 5 min)."
    "</div>",
    unsafe_allow_html=True,
)