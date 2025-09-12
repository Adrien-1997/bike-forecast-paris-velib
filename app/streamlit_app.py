# app/streamlit_app.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, List
import os, glob
import numpy as np
import pandas as pd
import requests
import folium
import streamlit as st

# -----------------------------------------------------------------------------
# PATHS & GLOBALS
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
FIGS_DIR = ROOT / "docs" / "assets" / "figs"   # graphes 72h (adaptable)
DEBUG = False

# Import interne (assure-toi de lancer avec PYTHONPATH=".")
import sys
sys.path.insert(0, str(ROOT))

from src.features import prepare_live_features
from src.forecast import load_model_bundle

# -----------------------------------------------------------------------------
# UI CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Prévisions Vélib’ Paris", page_icon="🚲", layout="wide")
st.session_state.setdefault("map_center", (48.8566, 2.3522))
st.session_state.setdefault("map_zoom", 15)
st.session_state.setdefault("map_highlight", None)

# CSS
st.markdown("""
<style>
/* Titres : respiration supplémentaire */
h2, h3 {
  margin-top: 1rem !important;   /* espace haut */
  margin-bottom: .6rem !important;
}

/* KPI cards : plus d’espace */
.kpi-row { margin: 1rem 0 !important; }
.kpi-card {
  background: var(--background-color);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 1rem 1.2rem;
  text-align: center;
  box-shadow: 0 1px 3px rgba(0,0,0,.08);
}
.kpi-title {
  color: var(--text-muted);
  font-size: .85rem;
  margin-bottom: .35rem;
}
.kpi-value {
  font-size: 1.35rem;
  font-weight: 700;
  color: var(--text-strong);
}

/* Badges : petits coussins + couleurs neutres */
.badges {
  display: flex;
  flex-wrap: wrap;
  gap: .8rem;            /* ⬅️ espace horizontal/vertical entre badges */
  margin: .5rem 0 1rem;  /* ⬅️ ajoute un peu d’espace autour du bloc */
}

.badge {
  display: inline-block;
  padding: .35rem .8rem;
  border-radius: 999px;
  font-size: .80rem;
  border: 1px solid var(--border-color);
  background: var(--badge-bg);
  color: var(--text-strong);
}


/* Variables couleurs clair/sombre */
:root {
  --background-color: #ffffff;
  --border-color: #e5e7eb;
  --text-muted: #6b7280;
  --text-strong: #111827;
  --badge-bg: #f9fafb;
}
@media (prefers-color-scheme: dark) {
  :root {
    --background-color: #1f2937;
    --border-color: #374151;
    --text-muted: #9ca3af;
    --text-strong: #f3f4f6;
    --badge-bg: #374151;
  }
}

.monitoring-fig img {
    max-width: 900px !important; /* taille max */
    margin: 0 auto !important;   /* centre l’image */
    display: block;              /* la force à se centrer */
}
            
/* Aligne le bouton avec les champs d'entrée */
.finder-card [data-testid="column"] {
  display: flex;
  align-items: flex-end;   /* ⬅️ aligne bas */
  justify-content: flex-start;
}
.finder-card .stButton > button {
  height: 42px !important;
  margin-bottom: 2px;      /* petit ajustement */
}
            




</style>
""", unsafe_allow_html=True)

TITLE = "🚲 Prévisions Vélib’ Paris"

# -----------------------------------------------------------------------------
# HELPERS — REQUÊTES DONNÉES (partagés)
# -----------------------------------------------------------------------------
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


def fetch_opendata_v1() -> pd.DataFrame:
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
    """Retourne (df, source_label, debug_reason, fetched_at_utc)."""
    try:
        df = fetch_opendata_v1()
        return df, "OpenData", "", df["fetched_at_utc"].iloc[0]
    except Exception as e2:
        df = fetch_synthetic(60)
        return df, "fallback", f"OpenData failed: {e2}", df["fetched_at_utc"].iloc[0]


# -----------------------------------------------------------------------------
# HELPERS — PRÉDICTIONS (partagés)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model_cached(h: int):
    return load_model_bundle(h, model_dir=str(MODELS_DIR))

def predict_nbvelos_t1h(df_live: pd.DataFrame) -> np.ndarray:
    """Prédit le nombre de vélos à T+1h. Fallback = persistance."""
    try:
        model, feats = load_model_cached(1)
        if not feats:
            raise ValueError("features missing in artefact")
        X = prepare_live_features(df_live, feats)
        best_it = getattr(model, "best_iteration", None)
        y = model.predict(X, num_iteration=best_it)
        cap = pd.to_numeric(df_live.get("capacity"), errors="coerce").fillna(0).to_numpy(float)
        y = np.clip(np.round(y), 0, cap).astype(int)
        return y
    except Exception:
        return pd.to_numeric(df_live.get("numbikesavailable"), errors="coerce").fillna(0).astype(int).to_numpy()

def compute_network_kpis(df_now: pd.DataFrame) -> dict:
    """KPIs réseau à partir des données actuelles."""
    try:
        stations = int(df_now["stationcode"].nunique())
        bikes_total = int(pd.to_numeric(df_now["numbikesavailable"], errors="coerce").fillna(0).sum())
        docks_total = int(pd.to_numeric(df_now["numdocksavailable"], errors="coerce").fillna(0).sum())
        cap_total   = int(pd.to_numeric(df_now["capacity"], errors="coerce").fillna(0).sum())
        occ_pct     = round(100 * (bikes_total / cap_total), 1) if cap_total > 0 else 0.0
    except Exception:
        stations, bikes_total, docks_total, occ_pct = 0, 0, 0, 0.0
    return dict(stations=stations, bikes_total=bikes_total, docks_total=docks_total, occ_pct=occ_pct)

# -----------------------------------------------------------------------------
# HELPERS — GEO, FINDER, MAP (spécifique page Carte)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def geocode_address(query: str) -> tuple[float, float] | None:
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
def autocomplete_nominatim(q: str, limit: int = 6) -> List[dict]:
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
            out.append({"label": label, "lat": float(it["lat"]), "lon": float(it["lon"])})
        return out
    except Exception:
        return []

def _ring_color(ratio: float) -> str:
    if pd.isna(ratio): return "#9aa0a6"
    if ratio >= 0.75:  return "#1a9641"
    if ratio >= 0.50:  return "#a6d96a"
    if ratio >= 0.25:  return "#fdae61"
    return "#d7191c"

def _make_divicon(nb_display: int, cap: int, size: int = 44, ratio_src: float | None = None) -> folium.DivIcon:
    ratio = ratio_src if ratio_src is not None else ((nb_display / cap) if cap and cap > 0 else np.nan)
    col   = _ring_color(ratio)
    deg   = int(max(0, min(1, 0 if pd.isna(ratio) else ratio)) * 360)
    html = f"""
    <div style="position:relative; width:{size}px; height:{size}px;
         filter: drop-shadow(0 1px 2px rgba(0,0,0,.25)); transform: translate(-50%, -100%);">
      <div style="width:100%; height:100%; border-radius:50%;
                  background: conic-gradient({col} {deg}deg, #e5e7eb 0deg);
                  display:flex; align-items:center; justify-content:center; border:2px solid #263238; box-sizing:border-box;">
        <div style="width:{size-12}px; height:{size-12}px; border-radius:50%; background:#fff;
                    display:flex; align-items:center; justify-content:center;
                    font: 700 {int(size*0.42)}px/1 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif; color:#263238;">
          {nb_display}
        </div>
      </div>
      <div style="position:absolute; left:50%; bottom:-8px; transform:translateX(-50%);
                  width:0; height:0; border-left:8px solid transparent; border-right:8px solid transparent;
                  border-top:12px solid #263238; opacity:.9;"></div>
    </div>
    """
    return folium.DivIcon(html=html, icon_size=(size, size), icon_anchor=(size/2, size))

def build_map(
    df: pd.DataFrame,
    center=(48.8566, 2.3522),
    zoom=12,
    highlight_stationcode: str | None = None,
    open_popup: bool = False,
    display_prediction: bool = False,
) -> folium.Map:
    m = folium.Map(location=center, zoom_start=zoom, control_scale=True, tiles="cartodbpositron")
    fg = folium.FeatureGroup(name="Stations").add_to(m)

    for _, r in df.iterrows():
        name = (r.get("name") or "—").strip()
        code = str(r.get("stationcode", ""))
        lat, lon = float(r["lat"]), float(r["lon"])
        cap = int(pd.to_numeric(r.get("capacity"), errors="coerce") or 0)
        nb_act  = int(pd.to_numeric(r.get("numbikesavailable"), errors="coerce") or 0)
        nb_pred = int(pd.to_numeric(r.get("y_nb_pred"), errors="coerce") or 0)

        nb_show = nb_pred if display_prediction else nb_act
        ratio_src = (nb_pred / cap) if (display_prediction and cap > 0) else (nb_act / cap if cap > 0 else np.nan)

        # --- Trend badge (▲ / ▼ / —) + % ---
        delta = nb_pred - nb_act
        if nb_act > 0:
            pct = int(round(100 * (delta / nb_act)))
            pct_txt = f"{'+' if delta>0 else ''}{pct}%"
        else:
            # si 0 vélo actuel : fallback sur variation absolue
            pct_txt = f"{'+' if delta>=0 else ''}{delta} vélos"

        if delta > 0:
            arrow = "▲"; col = "#16a34a"   # vert
        elif delta < 0:
            arrow = "▼"; col = "#dc2626"   # rouge
        else:
            arrow = "—"; col = "#6b7280"   # neutre

        trend_badge = (
            f"<span style=\"display:inline-flex;align-items:center;gap:.35rem;"
            f"font-weight:700;font-size:.85rem;color:{col};"
            f"padding:.20rem .55rem;border:1px solid {col}33;border-radius:999px;\">"
            f"{arrow} {pct_txt}</span>"
        )

        # Popup stylée
        popup_html = (
            f"<div style='font-family:Inter,system-ui,Segoe UI,Roboto,sans-serif;'>"
            f"<div style='font-weight:700;margin-bottom:.15rem'>{name}</div>"
            f"<div style='color:#6b7280;margin-bottom:.35rem'>#{code}</div>"
            f"<div style='display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:.35rem'>"
            f"<span style='padding:.2rem .5rem;border-radius:8px;background:#f3f4f6'>"
            f"Actuel: <b>{nb_act}</b> / {cap}</span>"
            f"<span style='padding:.2rem .5rem;border-radius:8px;background:#eef2ff'>"
            f"Prévision T+1h: <b>{nb_pred}</b></span>"
            f"</div>"
            f"{trend_badge}"
            f"</div>"
        )
        popup = folium.Popup(popup_html, max_width=300,
                             show=(open_popup and highlight_stationcode == code))

        # Icône : grossir et halo si surlignée
        size = 52 if highlight_stationcode == code else 44
        icon_html = ""
        # anneau comme avant
        ratio = ratio_src if ratio_src is not None else ((nb_show / cap) if cap and cap > 0 else np.nan)
        col_ring = _ring_color(ratio)
        deg = int(max(0, min(1, 0 if pd.isna(ratio) else ratio)) * 360)

        glow = "box-shadow:0 0 0 4px rgba(37,99,235,.25);" if highlight_stationcode == code else ""
        icon_html = f"""
        <div style="position:relative; width:{size}px; height:{size}px;
             filter: drop-shadow(0 1px 2px rgba(0,0,0,.25)); transform: translate(-50%, -100%);">
          <div style="width:100%; height:100%; border-radius:50%;
                      background: conic-gradient({col_ring} {deg}deg, #e5e7eb 0deg);
                      display:flex; align-items:center; justify-content:center; border:2px solid #263238;
                      box-sizing:border-box; {glow}">
            <div style="width:{size-12}px; height:{size-12}px; border-radius:50%; background:#fff;
                        display:flex; align-items:center; justify-content:center;
                        font: 700 {int(size*0.42)}px/1 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif; color:#263238;">
              {nb_show}
            </div>
          </div>
          <div style="position:absolute; left:50%; bottom:-8px; transform:translateX(-50%);
                      width:0; height:0; border-left:8px solid transparent; border-right:8px solid transparent;
                      border-top:12px solid #263238; opacity:.9;"></div>
        </div>
        """
        icon = folium.DivIcon(html=icon_html, icon_size=(size, size), icon_anchor=(size/2, size))

        mkr = folium.Marker(location=(lat, lon), icon=icon, tooltip=f"{name} (#{code})", popup=popup)
        mkr.add_to(fg)

    return m


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
    dff["capacity"] = pd.to_numeric(dff.get("capacity"), errors="coerce").fillna(0).astype(int)
    dff["numbikesavailable"] = pd.to_numeric(dff.get("numbikesavailable"), errors="coerce").fillna(0).astype(int)
    if "y_nb_pred" in dff.columns:
        dff["y_nb_pred"] = pd.to_numeric(dff["y_nb_pred"], errors="coerce").fillna(0).astype(int)
    dff = dff.dropna(subset=["lat","lon"])
    if dff.empty:
        return None

    qty_col = "y_nb_pred" if (use_prediction and "y_nb_pred" in dff.columns) else "numbikesavailable"

    R  = 6371000.0
    φ1 = np.radians(float(user_lat))
    φ2 = np.radians(dff["lat"].to_numpy(float))
    dφ = np.radians(dff["lat"].to_numpy(float) - float(user_lat))
    dλ = np.radians(dff["lon"].to_numpy(float) - float(user_lon))
    a  = (np.sin(dφ/2.0)**2) + np.cos(φ1) * np.cos(φ2) * (np.sin(dλ/2.0)**2)
    a  = np.clip(a, 0.0, 1.0)
    dff["dist_m"] = 2.0 * R * np.arcsin(np.sqrt(a))

    for radius in (250, 500, 800, 1200, 2000):
        cand = dff[(dff[qty_col] >= int(min_bikes)) & (dff["dist_m"] <= radius)].copy()
        if not cand.empty:
            cand.sort_values(["dist_m", qty_col], ascending=[True, False], inplace=True)
            return cand.iloc[0].to_dict()

    for v in range(int(min_bikes) - 1, 0, -1):
        cand = dff[dff[qty_col] >= v].copy()
        if not cand.empty:
            cand.sort_values(["dist_m", qty_col], ascending=[True, False], inplace=True)
            return cand.iloc[0].to_dict()

    score = dff["dist_m"] - 30.0 * dff[qty_col]
    return dff.iloc[int(np.argmin(score))].to_dict()

# -----------------------------------------------------------------------------
# ROUTER
# -----------------------------------------------------------------------------
PAGE = st.sidebar.radio("Navigation", ["Carte", "Monitoring réseau"], index=0)

# -----------------------------------------------------------------------------
# COMMON: CHARGE DONNÉES + PRÉDICTION (utilisé par les 2 pages)
# -----------------------------------------------------------------------------
# bouton refresh commun
if st.sidebar.button("Actualiser maintenant"):
    load_live_data_cached.clear()

df_now, source_label, debug_reason, fetched_utc = load_live_data_cached()
try:
    fetched_local = pd.to_datetime(fetched_utc, utc=True).tz_convert("Europe/Paris")
except Exception:
    fetched_local = pd.Timestamp.utcnow().tz_localize("UTC").tz_convert("Europe/Paris")

# prédiction T+1h (commune)
y_pred = predict_nbvelos_t1h(df_now)
df_with_pred = df_now.copy()
df_with_pred["y_nb_pred"] = y_pred
df_with_pred["horizon"] = 1

# badges communs (mode propre à chaque page)
def render_badges(extra: str = ""):
    st.markdown(
        f"<div class='badges'>"
        f"<span class='badge'>Source : <b>{source_label}</b></span>"
        f"<span class='badge'>Requête : {fetched_local.strftime('%d/%m %H:%M')}</span>"
        f"{extra}"
        f"</div>",
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# PAGE — CARTE
# -----------------------------------------------------------------------------
if PAGE == "Carte":
    st.markdown(f"### {TITLE}")

    # --- Sélecteur affichage (Actuel vs Prévision T+1h) ---
    try:
        display_mode = st.segmented_control(
            "Mode d'affichage",
            options=["Actuel", "Prévision T+1h"],
            selection_mode="single",
            default="Actuel",
            label_visibility="collapsed"
        )
    except Exception:
        display_mode = st.radio(
            "Mode d'affichage",
            options=["Actuel", "Prévision T+1h"],
            index=0,
            horizontal=True,
            label_visibility="collapsed"
        )
    display_pred = (display_mode == "Prévision T+1h")

    render_badges(
        f"<span class='badge'>Mode : <b>{'Prévision T+1h' if display_pred else 'Actuel'}</b></span>"
    )

    # --- Finder (adresse + filtre min vélos, calé sur le mode)
    st.markdown("<div class='finder-card'>", unsafe_allow_html=True)
    st.markdown("#### Trouver une station (≥ vélos)")

    c1, c2 = st.columns([2.6, 1.0])
    with c1:
        addr_text = st.text_input("Adresse / lieu", placeholder="Ex: 10 Rue de Rivoli, Paris", key="addr_text")
    with c2:
        min_bikes = st.number_input("Vélos min.", min_value=1, max_value=50, value=5, step=1)

    # Ligne séparée : bouton centré
    go = st.button("Trouver une station", type="primary")

    st.markdown("</div>", unsafe_allow_html=True)


    # Détermination de l’origine
    origin = None
    addr_pick = st.session_state.get("addr_pick")
    if addr_pick is not None:
        origin = addr_pick
        st.session_state["map_center"] = addr_pick
        st.session_state["map_zoom"] = 18
        st.session_state["map_highlight"] = None
    elif (addr_text or "").strip():
        geoc = geocode_address(addr_text)
        if geoc:
            origin = geoc
            st.session_state["map_center"] = geoc
            st.session_state["map_zoom"] = 18
            st.session_state["map_highlight"] = None

    user_lat, user_lon = origin if origin else (48.8566, 2.3522)

    # Recherche cible
    target = None
    if go:
        target = find_nearest_station(
            df_with_pred,
            user_lat=user_lat,
            user_lon=user_lon,
            min_bikes=int(min_bikes),
            use_prediction=bool(display_pred),
        )
        if target is None:
            st.warning("Aucune station trouvée avec ce seuil.")
        else:
            st.success(
                f"✅ Station la plus proche : {target['name']} "
                f"(#{target['stationcode']}) à ~{int(target['dist_m'])} m"
            )
            # mémoriser & zoomer centré
            st.session_state["map_highlight"] = str(target["stationcode"])
            st.session_state["map_center"] = (float(target["lat"]), float(target["lon"]))
            st.session_state["map_zoom"] = 17  # ⟵ zoom moins fort mais centré

            # --- Bouton "Copier les coordonnées" look primaire, sans rerun ---
            lat_s = f"{float(target['lat']):.6f}"
            lon_s = f"{float(target['lon']):.6f}"
            coords = f"{lat_s}, {lon_s}"

            # Bouton Streamlit désactivé (sert juste de style visuel, prend la bonne largeur/hauteur)
            st.button("Copier les coordonnées", type="primary", disabled=True, key="btn_copy_placeholder")

            # On injecte le vrai bouton HTML par-dessus (même texte, même place)
            st.markdown(
                f"""
                <script>
                const btn = window.parent.document.querySelector('button[data-testid="stBaseButton-secondary"][disabled]');
                if (btn) {{
                    btn.removeAttribute('disabled');
                    btn.onclick = function() {{
                        navigator.clipboard.writeText('{coords}');
                        alert("📋 Coordonnées copiées : {coords}");
                    }};
                }}
                </script>
                """,
                unsafe_allow_html=True
            )


    # Carte
    center = st.session_state.get("map_center", (48.8566, 2.3522))
    zoom   = st.session_state.get("map_zoom", 12)
    highlight_code = st.session_state.get("map_highlight")

    m = build_map(
        df_with_pred,
        center=center,
        zoom=zoom,
        highlight_stationcode=str(highlight_code) if highlight_code else None,
        open_popup=bool(highlight_code),
        display_prediction=bool(display_pred),
    )
    st.components.v1.html(m.get_root().render(), height=680)

    st.markdown(
        "<div style='color:#999;font-size:12px;margin-top:8px;'>"
        "Astuce : utilisez le bouton « Actualiser maintenant » dans la barre latérale pour recharger les données (cache 5 min)."
        "</div>",
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# PAGE — MONITORING RÉSEAU
# -----------------------------------------------------------------------------
else:

    def render_monitoring_page(
        df_now: pd.DataFrame,
        df_with_pred: pd.DataFrame,
        source_label: str,
        fetched_local: pd.Timestamp,
        figs_dir: Path,
    ):
        st.markdown("### 📊 Monitoring réseau")
        st.markdown(
            f"<div class='badges'>"
            f"<span class='badge'>Source : <b>{source_label}</b></span>"
            f"<span class='badge'>Requête : {fetched_local.strftime('%d/%m %H:%M')}</span>"
            f"<span class='badge'>Prévision : <b>T+1h</b></span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # KPIs
        kpis = compute_network_kpis(df_now)
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"<div class='kpi-row kpi-card'><div class='kpi-title'>Stations</div><div class='kpi-value'>{kpis['stations']}</div></div>", unsafe_allow_html=True)
        with k2:
            st.markdown(f"<div class='kpi-row kpi-card'><div class='kpi-title'>Vélos dispo (actuel)</div><div class='kpi-value'>{kpis['bikes_total']}</div></div>", unsafe_allow_html=True)
        with k3:
            st.markdown(f"<div class='kpi-row kpi-card'><div class='kpi-title'>Bornes libres (actuel)</div><div class='kpi-value'>{kpis['docks_total']}</div></div>", unsafe_allow_html=True)
        with k4:
            st.markdown(f"<div class='kpi-row kpi-card'><div class='kpi-title'>Occupation réseau</div><div class='kpi-value'>{kpis['occ_pct']} %</div></div>", unsafe_allow_html=True)

        # --- Historique 72h : empilé, centré, largeur maîtrisée ---
        st.markdown("### Historique (72h)")
        FIG_MAX_W = 900  # px (ajuste à 800–1000 si besoin)

        patterns = ["*72*.png", "*72*.jpg", "*72*.jpeg", "*72*.svg"]
        imgs: List[str] = []
        try:
            for patt in patterns:
                imgs.extend(sorted(glob.glob(str(figs_dir / patt)), key=os.path.getmtime, reverse=True))
        except Exception:
            imgs = []

        if not imgs:
            st.info(f"Aucun graphe 72h trouvé dans `{figs_dir}`. Vérifie les exports.")
        else:
            for p in imgs:
                # centre l’image en la plaçant dans une colonne au milieu
                _, mid, _ = st.columns([1, 2, 1])
                with mid:
                    st.image(p, width=FIG_MAX_W, caption=Path(p).name)

        # --- Top 10 risques (T+1h) ---
        st.markdown("### 🔴 Stations à risque à T+1h")

        dfm = df_with_pred.copy()
        for c in ["capacity","numbikesavailable","numdocksavailable","y_nb_pred"]:
            dfm[c] = pd.to_numeric(dfm.get(c), errors="coerce").fillna(0).astype(int)

        # Alerte diagnostic
        dfm["over_capacity"] = dfm["numbikesavailable"] > dfm["capacity"]
        n_over = int(dfm["over_capacity"].sum())
        if n_over:
            st.warning(f"{n_over} stations ont plus de vélos que de bornes (capacité obsolète / overflow).")

        # Capacité effective (pour tables/risques)
        dfm["cap_eff"]    = dfm[["capacity","numbikesavailable"]].max(axis=1)
        dfm["pred_bikes"] = np.minimum(dfm["y_nb_pred"], dfm["cap_eff"])
        dfm["pred_docks"] = (dfm["cap_eff"] - dfm["pred_bikes"]).clip(lower=0).astype(int)

        # Tri Top 10 avec cap_eff
        top_sat  = dfm.sort_values(["pred_docks","cap_eff"], ascending=[True, False]).head(10).copy()
        top_lack = dfm.sort_values(["pred_bikes","cap_eff"], ascending=[True, False]).head(10).copy()

        # Colonnes propres (on affiche la capacité effective)
        rename_map = {
            "stationcode": "Code",
            "name": "Station",
            "cap_eff": "Capacité",
            "numbikesavailable": "Vélos actuels",
            "numdocksavailable": "Bornes actuelles",
            "pred_bikes": "Vélos T+1h",
            "pred_docks": "Bornes T+1h",
        }
        for dfX in (top_sat, top_lack):
            dfX.rename(columns=rename_map, inplace=True)

        sat_cols  = ["Station","Code","Capacité","Bornes T+1h","Vélos T+1h","Bornes actuelles","Vélos actuels"]
        lack_cols = ["Station","Code","Capacité","Vélos T+1h","Bornes T+1h","Vélos actuels","Bornes actuelles"]


        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**Top 10 — Risque de saturation (T+1h)**")
            st.dataframe(
                top_sat.reindex(columns=[c for c in sat_cols if c in top_sat.columns]),
                use_container_width=True, hide_index=True
            )
        with c2:
            st.markdown("**Top 10 — Risque de manque (T+1h)**")
            st.dataframe(
                top_lack.reindex(columns=[c for c in lack_cols if c in top_lack.columns]),
                use_container_width=True, hide_index=True
            )

    # ⬇️  APPEL DE LA FONCTION  ⬇️
    render_monitoring_page(
        df_now=df_now,
        df_with_pred=df_with_pred,
        source_label=source_label,
        fetched_local=fetched_local,
        figs_dir=FIGS_DIR,  # doit être défini plus haut : ROOT / "docs" / "assets" / "figs"
    )
