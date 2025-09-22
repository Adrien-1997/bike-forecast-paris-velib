from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict

from nicegui import ui, app  # type: ignore

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
APP_TITLE = "Vélib’ — +5 min forecast"
PARIS = (48.8566, 2.3522)
DEFAULT_ZOOM = 13


# -----------------------------------------------------------------------------
# PLACEHOLDER DATA
# -----------------------------------------------------------------------------
def fake_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def placeholder_network_snapshot() -> Dict[str, Any]:
    return {
        "updated_at": fake_now_iso(),
        "stations": [
            {
                "id": 1,
                "name": "Châtelet",
                "lat": 48.8588,
                "lon": 2.3470,
                "bikes": 2,
                "docks": 18,
                "status": "open",
                "forecast_5min": {"bikes": 4, "delta": +2},
            },
            {
                "id": 2,
                "name": "République",
                "lat": 48.8674,
                "lon": 2.3630,
                "bikes": 12,
                "docks": 8,
                "status": "open",
                "forecast_5min": {"bikes": 10, "delta": -2},
            },
        ],
    }


# -----------------------------------------------------------------------------
# FASTAPI ENDPOINT
# -----------------------------------------------------------------------------
@app.get("/api/network")
async def api_network() -> Dict[str, Any]:
    return placeholder_network_snapshot()


# -----------------------------------------------------------------------------
# PAGE
# -----------------------------------------------------------------------------
@ui.page("/")
def main_page() -> None:
    # Leaflet CSS/JS
    ui.add_head_html(
        f"""
        <link rel="stylesheet"
              href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
          :root {{ --header-h: 68px; }}
          body {{ margin: 0; }}
          .content {{ height: calc(100vh - var(--header-h)); }}
          #map {{ width: 100%; height: 100%; }}
        </style>
        """
    )

    # Header
    with ui.header().classes("sticky top-0 z-50"):
        with ui.row().classes("items-center justify-between w-full p-2"):
            ui.label(APP_TITLE).classes("text-lg font-medium")
            last_update = ui.label("Updated: —")

    # Map container only
    with ui.element("div").classes("content"):
        ui.html('<div id="map"></div>')

    # Inject JavaScript separately
    ui.add_body_html(
        f"""
        <script>
          const map = L.map('map').setView([{PARIS[0]}, {PARIS[1]}], {DEFAULT_ZOOM});
          L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 19,
            attribution: '&copy; OpenStreetMap'
          }}).addTo(map);

          const markers = new Map();

          function colorForBikes(bikes) {{
            if (bikes >= 10) return '#22c55e';
            if (bikes >= 4) return '#eab308';
            return '#ef4444';
          }}

          function markerIcon(color) {{
            return L.divIcon({{
              html: `<div style="width:16px;height:16px;border-radius:50%;background:${{color}};border:2px solid white;"></div>`,
              iconAnchor: [8, 8]
            }});
          }}

          function upsertStation(s) {{
            const key = String(s.id);
            const col = colorForBikes(s.bikes);
            const html = `<b>${{s.name}}</b><br/>Now: ${{s.bikes}} bikes<br/>+5min: ${{s.forecast_5min?.bikes}}`;
            if (markers.has(key)) {{
              const m = markers.get(key);
              m.setIcon(markerIcon(col));
              m.setLatLng([s.lat, s.lon]);
              m.bindPopup(html);
            }} else {{
              const m = L.marker([s.lat, s.lon], {{ icon: markerIcon(col) }}).addTo(map);
              m.bindPopup(html);
              markers.set(key, m);
            }}
          }}

          function applyNetworkPayload(payload) {{
            if (!payload || !Array.isArray(payload.stations)) return;
            payload.stations.forEach(upsertStation);
            const ts = payload.updated_at || '';
            window.dispatchEvent(new CustomEvent('network-updated', {{ detail: {{ ts }} }}));
          }}

          window.__applyNetwork = applyNetworkPayload;
        </script>
        """
    )

    # Python → JS bridge
    def push_map_data() -> None:
        payload = placeholder_network_snapshot()
        ui.run_javascript(f"window.__applyNetwork({json.dumps(payload)});")
        last_update.set_text(f"Updated: {payload['updated_at']}")

    # Initial push
    push_map_data()


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title=APP_TITLE, reload=False)
