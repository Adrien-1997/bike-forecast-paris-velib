# Vélib’ Paris — Batch Forecast

> Prédictions d’occupation des stations à l’heure (pipeline batch). Données temps réel Paris Data, features calendaires, baseline LightGBM.

<div class="kpis">
  <div class="kpi"><div class="label">Snapshots</div><div class="value">22628</div></div>
  <div class="kpi"><div class="label">Stations</div><div class="value">1453</div></div>
  <div class="kpi"><div class="label">Dernière maj (Paris)</div><div class="value">—</div></div>
</div>

[:material-chart-line: Résultats](results.md){ .md-button }
[:material-heart-pulse: Monitoring](monitoring.md){ .md-button .md-button--secondary }

!!! tip "Exports"
    - [Prévision 24h (CSV)]({{ base_url }}/exports/velib_forecast_24h.csv){ target=_blank }
    - [Occupations horaires (sample CSV)]({{ base_url }}/exports/occ_hourly_sample.csv){ target=_blank }

**Stack rapide**
- Ingestion snapshots → agrégation horaire  
- Features : calendaires (+ météo)  
- Modèle : LightGBM baseline → 24 h rolling forecast
