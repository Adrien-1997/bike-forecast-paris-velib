# Vélib’ Forecast — Monitoring

Badges clés (glissants) :

<span class="metric-badge">MAE 24h: …</span>
<span class="metric-badge">RMSE 24h: …</span>
<span class="metric-badge">MAE 7j: …</span>
<span class="metric-badge">RMSE 7j: …</span>

- **Objectif** : prédire le nombre de vélos disponibles à **T+1h** par station (pas 15 min).
- **Modèle** : LightGBM (régression), split temporel 80/20, early stopping.
- **Données** : API Opendata Paris, jointures météo (Open-Meteo), features calendrier.

→ Lire le **[rapport complet](project/report.md)** et la **[Vue d’ensemble du monitoring](monitoring/index.md)**.
