# Vélib’ Paris — Forecast & Monitoring

Ce projet fournit :  
- **Monitoring temps réel** : carte interactive des stations (snapshot DB).  
- **Historique & KPI** : tendances d’occupation et métriques agrégées.  
- **Prévisions** *(WIP)* : nombre de vélos disponibles par station (T+1h, T+3h, T+6h).  

## Pipeline

Ingestion (GBFS / Opendata) → DuckDB (snapshots) → Agrégat horaire (+ météo)  
  ↓  
Exports Parquet/CSV  
  ↓  
Rapport (MkDocs) & App Streamlit (use case)  

- **DB** : `warehouse.duckdb`  
- **Exports** : `exports/velib_hourly.parquet|csv`  
- **Rapport** : GitHub Pages (`docs/*`)  
- **Use case** : Streamlit (reco utilisateur / opérations)  

## Liens rapides

- 🗺️ [Carte temps réel](results.md)  
- 📈 [Historique](history.md)  
- 🔮 [Prévisions](forecast.md)  

## Données & Features

- **Station** : `stationcode`, `name`, `capacity`, `lat/lon`  
- **Disponibilités** : `nb_velos`, `nb_bornes`, `occ_ratio`  
- **Météo** : `temp_C`, `precip_mm`, `wind_mps`  
- **Calendrier (Europe/Paris)** : heure locale (`hour`), jour semaine (`dow`), week-end (`is_weekend`), jour férié France (`is_holiday_fr`), composantes saisonnières (Fourier jour/semaine)  
- **Temps** : lags (`t-1h`, `t-2h`, `t-24h`), rollings (3h, 6h, 24h)  
- **Target modèle** : `y_nb` (nb vélos futurs), dérivé ensuite en `occ_ratio_pred`.  

## Objectif

Fournir un pipeline complet pour :  
1. **Surveiller** l’état du réseau Vélib’ (DB + carte temps réel).  
2. **Analyser** les dynamiques historiques (occupation, tendances, météo).  
3. **Prédire** le nombre de vélos disponibles par station dans les heures à venir.  
4. **Déployer** ces insights sous forme de **rapport statique** (GitHub Pages) et d’**app interactive** (Streamlit).  
