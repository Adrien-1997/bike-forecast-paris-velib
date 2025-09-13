# Rapport de Modélisation — Vélib’ Forecast T+1h

## 🎯 Objectif
Prédire le nombre de vélos disponibles par station Vélib’ à **+1 heure**  
(horizon = 60 minutes, granularité de collecte/agrégation = 15 minutes).

---

## 📥 Données & Ingestion

**Sources principales :**
- API **Opendata Paris Vélib’** (v2.1, temps réel).
- Fallbacks : GBFS (Smoove) et Opendata v1.
- Sauvegarde brute → DuckDB `warehouse.duckdb` (table `velib_snapshots`).

**Pipeline d’ingestion :**
- Fréquence : toutes les 5 minutes.
- Colonnes normalisées : `ts_utc`, `stationcode`, `numbikesavailable`, `numdocksavailable`, `capacity`, `mechanical`, `ebike`, `lat`, `lon`.

---

## 🗂 Agrégation 15 min

Script : `src/aggregate.py`

- Fenêtre : 15 minutes, moyenne par station.
- Variables clés :
  - `nb_velos_bin`, `nb_bornes_bin`, `capacity_bin`, `occ_ratio_bin`
- Ajout météo (via API Open-Meteo) :
  - `temp_C`, `precip_mm`, `wind_mps`
- Ajout timestamp horaire : `hour_utc`
- Export :
  - `docs/exports/velib.parquet` (référence unique, historique 90 jours glissant)
  - `docs/exports/velib.csv`

---

## 🛠 Features

Construites via `src/features.py` et `src/cal_features.py`.

### Occupation
- `nb_velos_bin`, `nb_bornes_bin`, `capacity_bin`, `occ_ratio_bin`

### Lags (par station)
- `lag_nb_{1,2,3,4,8,16}b`
- `lag_occ_{1,2,3,4,8,16}b`

### Rolling / Tendances
- `roll_nb_4b`, `roll_nb_8b`
- `roll_occ_4b`, `roll_occ_8b`
- `trend_nb_4b`, `trend_occ_4b`

### Météo
- `temp_C`, `precip_mm`, `wind_mps`

### Calendrier
- `hour`, `dow`, `is_weekend`, `is_rush_am`, `is_rush_pm`, `is_holiday`
- Encodage circulaire : `hour_sin`, `hour_cos`

---

## 🤖 Modèle

Script : `src/forecast.py`

- **Algo** : LightGBM Regressor
- **Cible** : `y_nb` = vélos dispo à T+60 min (+4 bins)
- **Split temporel** : 80 % train / 20 % valid
- **Params** :
  - `n_estimators=1000`
  - `learning_rate=0.08`
  - `subsample=0.9`, `colsample_bytree=0.9`
  - `early_stopping=50` sur RMSE
- **Artefact** : `models/lgb_nbvelos_T+60min.joblib`

---

## 📊 Résultats Offline

- **MAE** (Mean Absolute Error) : erreur moyenne (en nb vélos).
- **RMSE** (Root Mean Squared Error) : plus sensible aux grosses erreurs.
- Exemple (validation set) : MAE ≈ 2.3 vélos | RMSE ≈ 3.8 vélos


*(les valeurs exactes dépendent de la période d’entraînement et du jeu de données)*

---

## 📈 Monitoring Online

Suivi continu via `tools/generate_monitoring.py` + GitHub Pages.

### Métriques
- MAE et RMSE glissants (24h, 7j)
- Erreur relative moyenne (% occupation)
- % stations avec erreur > seuil (ex. > 5 vélos)

### Drift
- **Feature drift** : PSI sur `occ_ratio_bin`, `temp_C`, `precip_mm`, `wind_mps`
- **Target drift** : distribution de `nb_velos_bin` observé
- Spatial drift possible (heatmap erreurs par quartier)

### Intégrité
- Stations sans données (drop API)
- Capacités nulles ou incohérentes

---

## 🗂 Schéma d’Architecture

```mermaid
flowchart TD
  A[Ingestion 5 min] --> B[DuckDB snapshots]
  B --> C[Aggregate 15 min]
  C -->|docs/exports/velib.parquet| D[Features (lags, météo, calendrier)]
  D --> E[Forecast LGBM T+1h]
  E --> F[Monitoring (metrics, drift)]
  F --> G[MkDocs static site]
