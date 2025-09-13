# Objectif & Modèle

## Objectif
Prédire `nb_velos_bin` à **+60 minutes** par station.

## Features
- Occupation station: `nb_velos_bin`, `nb_bornes_bin`, `capacity_bin`, `occ_ratio_bin`
- Lags: `lag_nb_{1,2,3,4,8,16}`, `lag_occ_{...}`
- Rolling/Trend: `roll_nb_4b`, `roll_occ_8b`, pente courte (4 bins)
- Météo: `temp_C`, `precip_mm`, `wind_mps`
- Calendrier: `hour`, `dow`, `is_weekend`, `is_rush_am/pm`, `is_holiday`, `hour_sin`, `hour_cos`

## Modèle
- Algo: **LightGBM**
- Cible: `y_nb_pred = nb_velos_bin (T+1h)`
- Split: 80% train / 20% valid (ordre temporel)
- Early stopping: RMSE/MAE
- Artefact: `models/lgb_nbvelos_T+60min.joblib`
