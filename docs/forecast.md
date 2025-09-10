# Prévisions
**Échéance la plus récente** : `2025-09-09 12:00:00` (UTC)

## Top-10 stations à risque (faible nb vélos prévu T+1h)

| station | y_nb_pred | occ_ratio_pred |
|---|---:|---:|
| `32308` | 0 | 0.00 |
| `32601` | 0 | 0.00 |
| `31203` | 0 | 0.00 |
| `45504` | 0 | 0.14 |
| `15018` | 0 | 0.00 |
| `20016` | 0 | 0.01 |
| `4201` | 0 | 0.02 |
| `44018` | 0 | 0.01 |
| `20037` | 0 | 0.01 |
| `31024` | 0 | 0.01 |

## Observé vs Prédit (échantillon)

### Station `32308`

![obs vs pred](assets/figs/obs_pred_32308_T+1h.png)

### Station `32601`

![obs vs pred](assets/figs/obs_pred_32601_T+1h.png)

### Station `31203`

![obs vs pred](assets/figs/obs_pred_31203_T+1h.png)

### Station `42503`

![obs vs pred](assets/figs/obs_pred_42503_T+1h.png)

### Station `12129`

![obs vs pred](assets/figs/obs_pred_12129_T+1h.png)


## Qualité (in-sample, ordre de grandeur)
- MAE ≈ **1.13** vélos — RMSE ≈ **1.61** vélos
![residuals](assets/figs/residuals_T+1h.png)

## Importance des variables
![importance](assets/figs/feat_importance_T+1h.png)

> Remarque : ces métriques sont in-sample (à raffiner avec une validation temporelle TSSplit).