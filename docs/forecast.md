# Prévisions
**Échéance la plus récente** : `2025-09-09 11:00:00` (UTC)

## Top-10 stations à risque (faible nb vélos prévu T+1h)

| station | y_nb_pred | occ_ratio_pred |
|---|---:|---:|
| `20027` | 0 | 0.00 |
| `32601` | 0 | 0.00 |
| `20020` | 0 | 0.01 |
| `20035` | 0 | 0.01 |
| `20117` | 0 | 0.01 |
| `13203` | 0 | 0.01 |
| `48010` | 0 | 0.02 |
| `21505` | 0 | 0.02 |
| `18024` | 0 | 0.01 |
| `20103` | 0 | 0.02 |

## Observé vs Prédit (échantillon)

### Station `20027`

![obs vs pred](assets/figs/obs_pred_20027_T+1h.png)

### Station `32601`

![obs vs pred](assets/figs/obs_pred_32601_T+1h.png)

### Station `20020`

![obs vs pred](assets/figs/obs_pred_20020_T+1h.png)

### Station `21301`

![obs vs pred](assets/figs/obs_pred_21301_T+1h.png)

### Station `4110`

![obs vs pred](assets/figs/obs_pred_4110_T+1h.png)


## Qualité (in-sample, ordre de grandeur)
- MAE ≈ **1.05** vélos — RMSE ≈ **1.50** vélos
![residuals](assets/figs/residuals_T+1h.png)

## Importance des variables
![importance](assets/figs/feat_importance_T+1h.png)

> Remarque : ces métriques sont in-sample (à raffiner avec une validation temporelle TSSplit).