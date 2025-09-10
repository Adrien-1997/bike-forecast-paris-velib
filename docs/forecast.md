# Prévisions
**Échéance la plus récente** : `2025-09-10 10:00:00` (UTC)

## Top-10 stations à risque (faible nb vélos prévu T+1h)

| station | y_nb_pred | occ_ratio_pred |
|---|---:|---:|
| `22019` | 0 | 0.00 |
| `44018` | 0 | 0.00 |
| `45504` | 0 | 0.00 |
| `17125` | 0 | 0.00 |
| `18112` | 0 | 0.00 |
| `14026` | 0 | 0.00 |
| `32308` | 0 | 0.00 |
| `4201` | 0 | 0.00 |
| `31025` | 0 | 0.01 |
| `20103` | 0 | 0.01 |

## Observé vs Prédit (échantillon)

### Station `22019`

![obs vs pred](assets/figs/obs_pred_22019_T+1h.png)

### Station `44018`

![obs vs pred](assets/figs/obs_pred_44018_T+1h.png)

### Station `45504`

![obs vs pred](assets/figs/obs_pred_45504_T+1h.png)

### Station `42503`

![obs vs pred](assets/figs/obs_pred_42503_T+1h.png)

### Station `12129`

![obs vs pred](assets/figs/obs_pred_12129_T+1h.png)


## Qualité (in-sample, ordre de grandeur)
- MAE ≈ **1.16** vélos — RMSE ≈ **1.65** vélos
![residuals](assets/figs/residuals_T+1h.png)

## Importance des variables
![importance](assets/figs/feat_importance_T+1h.png)

> Remarque : ces métriques sont in-sample (à raffiner avec une validation temporelle TSSplit).