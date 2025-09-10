# Prévisions
**Échéance la plus récente** : `2025-09-10 09:00:00` (UTC)

## Top-10 stations à risque (faible nb vélos prévu T+1h)

| station | y_nb_pred | occ_ratio_pred |
|---|---:|---:|
| `15039` | 0 | 0.00 |
| `18112` | 0 | 0.00 |
| `44018` | 0 | 0.00 |
| `20035` | 0 | 0.00 |
| `20027` | 0 | 0.00 |
| `20019` | 0 | 0.00 |
| `15047` | 0 | 0.00 |
| `15041` | 0 | 0.00 |
| `16108` | 0 | 0.00 |
| `18137` | 0 | 0.00 |

## Observé vs Prédit (échantillon)

### Station `15039`

![obs vs pred](assets/figs/obs_pred_15039_T+1h.png)

### Station `18112`

![obs vs pred](assets/figs/obs_pred_18112_T+1h.png)

### Station `44018`

![obs vs pred](assets/figs/obs_pred_44018_T+1h.png)

### Station `42503`

![obs vs pred](assets/figs/obs_pred_42503_T+1h.png)

### Station `12129`

![obs vs pred](assets/figs/obs_pred_12129_T+1h.png)


## Qualité (in-sample, ordre de grandeur)
- MAE ≈ **1.15** vélos — RMSE ≈ **1.63** vélos
![residuals](assets/figs/residuals_T+1h.png)

## Importance des variables
![importance](assets/figs/feat_importance_T+1h.png)

> Remarque : ces métriques sont in-sample (à raffiner avec une validation temporelle TSSplit).