# Points clés — Performance des prévisions

- **MAE**: 4.67 — **RMSE**: 7.34 au meilleur horizon (60 min).
- **MAPE**: 8497520.0% ; **sMAPE**: 51.6% (indicatifs).
- **Obs vs Préd** agrégé (`mon_pred_vs_true.png`) : vérifie l’alignement temporel.
- **Focus station** (`obs_vs_pred_station_24h.png`) : lisibilité des sous/sur-prédictions locales.
- **MAE heure×jour** (`errors_hour_x_dow.png`) : créneaux où le modèle est fragile.
- **Résidus** (`residual_hist.png`) & **biais horaire** (`bias_over_time.png`) : dérives systématiques.
- **Calibration** (`calibration_plot.png`) : cohérence niveaux prédits vs réalisés.
