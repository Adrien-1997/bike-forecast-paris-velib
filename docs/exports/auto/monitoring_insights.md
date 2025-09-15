# Points clés — Monitoring (données & modèle)

- **Données manquantes** — bikes: 0.0%, capacity: 0.0%.
- **Drift (PSI)** — top: capacity (PSI 0.235) ; bikes (PSI 0.044) ; occ (PSI 0.009).
- **Features dominantes (proxy corrélation)** : occ, hour_cos, hour_sin, dow.
- **MAE quotidien (dernier jour)** : nan (cf. `mon_error_trend.png`).

### Règles & seuils (recommandations)
- **SLO** : surveiller MAE et RMSE par horizon ; alerte si dérive >+20% sur 3 jours.
- **Réentraînement** : déclencher si **PSI ≥ 0.2** sur une feature clé **ou** MAPE ↑ soutenue.
- **Sanity checks** : clamp [0, capacity], entrées hors-plage, valeurs manquantes critiques.
