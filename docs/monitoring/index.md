# Monitoring — Présentation générale

Cette section répond à une question simple : **les données et le modèle sont-ils en bon état aujourd’hui ?**  
Elle se compose de trois pages complémentaires :

- [**Santé des données**](./data-health.md) — fraîcheur, complétude, schéma, anomalies d’ingestion.  
- [**Drift des données**](./drift.md) — dérive des distributions des variables (covariate/prior shift).  
- [**Santé du modèle**](./model-health.md) — performance dans le temps, calibration, biais, couverture prédictive.

> **Sources**  
> - `events.parquet` (pas 15 min) : états réseau par station (vélos/docks), utilisé pour qualité et drift.  
> - `perf.parquet` : cibles `y_true` et prédictions (`y_pred`, `y_pred_baseline` si dispo), utilisé pour santé du modèle.

> **Fenêtrage de référence**  
> Sauf mention contraire, les contrôles comparent une **fenêtre courante** (ex. 7 derniers jours) à une **référence** (ex. 28–90 jours précédents), en **UTC**.

---

## 1) Santé des données (ce que vous trouverez dans `data-health.md`)

### Objectif
Vérifier que le **pipeline d’ingestion** fournit des données **fraîches, complètes et conformes** au contrat attendu.

### Questions auxquelles la page répond
- Les données sont-elles **fraîches** (latence nominale respectée) ?  
- Quel est le **taux de complétude** (stations × timestamps) et où sont les trous ?  
- Le **schéma** (colonnes/types/unités) est-il conforme ? Y a-t-il des champs anormaux (valeurs négatives, hors bornes, constantes) ?  
- Observe-t-on des **ruptures** (station figée, série plate, duplication, horodatages non monotones) ?

### Indicateurs clés (KPIs)
- **Fraîcheur** : âge du dernier batch (P50/P95) vs SLO (ex. ≤ 5 min).  
- **Complétude** : part d’horodatages valides (global, par station, par heure).  
- **Latence d’ingestion** : distribution (médiane, P95) des délais `observation_ts → disponibilité`.  
- **Schéma & contraintes** : présence colonnes, types, bornes (`velos ≥ 0`, `docks ≥ 0`), somme plausible (`velos + docks ≈ capacité` si dispo).  
- **Anomalies** : % de doublons, % de séries plates > N pas, timestamps non strictement croissants.

### Méthodes & règles
- **Contrat de schéma** : liste canonique des colonnes + types + bornes souples (warnings) et dures (erreurs).  
- **Détection plateaux** : séquence ≥ K pas sans variation → alerte station.  
- **Contrôle temps** : pas manquants, pas dupliqués, dérive d’horloge (skew).  
- **Cartographie** : heatmap complétude (stations en lignes × temps en colonnes).

### Visualisations attendues
- **Jauges** fraîcheur & complétude (global + P95).  
- **Heatmap** de complétude par station/jour.  
- **Top stations problématiques** (latence, trous, séries plates).  
- **Rapport de validation de schéma** (table des checks pass/fail).

### Seuils/Alertes (par défaut, ajustables)
- Fraîcheur P95 > SLO×2 → **Alerte majeure**.  
- Complétude < 98 % sur J-1 → **Alerte**.  
- > 1 % de timestamps dupliqués → **Alerte**.  
- > N stations plates ≥ K pas → **Alerte**.

> **Limites**  
> La qualité “technique” n’implique pas la **représentativité** (couverte par la page *Drift*).

---

## 2) Drift des données (ce que vous trouverez dans `drift.md`)

### Objectif
Détecter la **dérive des distributions** entre une fenêtre courante et une fenêtre de référence, pour anticiper un **risque de dégradation** du modèle.

### Questions auxquelles la page répond
- Quelles **features** ont le plus dérivé ? La cible (`y_true`) a-t-elle changé de régime ?  
- La dérive est-elle **globale** ou concentrée sur certains **segments** (clusters, zones, heures) ?  
- Les dérives détectées sont-elles **persistantes** (structurées) ou **ponctuelles** (événement) ?

### Indicateurs & tests
- **PSI/CSI** par variable (binning robuste)  
  - Interprétation usuelle PSI : `< 0,1 : faible · 0,1–0,25 : modérée · > 0,25 : forte`.  
- **K–S** pour variables continues, **χ²** pour catégorielles.  
- **Δ moyenne/variance** normalisés (z-scores).  
- **Drift de cible** (prior shift) : évolution de la distribution de `y_true`.  
- **Drift conditionnel** : par **cluster de stations**, par **heure du jour**, par **arrondissement**.

### Méthodes
- **Fenêtrage** : référence glissante (ex. 28–90 j) vs courant (7 j).  
- **Stratification** : calcul des métriques **par segment** (clusters réseau, zones).  
- **Stabilité** : lissage temporel (EMA) pour éviter sur-réaction au bruit.

### Visualisations attendues
- **Barres** PSI/K–S top-N variables dérivées.  
- **Cartes** du drift agrégé par zone/station.  
- **Séries** temporelles de PSI global (EMA) pour suivre la tendance.  
- **Small multiples** : drift par cluster.

### Seuils/Alertes (par défaut, ajustables)
- PSI global (médiane des features clés) > 0,1 sur 3 jours consécutifs → **Alerte**.  
- PSI d’une feature critique > 0,25 sur 2 jours → **Alerte majeure**.  
- Drift de cible notable (Δ moyenne > 1 σ) → **Alerte**.

> **Limites**  
> Un drift **n’implique pas** forcément une dégradation du modèle (page *Santé du modèle* pour confirmation).

---

## 3) Santé du modèle (ce que vous trouverez dans `model-health.md`)

### Objectif
Surveiller la **performance dans le temps**, la **calibration** et la **couverture** des prédictions, pour décider d’un **ré-entraînement** ou d’un **fallback**.

### Questions auxquelles la page répond
- L’erreur (MAE/RMSE) se **dégrade-t-elle** ? Où (heures, clusters, zones) ?  
- Le **lift** vs baseline reste-t-il positif et stable ?  
- La **calibration** est-elle correcte globalement et par segments ?  
- La **couverture** (`% y_pred` disponible) est-elle conforme ?

### Indicateurs clés
- **MAE / RMSE / biais** par jour et par segments (heure, station, cluster, zone).  
- **Lift** vs persistance = `(MAE_base − MAE_model)/MAE_base`.  
- **Calibration** : pente/intercept `y_true ~ y_pred` (global & segments).  
- **Couverture prédictive** : part d’horodatages avec `y_pred`.  
- **Stabilité résiduelle** : variance des résidus, auto-corrélation.

### Visualisations attendues
- **Séries** MAE/RMSE/lift (global + segments).  
- **Cartes** d’erreur par station/zone.  
- **Courbes** de calibration (global + segments).  
- **Table** “Top N stations en dégradation”.

### Seuils/Politiques (par défaut, ajustables)
- **Dégradation** : MAE_7j − MAE_28j > 10 % **et** lift_7j < lift_28j − 5 pts → **Alerte**.  
- **Couverture** : < 99 % sur J-1 → **Alerte**.  
- **Calibration** : |pente−1| > 0,1 **ou** |intercept| > 0,5 → **Alerte**.  
- **Gating ré-entraînement** : 3 alertes “Dégradation” sur 10 jours → **Planifier retrain**.  
- **Fallback** : couverture < 95 % **ou** lift < 0 sur 3 jours → activer **baseline** le temps du correctif.

> **Limites**  
> La performance agrégée peut masquer des **poches locales** de dégradation → toujours lire les découpages.

---

## Gouvernance & opérations

- **SLO** (exemple) : fraîcheur P95 ≤ 5 min, complétude ≥ 98 %, couverture prédictive ≥ 99 %.  
- **Alertes** : seuils ci-dessus, agrégés quotidiennement (avec résumé par e-mail/Slack).  
- **Journal** : chaque alerte consigne date, segment, métrique, décision (no-op / retrain / correctif ingestion).  
- **Décisions** :  
  - *No-op* si dérive sans impact mesuré sur MAE/lift.  
  - *Correctif pipeline* si anomalies d’ingestion.  
  - *Ré-entraînement* si dégradation persistante **et** drift plausible.

---

## Aller plus loin
- Page suivante : [**Santé des données**](./data-health.md)  
- Ou explorez : [**Drift des données**](./drift.md) • [**Santé du modèle**](./model-health.md)
