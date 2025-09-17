# Modèle — Présentation générale

Cette section documente **comment le modèle est entraîné, évalué et interprété**. Elle se compose de trois pages complémentaires :

- [**Performance & baseline**](./performance.md) — qualité prédictive, comparaisons et lift vs persistance.  
- [**Pipeline d’entraînement & features**](./pipeline.md) — données d’entrée, cibles, horizon, versions.  
- [**Explicabilité & calibration**](./explainability.md) — résidus, importance des variables, calibration & biais.

> **Cible & horizon (référence actuelle)**  
> La cible `y_true` est l’**occupation** (vélos disponibles) par station. L’horizon par défaut est **60 minutes** avec un pas temporel de **15 minutes** (4 pas).  
> Les évaluations sont **time-aware** (pas de fuite de données), avec découpes chronologiques et/ou validation à origine glissante.

> **Jeux de données utilisés**  
> - `perf.parquet` : séries `y_true` et, si présent, `y_pred` (modèle) et `y_pred_baseline` (persistance).  
> - `events.parquet` : états par station au pas de 15 minutes (supports features & diagnostics).

---

## 1) Performance & baseline (ce que vous trouverez dans `performance.md`)

### Objectif
Mesurer la **qualité des prévisions** du modèle et la situer **par rapport à une baseline** simple (persistance).

### Questions auxquelles la page répond
- Quelle est l’erreur moyenne **globale** (toutes stations, toutes heures) ?  
- Dans quels **segments** (heure, jour, station, cluster, zone) le modèle est-il le plus/moins performant ?  
- Quel est le **gain vs baseline** (lift) et comment évolue-t-il dans le temps ?

### Métriques principales
- **MAE** (Mean Absolute Error) — robustesse et lisibilité opérationnelle.  
- **RMSE** — pénalise davantage les gros écarts.  
- **ME (biais)** — moyenne des erreurs signée (sous/sur-prédiction).  
- **Coverage prédictif** — part d’horodatages pour lesquels une prédiction existe.  
- **Lift vs baseline** = `(MAE_baseline − MAE_modèle) / MAE_baseline` (positif = mieux que la persistance).  
- **R²** (optionnel, sur séries agrégées) — à manier avec prudence pour des données bornées/peu linéaires.

### Découpages & comparaisons
- **Par station** (top/bottom-10, distribution), **par cluster** (archétypes d’usage), **par heure du jour**, **semaine/week-end**, **par arrondissement/quartier**.  
- **Chronologique** : courbe MAE quotidienne/hebdomadaire, détection de dégradations.  
- **Capacité** : erreur normalisée par capacité estimée (si disponible) pour comparer des stations hétérogènes.

### Visualisations attendues
- **Observé vs prédit** (ex. séries 24 h / 7 j) sur échantillon de stations représentatives.  
- **Barres** MAE/RMSE par segment, **cartes** d’erreur, **sparklines** du lift.  
- **Distribution** des erreurs (histogramme/résidus).

### Lecture & limites
- La **persistance** (dernier état connu) est une baseline forte à court terme ; le lift est donc une mesure exigeante.  
- Les métriques agrégées peuvent masquer des comportements **station-spécifiques** (d’où l’analyse segmentée).

---

## 2) Pipeline d’entraînement & features (ce que vous trouverez dans `pipeline.md`)

### Objectif
Décrire précisément **d’où viennent les données**, **comment on fabrique les features**, **comment on entraîne** et **versionne** le modèle.

### Données d’entrée
- **Séries réseau** : vélos/docks disponibles par station (`events.parquet`).  
- **Calendrier** : heure/minute, jour de semaine, jours fériés, vacances (si disponibles).  
- **Météo** (optionnel) : température, pluie, vent (si intégrée et historisée).  
- **Géographie légère** (optionnelle) : arrondissement/quartier, altitude, distance centre.

### Construction des features (exemples typiques)
- **Retards (lags)** : valeurs H−15, −30, −45, −60… (pas d’info future).  
- **Fenêtres glissantes** : moyennes/medians/écarts-types sur 1–6 h, indicateurs de volatilité.  
- **Saisonnalité** : heure **sin/cos**, jour de semaine encodé, période scolaire.  
- **Interactions légères** : lags × heure, météo × heure (si pertinent).  
- **Capacité & normalisation** : ratios (`y/capacité`) lorsque la comparaison inter-stations est requise.

### Entraînement & validation
- **Découpes temporelles** : train → val → test **dans l’ordre du temps**.  
- **Validation à origine glissante** (rolling origin) pour évaluer la robustesse.  
- **Objectif** : minimiser MAE (primat opérationnel), contrôle RMSE/biais.  
- **Anti-fuite** : toutes les features utilisent **exclusivement** des informations ≤ t (aucun futur).

### Artefacts & versioning
- Modèle sérialisé (ex. `joblib`) + **signature** des features attendues.  
- **Version sémantique** (ex. `vX.Y.Z`) liée au schéma de features & à l’horizon.  
- Journal de **reproductibilité** : seed, plage temporelle d’entraînement, métriques de val/test.  
- **Planification** : ré-entraînement périodique (ex. quotidien/hebdo) ou à l’alerte monitoring.

### Déploiement & prédiction
- **Chargement** depuis l’artefact, vérification du **contrat de features** (colonnes, types, ordre).  
- Production de `y_pred` à chaque pas pour chaque station prévue, **timestampée à t** (et non t+h).

---

## 3) Explicabilité & calibration (ce que vous trouverez dans `explainability.md`)

### Objectif
Rendre les prévisions **intelligibles** (quelles variables comptent ? quand, où ?) et **fiables** (calibration, biais, incertitudes).

### Résidus & diagnostic
- **Résidus** `y_true − y_pred` : distribution, QQ-plot, autocorrélation.  
- **Hétéroscédasticité** : variance des résidus vs niveau d’occupation.  
- **Outliers/épisodes** : séquences d’erreurs anormalement longues (liées à ruptures d’ingestion, événements).

### Importance & explications
- **Permutation importance** (globale) sur échantillon **time-aware**.  
- **Ablation** par familles de features (lags, saisonnalité, météo) pour la **valeur incrémentale**.  
- **Profils moyens conditionnels** (PDP/ICE) sur variables clés.  
- **Segments** : importance et erreurs **par cluster de stations** (transparence sur où le modèle “comprend” mieux).

### Calibration & biais
- **Régression d’étalonnage** `y_true = α + β·y_pred` :  
  - **β ≈ 1** & **α ≈ 0** → bonne calibration moyenne.  
  - Pentes par **segments** (heure, cluster, capacité, zone) pour détecter des biais structurels.  
- **Erreur relative** par niveau d’occupation (bas/moyen/haut) — utile pour l’opérationnel.

### Incertitude (si activée)
- **Intervalles** par quantiles ou par **jackknife/bootstrap**.  
- **Coverage nominal vs empirique** (ex. 80 % nominal ↔ ~80 % observé).  
- Signalement des **stations à forte incertitude** (utile pour le monitoring).

### Visualisations attendues
- **Cartes** du biais par station/zone, **barres** d’importance, **PDP** pour 2–3 features clés, **courbes** de calibration globales et par segments.

### Lecture & limites
- L’explicabilité **décrit des associations**, pas des causalités.  
- La calibration moyenne peut être bonne tout en étant **mauvaise localement** : d’où l’analyse par segments.

---

## Valeur de la section “Modèle”
- **Opérationnel** : savoir quand/où la prévision est fiable, et de combien elle améliore la baseline.  
- **Ingénierie** : pipeline clair, versionné, reproductible.  
- **Confiance** : transparence sur **pourquoi** le modèle prédit ce qu’il prédit, et **comment** il se comporte selon les contextes.

---

## Aller plus loin
- Page suivante : [**Performance & baseline**](./performance.md)  
- Ou explorez : [**Pipeline d’entraînement & features**](./pipeline.md) • [**Explicabilité & calibration**](./explainability.md)
