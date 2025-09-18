# Données — Présentation générale

Cette section regroupe tout ce qui concerne **les jeux de données publiés, leur contrat de schéma et la méthodologie** qui garantit leur qualité et leur reproductibilité.

- [**Exports**](./exports.md) — fichiers publiés, formats, clés, cadence de mise à jour.  
- [**Dictionnaire & schéma**](./dictionary.md) — description exhaustive des champs (types, unités, contraintes).  
- [**Méthodologie & licences**](./methodology.md) — pipeline de fabrication, règles de qualité, versionnage, usages et licences.

> **Granularité & temps**  
> Tous les exports sont fournis au **pas de 15 minutes**. Les timestamps `ts` sont **UTC (naïfs)** et arrondis au quart d’heure.

> **Jeux de données de référence**  
> - `events.parquet` — état du réseau par station (bikes, capacity, **occ**), avec méta et météo si disponibles.  
> - `perf.parquet` — cibles et prédictions alignées sur le temps **source T** : `y_true` (observé à T+h), `y_pred` (modèle, si injecté), `y_pred_baseline` (persistance), `horizon_min`.

---

## 1) Exports (ce que vous trouverez dans `exports.md`)

### Objectif
Lister **tout ce qui est publiqué** (fichiers de données et tables dérivées), avec le **contrat minimal** pour consommer ces fichiers sans surprise.

### Fichiers principaux
- **`docs/exports/events.parquet`**  
  - **Clé** : `(ts, station_id)` unique.  
  - **Colonnes canoniques (si présentes)** :  
    - `ts` *(UTC, 15 min)* — horodatage du **bin source**.  
    - `station_id` *(str)* — identifiant station stable.  
    - `bikes` *(int ≥ 0)* — vélos disponibles.  
    - `capacity` *(int ≥ 0)* — capacité estimée du dock.  
    - `occ` *(float ∈ [0,1])* — ratio d’occupation (ex. `bikes / capacity`, si capacité connue).  
    - **Métadonnées** : `name` *(str)*, `lat` *(float)*, `lon` *(float)*, `hour_utc` *(0–23)*.  
    - **Météo (optionnelles)** : `temp_C` *(°C)*, `precip_mm` *(mm)*, `wind_mps` *(m/s)*.
- **`docs/exports/perf.parquet`**  
  - **Clé** : `(ts, station_id)` unique.  
  - **Colonnes canoniques** :  
    - `ts` *(UTC, 15 min)* — **même bin source T** que `events`.  
    - `station_id` *(str)*.  
    - `y_true` *(float/int ≥ 0)* — cible observée à **T+h** (ramenée à T via `shift(-steps)`).  
    - `y_pred_baseline` *(float ≥ 0)* — **persistance** (valeur observée à T).  
    - `y_pred` *(float ≥ 0, optionnel)* — prédiction **modèle** alignée sur T (injectée après coup).  
    - `horizon_min` *(int > 0, ex. 60)* — horizon en minutes.

### Tables secondaires (générées pour la lecture & le monitoring)
Exportées sous `docs/assets/tables/` :
- `global_metrics.csv`, `error_by_horizon.csv`, `residuals_summary.csv`, `calibration_table.csv`,  
  `daily_error.csv`, `data_health.csv`, `psi_features.csv`, `feature_importance_proxy.csv`.

### Cadence & fraîcheur
- Ingestion et normalisation **toutes les 15 minutes** (ou au rythme de la source).  
- Les assets analytiques (tables/figures) sont régénérés **plusieurs fois par jour**.  
- La page Exports précisera **la date/heure du dernier build** et la **fenêtre couverte**.

### Garanties minimales
- Pas de **fusion en avance** (aucune fuite de futur) ; `perf.parquet` est strictement aligné **à T**.  
- **Aucune imputation lourde** dans les exports (ni interpolation) : les trous reflètent l’état réel de l’ingestion.  
- Clés `(ts, station_id)` **sans doublons** ; horodatages **arrondis 15 min**.

---

## 2) Dictionnaire & schéma (ce que vous trouverez dans `dictionary.md`)

### Objectif
Fournir un **contrat formel** : noms, types, unités, domaines de valeurs, contraintes, et **règles de validation**. Ce document permet d’intégrer les données sans lire le code.

### Contenu détaillé
- **Table “Champs canoniques”** (par fichier) avec :  
  - **Nom** (canonique) · **Type** (pandas/SQL) · **Unité / domaine** · **Description** · **Obligatoire ?** · **Valeur par défaut** · **Exemples**.  
- **Clés & unicité**  
  - `(ts, station_id)` est la **clé primaire** (unique) pour `events.parquet` et `perf.parquet`.  
- **Règles de cohérence**  
  - `bikes >= 0`, `capacity >= 0`, `0 ≤ occ ≤ 1`.  
  - Si `capacity` connue, alors `occ ≈ bikes / capacity` (tolérance).  
  - `y_true`, `y_pred`, `y_pred_baseline` **non négatifs** ; `horizon_min > 0`.  
  - **Horodatage** : `ts` en **UTC (naïf)**, arrondi *:00, :15, :30, :45*.  
- **Types & formats**  
  - `ts` → `timestamp without timezone` (SQL) / `datetime64[ns]` (pandas).  
  - `station_id` → `string` (pas d’entier imposé).  
  - `name` → texte libre (pas normalisé), `lat/lon` → `float`.  
- **Champs optionnels**  
  - `temp_C` (°C), `precip_mm` (mm), `wind_mps` (m/s) ; `hour_utc` (entier 0–23).  
- **Erreurs & valeurs interdites**  
  - Timestamps **dupliqués** par station : interdits.  
  - Timestamps non multiples de 15 minutes : interdits.  
  - Valeurs négatives sur le décompte : interdites.

### Validation & rapport
- La page détaillera le **jeu de règles** exécuté à chaque build (présence de colonnes, types, bornes), avec un **extrait du rapport** (passes/fails, premiers exemples d’anomalies).
- Lien vers les **tables de monitoring** pertinentes (`data_health.csv`, etc.) pour vérifier la conformité dans le temps.

---

## 3) Méthodologie & licences (ce que vous trouverez dans `methodology.md`)

### Objectif
Documenter **comment** sont produits les exports et **dans quel cadre** ils peuvent être utilisés.

### Méthodologie de fabrication (vue d’ensemble)
- **Normalisation** : renommage des colonnes source → **schéma canonique**, arrondi des timestamps à 15 min, harmonisation des types.  
- **Séparation** en deux vues :  
  - **`events.parquet`** (état instantané) ;  
  - **`perf.parquet`** (vérité à T+h ramenée à T, baseline, et prédictions si injectées).  
- **Cible & baseline** : `y_true` = `bikes` à **T+h** par station (via décalage négatif) ; `y_pred_baseline` = persistance (`bikes` à **T**).  
- **Injection modèle** : `y_pred` est ajoutée **après** normalisation, en garantissant l’alignement sur **T** et le **mapping station** robuste (nom/lat/lon → `station_id`).

### Qualité & monitoring
- Contrôles automatiques **à chaque build** : fraîcheur, complétude, schéma, anomalies (séries plates, doublons).  
- **Drift** suivi par PSI/K–S sur les features clés (si disponibles) et dérive de cible.  
- **Traçabilité** : horodatages de build, fenêtres utilisées, et métriques globales exposés dans `docs/assets/tables/`.

### Versionnage & compatibilité
- Le **contrat de schéma** est **stable** ; toute rupture sera annoncée via **bump de version** et release notes.  
- Les colonnes optionnelles peuvent **apparaître/disparaître** sans rompre le contrat (elles sont marquées comme *optionnelles* dans le dictionnaire).

### Licences & usages
- **Données dérivées** : les exports restent soumis à la **licence de la source originale** (respecter attribution/partage).  
- **Code** : licence du dépôt (ex. MIT) ; préciser droits et limites.  
- **Usages** : pas de tentative de **ré-identification** ; pas d’usage contraire aux CGU de la source ; indiquer l’**UTC** lors de toute republication de chronologies.

### Limites & transparence
- Les exports **reflètent l’état réel** de l’ingestion : absence d’imputation lourde ; les trous sont signalés, pas “réparés”.  
- Les capacités peuvent évoluer ; `occ` est une approximation lorsque la capacité n’est pas officiellement publiée.

---

## Valeur de la section “Données”
- **Interopérabilité** : schéma clair, colonnes stables, unités explicites.  
- **Fiabilité** : validation automatique, métriques de qualité publiées.  
- **Reproductibilité** : méthodologie documentée, alignement temporel strict, versionnage annoncé.

---

## Aller plus loin
- Page suivante : [**Exports**](./exports.md)  
- Ou explorez : [**Dictionnaire & schéma**](./dictionary.md) • [**Méthodologie & licences**](./methodology.md)
