# Monitoring API — routes JSON de supervision

Ce dossier regroupe **toutes les routes FastAPI dédiées au monitoring** du projet Vélib’ Forecast.  
Elles ne recalculent rien : elles exposent uniquement les artefacts JSON produits par les jobs `service/jobs/*` et stockés sur GCS.

Répertoire concerné :

```text
apps/api/routes/monitoring/
├─ network_overview.py      # /monitoring/network/overview/...
├─ network_dynamics.py      # /monitoring/network/dynamics/...
├─ network_stations.py      # /monitoring/network/stations/...
├─ model_performance.py     # /monitoring/model/performance/...
├─ model_explainability.py  # /monitoring/model/explainability/...
├─ data_health.py           # /monitoring/data/health/...
├─ data_drift.py            # /monitoring/data/drift/...
├─ data_freshness.py        # /monitoring/data/freshness
└─ intro.py                 # /monitoring/intro...
```

---

## 1. Principe général

- Tous ces modules lisent leurs fichiers sous **`GCS_MONITORING_PREFIX`** (env),
  par exemple :

  ```text
  GCS_MONITORING_PREFIX = "gs://.../velib/monitoring"
  ```

- Chaque route construit une URI du type :

  ```text
  gs://.../monitoring/<page>/<section>/<latest|timestamp>/<document>.json
  ```

- Les helpers communs (_split_gs, _gcs_read_json, _json_sanitize, _proxy_json…) :
  - téléchargent le JSON depuis GCS,
  - remplacent NaN / ±Inf par `null` pour garantir un JSON valide,
  - renvoient une `JSONResponse` avec des headers `Cache-Control` adaptés.

- Plusieurs routes acceptent un paramètre de **time‑travel** `?at=` :

  - `?at=latest` ou omis → snapshot courant,
  - `?at=YYYY-MM-DDTHH-MM-SSZ` → snapshot daté (aligné sur les jobs batch),
  - valeurs invalides → repli sur `latest` (évite le path traversal).

---

## 2. Network — Overview

**Fichier :** `network_overview.py`  
**Prefix API :** `/monitoring/network/overview`

Alimenté par le job `build_network_overview.py`, qui écrit :

```text
<MON>/network/overview/<latest|TS>/
  kpis.json
  snapshot_distribution.json
  today_curve.json
  ref_median_curve.json
  kpis_today_vs_lags.json
  snapshot_map.json
  stations_tension.json
```

Endpoints :

- `GET /monitoring/network/overview/available`  
  → liste les documents disponibles + rappel du time‑travel.

- `GET /monitoring/network/overview/{doc}?at=...`  
  → sert le JSON correspondant (avec sanitation NaN/Inf).

`doc` ∈ :

- `kpis` : KPIs globaux réseau (coverage, tension, etc.).
- `snapshot_distribution` : distribution de l’état réseau sur le dernier snapshot.
- `today_curve` : courbes temporelles de la journée courante.
- `ref_median_curve` : courbes de référence (médiane historique).
- `kpis_today_vs_lags` : comparaison “aujourd’hui vs J‑1 / J‑7…”.
- `snapshot_map` : structure de carte (stations, occ_ratio, classes de tension).
- `stations_tension` : liste des stations en tension (pénurie / saturation).

---

## 3. Network — Dynamics

**Fichier :** `network_dynamics.py`  
**Prefix API :** `/monitoring/network/dynamics`

Alimenté par le job `build_network_dynamics.py` :

```text
<MON>/network/dynamics/<latest|TS>/
  heatmaps_profiles.json
  hourly_pen_sat.json
  episodes.json
  by_zone.json
  tension_by_station.json
  regularity_today.json
```

Endpoints :

- `GET /monitoring/network/dynamics/available`
- `GET /monitoring/network/dynamics/{doc}?at=...`

`doc` ∈ :

- `heatmaps_profiles` : heatmaps et profils horaires globaux.
- `hourly_pen_sat` : pénurie/saturation par heure.
- `episodes` : épisodes de tension identifiés (durée, intensité…).
- `by_zone` : agrégats par zone spatiale / cluster.
- `tension_by_station` : indicateurs détaillés par station.
- `regularity_today` : régularité du réseau sur la journée courante.

---

## 4. Network — Stations

**Fichier :** `network_stations.py`  
**Prefix API :** `/monitoring/network/stations`

Alimenté par `build_network_stations.py` :

```text
<MON>/network/stations/<latest|TS>/
  kpis.json
  centroids.json
  pca_scatter.json
  pca_circle.json
  stats7.json

# Legacy global (optionnel)
<MON>/network/stations.json
```

Endpoints :

- `GET /monitoring/network/stations/available`
- `GET /monitoring/network/stations/{doc}?at=...`
- `GET /monitoring/network/stations` (legacy, renvoie stations.json si présent)

`doc` ∈ :

- `kpis` : KPIs station / cluster.
- `centroids` : centres des clusters stations.
- `pca_scatter` : coordonnées PCA des stations (scatter plot).
- `pca_circle` : cercle de corrélation PCA.
- `stats7` : statistiques condensées (7 jours, etc.).

---

## 5. Model — Performance

**Fichier :** `model_performance.py`  
**Prefix API :** `/monitoring/model/performance`

Alimenté par `build_model_performance.py` :

```text
<MON>/model/performance/<latest|TS>/manifest.json
<MON>/model/performance/<latest|TS>/h{H}/
  kpis.json
  daily_metrics.json
  by_hour.json
  by_dow.json
  by_station.json
  by_cluster.json
  lift_curve.json
  hist_residuals.json
  station_timeseries.json
```

Endpoints :

- `GET /monitoring/model/performance/available`  
  → liste les horizons dispo + docs.

- `GET /monitoring/model/performance/manifest?h=15&at=...`  
  → sert `manifest.json` pour un horizon donné.

- `GET /monitoring/model/performance/{doc}?h=15&at=...`  
  → sert un document par horizon (en minutes).

`doc` ∈ :

- `kpis` : métriques globales modèle (MAE, RMSE, etc.).
- `daily_metrics` : trajectoire jour par jour.
- `by_hour` : performance par heure de la journée.
- `by_dow` : performance par jour de semaine.
- `by_station` : métriques détaillées par station.
- `by_cluster` : agrégats par cluster (si fournis côté job).
- `lift_curve` : lift vs baseline.
- `hist_residuals` : histogrammes de résidus.
- `station_timeseries` : séries temporelles performance station (focus).

---

## 6. Model — Explainability

**Fichier :** `model_explainability.py`  
**Prefix API :** `/monitoring/model/explainability`

Alimenté par `build_model_explainability.py` :

```text
<MON>/model/explainability/<latest|TS>/h{H}/
  overview.json
  residuals.json
  calibration.json
  uncertainty.json
  feature_importance.json
```

Endpoints :

- `GET /monitoring/model/explainability/available`
- `GET /monitoring/model/explainability/manifest?h=15&at=...`  
  → renvoie `overview.json` comme manifest par horizon.

- `GET /monitoring/model/explainability/{doc}?h=15&at=...`

`doc` ∈ :

- `overview` : synthèse globale d’explicabilité.
- `residuals` : décorticage des résidus par dimensions.
- `calibration` : calibration du modèle vs réalité.
- `uncertainty` : indicateurs d’incertitude (si produits).
- `feature_importance` : importance des features (SHAP/GAIN…).

---

## 7. Data — Health

**Fichier :** `data_health.py`  
**Prefix API :** `/monitoring/data/health`

Alimenté par `build_data_health.py` :

```text
<MON>/data/health/<latest|TS>/
  kpis.json
  station_health.json
  coverage_by_hour.json
  anomalies.json             # fusion de plusieurs sources (legacy)
  alerts.json
  anomalies_manifest.json    # manifest détaillé
  anomalies_flat.json        # version aplatie
  anomalies_duplicates.json
  anomalies_missing.json
```

Endpoints (principalement) :

- `GET /monitoring/data/health/available`
- `GET /monitoring/data/health/{doc}?at=...`

`doc` ∈ :

- `kpis` : santé globale des données.
- `station_health` : score / statut par station.
- `coverage_by_hour` : taux de présence des données par heure.
- `anomalies*` : différentes vues sur les anomalies détectées.
- `alerts` : alertes prêtes à afficher côté UI.

---

## 8. Data — Drift

**Fichier :** `data_drift.py`  
**Prefix API :** `/monitoring/data`

Alimenté par `build_data_drift.py` :

```text
<MON>/data/drift/latest/
  psi_by_feature.json
  ks_by_feature.json
  deltas_by_feature.json
  psi_global_daily_ema.json
  summary.json
  alerts.json
  bounds.json
  zones.json
  features_detected.json
```

Endpoints :

- `GET /monitoring/data/drift`  
  → document principal (résumé) de drift.

- `GET /monitoring/data/drift/{name}`  
  → accès direct à un fichier particulier, avec `name` ∈ :

  - `psi_by_feature`
  - `ks_by_feature`
  - `deltas_by_feature`
  - `psi_global_daily_ema`
  - `summary`
  - `alerts`
  - `bounds`
  - `zones`
  - `features_detected`

- `GET /monitoring/data/drift/available`  
  → expose la liste des noms valides + quelques métadonnées.

---

## 9. Data — Freshness

**Fichier :** `data_freshness.py`  
**Prefix API :** `/monitoring/data`

Alimenté par `build_data_health.py` (ou job dédié) :

```text
<MON>/data/freshness/latest.json
```

Endpoint :

- `GET /monitoring/data/freshness`  
  → renvoie le JSON “freshness” consolidé (P95/P50 des retards, météo, etc.).

TTL très court (30 s) pour suivre la fraîcheur des flux.

---

## 10. Intro (bloc de synthèse monitoring)

**Fichier :** `intro.py`  
**Prefix API :** `/monitoring`

Alimenté par `build_monitoring_intro.py` :

```text
<MON>/intro/<latest|TS>/intro.json
```

Endpoints :

- `GET /monitoring/intro/available`  
  → indique que `intro` est le seul document + rappelle le time‑travel.

- `GET /monitoring/intro?at=...`  
  → sert `intro.json`, qui synthétise :
    - quelques KPIs clés,
    - des liens vers les autres pages,
    - de la méta‑information sur la fenêtre temporelle.

---

Ce README documente **le contrat JSON des routes de monitoring**.  
Pour la logique de calcul (clusters, PSI, lift, features…), se référer au README du dossier `service/` et aux jobs `build_*.py` correspondants.
