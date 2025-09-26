# RUNBOOK — Vélib’ Pipeline

## GCS Layout
```
gs://<bucket>/velib/
  bronze/date=YYYY-MM-DD/hour=HH/*.parquet
  db/
    daily/velib_YYYYMMDD.duckdb
    reporting/velib_reporting.duckdb
  serving/features_4h/latest.parquet
  exports/*.csv
  locks/
```
- **bronze/** : snapshots 5 min (append)
- **db/daily/** : shards DuckDB journaliers J-1
- **db/reporting/** : DuckDB central (silver/gold)
- **serving/** : fichier unique pour l’app
- **exports/** : CSV/Parquet publiables (optionnel)

## Jobs & Horaires (Cloud Run Jobs + Cloud Scheduler)
- **velib-ingest** — */5 min* — écrit bronze (`JOB=ingest`)
- **velib-features-4h** — */10 min* — écrit serving (`JOB=build_features_4h`)
- **velib-dim-station** — 00:55 — MAJ dim.dim_station (`JOB=dim_station`)
- **velib-compact-daily** — 01:00 — compact J-1 → daily shard (`JOB=compact_daily`)
- **velib-daily** — 01:05 — calcule silver & gold J-1 (`JOB=build_daily`)
- **velib-windows** — 01:10 — gold rolling 7j (`JOB=build_windows`)
- **velib-monthly** — 01:15 — gold.monthly_metrics (`JOB=build_monthly`)

> Chaque job utilise l’image Artifact Registry et des **env vars** (voir section suivante).

## Variables d’Environnement (par job)
### ingest
```
JOB=ingest
INGEST_TO_GCS=1
GCS_RAW_PREFIX=gs://<bucket>/velib/bronze
```
### features_4h
```
JOB=build_features_4h
GCS_RAW_PREFIX=gs://<bucket>/velib/bronze
GCS_SERVING_PREFIX=gs://<bucket>/velib/serving/features_4h
WINDOW_MIN=240
TZ_APP=Europe/Paris
```
### dim_station
```
JOB=dim_station
GCS_DB_URI=gs://<bucket>/velib/db/reporting/velib_reporting.duckdb
DB_LOCAL=/tmp/velib_reporting.duckdb
```
### compact_daily
```
JOB=compact_daily
GCS_RAW_PREFIX=gs://<bucket>/velib/bronze
GCS_DB_DAILY=gs://<bucket>/velib/db/daily
DB_LOCAL=/tmp/velib_daily.duckdb
```
### daily / windows / monthly
```
GCS_DB_URI=gs://<bucket>/velib/db/reporting/velib_reporting.duckdb
DB_LOCAL=/tmp/velib_reporting.duckdb
```
- daily: `JOB=build_daily`
- windows: `JOB=build_windows`, `WINDOW_DAYS=7`
- monthly: `JOB=build_monthly`

### (Optionnel) Locks GCS
```
GCS_LOCK=gs://<bucket>/velib/locks/<job>.lock
```

## Commandes Utiles
### Exécuter un job manuellement
```bash
gcloud run jobs execute velib-ingest --region europe-west1
```

### Vérifier la dernière ingestion
```bash
day=$(date -u +%F); hour=$(date -u +%H)
gsutil ls gs://<bucket>/velib/bronze/date=$day/hour=$hour/*.parquet
```

### Ouvrir la DB reporting en local
```bash
gsutil cp gs://<bucket>/velib/db/reporting/velib_reporting.duckdb velib_reporting.duckdb
duckdb velib_reporting.duckdb
-- puis:
SELECT * FROM gold.data_health_daily ORDER BY date DESC LIMIT 10;
```

### Serving — contrôle rapide
```bash
gsutil ls -al gs://<bucket>/velib/serving/features_4h/latest.parquet
```

## Qualité & Contrôles
- **Bronze** : 0 ≤ bikes ≤ capacity ; lat/lon bbox Paris ; timestamps UTC
- **Silver/Gold** : completeness_pct ~ stations*288 ; freshness_min < 15
- **Serving** : présence du fichier, taille non nulle, fraîcheur < 15 min

## Alerting (Cloud Monitoring)
- **Job failure** :
  - Log filter: `resource.type="cloud_run_job" severity>=ERROR`
  - Condition: count > 0 sur 15 min ; canal email/Slack
- **Freshness > 15 min** :
  - Vérifier l’âge de `serving/features_4h/latest.parquet` via un check planifié
  - Ou calculer `freshness_min` depuis `gold.data_health_daily`

## CI/CD (Cloud Build)
- `cloudbuild.yaml` à la racine : build & push image → update jobs
- Trigger sur branche `main`
- Rôles Cloud Build SA : `roles/artifactregistry.writer`, `roles/run.admin`, `roles/iam.serviceAccountUser`

## Lifecycle GCS (exemple)
- `bronze/` : 365 j
- `db/daily/` : 730 j
- `serving/` : 2 j
- `exports/` : 180 j
Appliquer : `gsutil lifecycle set bucket-lifecycle.json gs://<bucket>`

## IAM
- Service Account des jobs : `roles/storage.objectAdmin` sur le bucket
- Scheduler → Run : OIDC `run.invoker`
- Secrets (HF_TOKEN, autres) via Secret Manager → env vars

## Dépannage Rapide
- **Parquet non trouvé** : vérifier `GCS_RAW_PREFIX`, horodatage UTC vs Paris, IAM sur bucket
- **ImportError pyarrow** : add `pyarrow` dans requirements image
- **DB reporting manquante** : job `build_daily` la crée si besoin ; vérifier env `GCS_DB_URI`
- **429 HF** (exports) : backoff déjà géré dans `tools/push_hf.py`

---

### SLA interne
- Fraîcheur serving < 10 min (objectif)
- Batch quotidien complété avant 01:30 Europe/Paris
- Disponibilité jobs > 99% / mois

### Contacts
- Owner GCP: <ton-email@domain.tld>
- Bucket: `gs://velib-forecast-472820_cloudbuild/velib`
- Projet: `velib-forecast-472820`
