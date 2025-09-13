# 🚲 Vélib’ Paris — Forecast & Monitoring

[![CI — ingestion](https://github.com/Adrien-1997/bike-forecast-paris-velib/actions/workflows/ingest.yml/badge.svg)](https://github.com/Adrien-1997/bike-forecast-paris-velib/actions/workflows/ingest.yml)
[![CI — training](https://github.com/Adrien-1997/bike-forecast-paris-velib/actions/workflows/train.yml/badge.svg)](https://github.com/Adrien-1997/bike-forecast-paris-velib/actions/workflows/train.yml)
[![Docs](https://github.com/Adrien-1997/bike-forecast-paris-velib/actions/workflows/site.yml/badge.svg?branch=main)](https://adrien-1997.github.io/bike-forecast-paris-velib/)
[![App Streamlit](https://img.shields.io/badge/app-streamlit-green)](https://adrien-1997-bike-forecast-paris-velib-appstreamlit-app-vq1xma.streamlit.app/)
![Version](https://img.shields.io/badge/version-v1.2.0-blue.svg)
![License](https://img.shields.io/badge/License-MIT-black)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB)

![Carte réseau](docs/assets/figs/map.png)

**Prévoir et surveiller en temps réel l’usage du réseau Vélib’ parisien à partir des données publiques (GBFS).**

👉 [📖 Documentation](https://adrien-1997.github.io/bike-forecast-paris-velib/)  
👉 [🎛️ Démo Streamlit](https://adrien-1997-bike-forecast-paris-velib-appstreamlit-app-vq1xma.streamlit.app/)

---

## 📊 Fonctionnalités

- **Ingestion toutes les 15 min** : snapshot complet du réseau, stocké en DuckDB.  
- **Agrégation 15 min** : exports standardisés (`docs/exports/velib.parquet`, `.csv`) enrichis avec météo.  
- **Monitoring temps réel** : carte interactive avec occupation, vélos disponibles, bornes libres.  
- **KPI & historique** : suivi de l’occupation, vélos totaux, disponibilité réseau.  
- **Prévisions (ML)** : modèle **LightGBM** prédit le nombre de vélos disponibles à **+1h (T+60 min)**.  
- **CI/CD GitHub Actions** :
  1) `velib-ingest` (toutes les 5 min) → snapshots API → DuckDB → agrégation 15 min → export `docs/exports/velib.parquet`.
  2) `velib-train` (quotidien, fallback) → réentraînement forcé du LightGBM (MAE/RMSE valid), mise à jour `models/*.joblib` et `docs/exports/baseline.json`.
  3) `monitoring-site` (*/15 min) → génère métriques & pages MkDocs, détecte drift/perf (**PSI ≥ 0.20** ou **MAE_24h ≥ 1.20×baseline**) et **déclenche un retrain immédiat si seuil dépassé**, reconstruit les pages (importances à jour), puis build & déploie sur `gh-pages`.
- **Artefacts ML** : modèle sauvegardé dans `models/lgb_nbvelos_T+60min.joblib` (téléchargeable aussi comme artifact CI).

---

## 📷 Aperçu

### Occupation moyenne réseau (72h)
![Occupation moyenne](docs/assets/figs/occupancy_last72h.png)

### Vélos disponibles (total réseau, 72h)
![Bikes total](docs/assets/figs/bikes_total_last72h.png)

### Exemple de prévision (T+60 min)
![Prévision station](docs/assets/figs/obs_pred_42503_T+1h.png)

---

## 🛠️ Pipeline technique

```
flowchart LR
    A[Ingestion GBFS (Opendata Paris) chaque 15 min] --> B[Snapshots DuckDB]
    B --> C[Agrégation 15 min + Météo]
    C --> D[Exports (Parquet/CSV) docs/exports/]
    D --> E[ML Forecast LightGBM T+60min]
    E --> F[Monitoring & Visualisation]
    F --> G[Docs (MkDocs) + App (Streamlit)]
```

---

## 🚀 Déploiement

### Exécution locale
```bash
# 1. Ingestion (snapshots DuckDB)
python -m src.ingest

# 2. Agrégation 15 min + météo
python -m src.aggregate

# 3. Entraînement ML (LightGBM T+60 min)
python -m src.forecast

# 4. Génération du reporting (figures, KPI, forecast pages)
python tools/generate_monitoring.py

# 5. Lancer la documentation en local
mkdocs serve
```

### CI/CD GitHub Actions
- `ingest.yml` : ingestion des données.  
- `train.yml` : entraînement quotidien du modèle LightGBM.  
- `site.yml` : génération et déploiement de la doc sur GitHub Pages.  

---

## 📂 Structure du projet

```
bike-forecast-paris-velib/
├── app/              # application Streamlit
├── src/              # ingestion, features, forecast
├── tools/            # scripts (report, map, monitoring)
├── exports/          # données exportées (csv, parquet)
├── docs/             # site MkDocs (figures, pages)
│   ├── assets/figs/  # visualisations générées
│   └── *.md
├── models/           # modèles ML sauvegardés (.joblib)
├── warehouse.duckdb  # snapshots DB (non versionné conseillé)
├── mkdocs.yml        # configuration site
└── requirements.txt
```

---

## 👤 Auteur

Projet développé par **Adrien Morel** — Data Scientist (maths appliquées & machine learning).  
👉 [Portfolio](https://portfolio-ad94d.web.app/) • [LinkedIn](https://www.linkedin.com/in/adrien-m-1997)

---

## 📜 Licence

Ce projet est distribué sous licence MIT.
