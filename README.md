# 🚲 Vélib’ Paris — Forecast & Monitoring

[![CI — pipeline](https://github.com/Adrien-1997/bike-forecast-paris-velib/actions/workflows/ingest.yml/badge.svg)](https://github.com/Adrien-1997/bike-forecast-paris-velib/actions/workflows/ingest.yml)
[![Model training](https://github.com/Adrien-1997/bike-forecast-paris-velib/actions/workflows/train.yml/badge.svg)](https://github.com/Adrien-1997/bike-forecast-paris-velib/actions/workflows/train.yml)
[![Docs](https://github.com/Adrien-1997/bike-forecast-paris-velib/actions/workflows/gh-pages.yml/badge.svg)](https://adrien-1997.github.io/bike-forecast-paris-velib/)
[![App Streamlit](https://img.shields.io/badge/app-streamlit-green)](https://adrien-1997-bike-forecast-paris-velib-appstreamlit-app-vq1xma.streamlit.app/)
[![Version](https://img.shields.io/badge/version-v1.1.0-blue.svg)](https://github.com/Adrien-1997/bike-forecast-paris-velib/releases)
![License](https://img.shields.io/badge/License-MIT-black)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB)

![Carte réseau](docs/assets/map.png)

**Prévoir et surveiller en temps réel l’usage du réseau Vélib’ parisien à partir des données publiques (GBFS).**

👉 [📖 Documentation](https://adrien-1997.github.io/bike-forecast-paris-velib/)  
👉 [🎛️ Démo Streamlit](https://adrien-1997-bike-forecast-paris-velib-appstreamlit-app-vq1xma.streamlit.app/)

---

## 📊 Fonctionnalités

- **Monitoring temps réel** : carte interactive des stations avec occupation, vélos disponibles et bornes libres.  
- **KPI & historique** : évolution des 72 dernières heures (occupation moyenne, vélos totaux, disponibilité réseau).  
- **Prévisions (ML)** : modèles **LightGBM** prédisent la disponibilité des stations à **+1h, +3h et +6h**.  
- **Carte enrichie** : visualisation combinant état actuel et projections à court terme.  
- **CI/CD GitHub Actions** : ingestion, entraînement, mise à jour automatique des exports et du site.

---

## 📷 Aperçu

### Occupation moyenne réseau (72h)
![Occupation moyenne](docs/assets/figs/occupancy_last72h.png)

### Vélos disponibles (total réseau, 72h)
![Bikes total](docs/assets/figs/bikes_total_last72h.png)

### Exemple de prévision par station
![Prévision station](docs/assets/figs/obs_pred_42503_T+1h.png)

---

## 🛠️ Pipeline technique

```mermaid
flowchart LR
    A[Ingestion GBFS (Opendata Paris)] --> B[Snapshots DuckDB]
    B --> C[Aggregation (hourly)]
    C --> D[ML Forecast (LightGBM)]
    D --> E[Exports (CSV, Parquet)]
    E --> F[Monitoring & Visualisation]
    F --> G[Docs (MkDocs) + App (Streamlit)]
```

---

## 🚀 Déploiement

### Exécution locale
```bash
# 1. Ingestion des données brutes
python -m src.ingest

# 2. Agrégation horaire
python -m src.aggregate

# 3. Génération du reporting (figures, KPI)
python tools/make_report.py

# 4. Lancer la documentation en local
mkdocs serve
```

### CI/CD GitHub Actions
- `ingest.yml` : ingestion des données.  
- `train.yml` : entraînement quotidien du modèle LightGBM.  
- `gh-pages.yml` : génération et déploiement de la doc sur GitHub Pages.  

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
