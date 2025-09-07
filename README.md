# Vélib' Paris — Batch Forecast (24h)

[![Pipeline](https://img.shields.io/github/actions/workflow/status/Adrien-1997/bike-forecast-paris-velib/pipeline.yml?branch=main&label=pipeline)](https://github.com/Adrien-1997/bike-forecast-paris-velib/actions/workflows/pipeline.yml)
[![Docs](https://img.shields.io/badge/Docs-GitHub%20Pages-blue)](https://adrien-1997.github.io/bike-forecast-paris-velib/)
[![License](https://img.shields.io/github/license/Adrien-1997/bike-forecast-paris-velib)](./LICENSE)

Prévision **24h** du taux d’occupation des stations **Vélib’ (Paris)**.  
Pipeline “data → décision” minimal : **ingestion** (OpenData), **agrégation** (DuckDB), **modèle** (LightGBM), **exports** (CSV/Parquet) et **site statique** (MkDocs).

- **Démo / Résultats** : https://adrien-1997.github.io/bike-forecast-paris-velib/  
  (graphe en heure locale, top volatilité, carte stations, liens CSV)

---

## Why (20s)

- Les opérations ont besoin d’un **signal simple** pour organiser rééquilibrage & maintenance.
- Ce repo montre **une boucle complète** (collecte planifiée → forecast → partage) avec des choix pragmatiques et faciles à industrialiser.

---

## Features

- Ingestion snapshots temps réel → **DuckDB** (UTC)
- Agrégat horaire par station (**occ_ratio_hour**)
- **LightGBM** baseline + **prévision 24h**
- Enrichissement **météo** (historique + forecast Open-Meteo)
- **Exports** prêts Excel/BI (CSV/Parquet)
- **Docs automatiques** (MkDocs/GitHub Pages) + **carte Folium**
- **Pipeline CRON** toutes les 15 min (GitHub Actions)

---

## Architecture

```
Paris Data (Vélib’ temps réel)
    │  (API Explore v2.1, pagination)
    ▼
Ingestion → DuckDB ──┐
    ▼                │ snapshots (UTC)
Agrégation horaire   │
(occ_ratio/station)  │
    ▼                │
Features calendrier + météo (Open-Meteo)
    ▼
LightGBM baseline → Prévision 24h/station
    ▼
Exports CSV/Parquet + Site MkDocs (GitHub Pages)
```

---

## Résultats (extrait)

![sample](docs/assets/sample_forecast.png)

- **Heure affichée : Europe/Paris**  
- **Carte interactive** : `Results → Carte des stations`  
- **Exports** :
  - `exports/velib_forecast_24h.csv` — prévision 24h par station
  - `exports/velib_hourly.parquet` — occupation horaire par station

---

## Données & limites

- **Vélib’ — disponibilité en temps réel** (Paris Data / Opendatasoft).  
  Endpoint Explore v2.1, pagination `limit<=100` + `offset`.  
  Le schéma peut légèrement varier (`OUI/NON` vs booléens).
- **Open-Meteo** : historique pour l’entraînement, prévision pour le 24h.  
- Détails : **[DATA_SOURCES.md](./DATA_SOURCES.md)**.  
- L’agrégat horaire le plus récent correspond à la **dernière heure complète** (UTC).

---

## Quickstart (local)

> Windows — utilisez `py`; Python **3.11** recommandé.

```powershell
# 1) Environnement
py -3.11 -m venv .venv
.\.venv\Scriptsctivate
py -m pip install -U pip
py -m pip install -r requirements.txt

# 2) (option) Collecter un snapshot supplémentaire
py -m src.ingest

# 3) Agréger & entraîner
py -m src.aggregate
py -m src.run_batch

# 4) Générer la page "Results" + (option) la carte
py tools\make_report.py
py tools\make_map.py

# 5) Servir la doc en local
py -m mkdocs serve -a 127.0.0.1:8000
```

Dépendances clés : `duckdb`, `pandas`, `lightgbm`, `requests`, `pyarrow`, `matplotlib`, `tabulate`, `mkdocs-material`, `mkdocs-jupyter` (+ `folium` pour la carte).

---

## CI/CD

- **`.github/workflows/pipeline.yml`** (CRON 15 min) :  
  ingestion → prune (>30j) → agrégation → forecast → build doc → **publish `gh-pages`**.
- Activer GitHub Pages : **Settings → Pages** → Branch **`gh-pages`**, Folder **`/ (root)`**.

---

## Structure

```
.
├─ src/
│  ├─ velib_client.py        # client OpenData (pagination)
│  ├─ ingest.py              # append snapshot → DuckDB
│  ├─ aggregate.py           # taux d’occupation horaire (+ météo)
│  ├─ forecast.py            # LGBM baseline + forecast 24h
│  └─ run_batch.py           # orchestration batch local
├─ tools/
│  ├─ make_report.py         # docs/results.md + graphe
│  └─ make_map.py            # carte Folium (optionnel)
├─ exports/                  # CSV/Parquet (générés)
├─ docs/                     # site MkDocs (généré + contenu)
├─ .github/workflows/        # pipeline CI/CD
├─ warehouse.duckdb          # stockage snapshots (versionné)
├─ requirements.txt
├─ mkdocs.yml
├─ DATA_SOURCES.md
└─ LICENSE
```

---

## Roadmap

- [ ] Pluie/vent (Open-Meteo) + jours fériés (features calendaires)  
- [ ] Métriques métier (MAE aux heures de pointe, par station)  
- [ ] Alertes simples (seuils d’occupation)  
- [ ] Packaging CLI (`python -m bikevelib run --h=24`)

---

## Licence & crédits

- **MIT** — voir [LICENSE](./LICENSE)  
- Données : Paris Data (Opendatasoft) & Open-Meteo (voir `DATA_SOURCES.md`)  
- Auteur : **Adrien Morel** — Paris
