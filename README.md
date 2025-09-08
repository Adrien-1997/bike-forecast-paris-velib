# 🚲 Vélib’ Paris — Forecast & Risk (T+3h/T+6h)

[![CI — pipeline](https://github.com/Adrien-1997/bike-forecast-paris-velib/actions/workflows/pipeline.yml/badge.svg)](https://github.com/Adrien-1997/bike-forecast-paris-velib/actions/workflows/pipeline.yml)
[![Docs — GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-2962FF)](https://adrien-1997.github.io/bike-forecast-paris-velib/)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB)
![License](https://img.shields.io/badge/License-MIT-black)

**But :** aider les équipes exploitation/logistique à **anticiper les stations à risque** (rupture <20% ou surcharge >80%) sur les **3–6 prochaines heures**, avec une boucle **data → modèle → résultats** rafraîchie **toutes les 15 min**.

- **Résultats en ligne :** https://adrien-1997.github.io/bike-forecast-paris-velib/  
  (page *Results* : graphe historique+prévision, top risques/volatilité, corrélation, carte)
- **Repo :** https://github.com/Adrien-1997/bike-forecast-paris-velib

---

## ✨ Fonctionnalités

- **Ingestion temps réel** Vélib’ (Paris Data) → **DuckDB** (UTC)
- **Agrégat horaire** par station (`occ_ratio_hour`)
- **Prévision 24h** (baseline tabulaire ; features calendrier + météo Open-Meteo)
- **Scores de risque** (seuils **0.20** / **0.80**) pour T+3h/T+6h
- **Exports** prêts pour Excel/BI (CSV/Parquet)
- **Docs automatiques** (MkDocs) + **Carte** Folium (dernier snapshot)
- **CI GitHub Actions** : CRON */15 pour collecter → agréger → prévoir → publier

![hero](docs/assets/hero_occ.png)

---

## 🧠 Logique “risque” (opérationnelle)

- **Rupture** si `occupation ≤ 0.20` ; **Surcharge** si `occupation ≥ 0.80`.  
- Pour chaque station, on calcule un **score [0–1]** = distance normalisée au seuil le plus proche, puis on prend le **max** sur les horizons **T+3h** / **T+6h** pour prioriser l’action.

> Les seuils 0.20 / 0.80 sont des valeurs par défaut (démo) et **ajustables**.

---

## ⚙️ Quickstart (local)

### Windows (PowerShell)

    # 1) Environnement
    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate
    py -m pip install -U pip
    py -m pip install -r requirements.txt

    # 2) (option) Snapshot temps réel
    py -m src.ingest   # écrit dans warehouse.duckdb

    # 3) Agréger & Prévoir
    py -m src.aggregate
    py -m src.run_batch

    # 4) Générer la page "Results" + Carte
    py tools\make_report.py
    py tools\make_map.py

    # 5) Servir la doc en local
    py -m mkdocs serve -a 127.0.0.1:8000

### macOS / Linux

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -U pip
    pip install -r requirements.txt

    python -m src.ingest
    python -m src.aggregate
    python -m src.run_batch

    python tools/make_report.py
    python tools/make_map.py

    python -m mkdocs serve -a 127.0.0.1:8000

*(option) App Streamlit locale :* `streamlit run app/app.py`

---

## 🗂️ Structure du repo

    .
    ├─ .github/workflows/
    │  └─ pipeline.yml            # CRON */15 → data + forecast + docs + gh-pages
    ├─ app/
    │  └─ app.py                  # App Streamlit (démo risque / priorités)
    ├─ src/
    │  ├─ velib_client.py         # client OpenData (Explore v2.1 paginé)
    │  ├─ ingest.py               # append snapshot → DuckDB
    │  ├─ aggregate.py            # agrégat horaire + jointure météo
    │  ├─ forecast.py             # entraînement + 24h/station (baseline)
    │  ├─ cal_features.py         # WE / heures de pointe / fériés FR
    │  ├─ weather.py              # Open-Meteo (archive+forecast + cache)
    │  └─ run_batch.py            # orchestration batch local
    ├─ tools/
    │  ├─ make_report.py          # génère docs/results.md + visuels
    │  ├─ make_map.py             # Folium (docs/assets/map.html)
    │  └─ make_share_image.py     # mosaïque LinkedIn 1200×627 (option)
    ├─ docs/                      # site MkDocs (Material)
    │  ├─ index.md
    │  ├─ results.md
    │  └─ assets/
    │     ├─ extra.css
    │     ├─ *.png
    │     └─ map.html
    ├─ exports/                   # CSV/Parquet générés (ignorés du VCS)
    ├─ data/                      # caches (ex. météo) ignorés du VCS
    ├─ warehouse.duckdb           # stockage snapshots local (ignoré)
    ├─ mkdocs.yml
    ├─ requirements.txt
    ├─ DATA_SOURCES.md
    └─ README.md

---

## 🔄 CI/CD

- **Workflow** : `.github/workflows/pipeline.yml`  
  Étapes : *ingest → prune (>30j) → aggregate → forecast → report → mkdocs build → publish gh-pages*.  
- **GitHub Pages** : branche `gh-pages` (dossier racine).  
- Déploiement manuel possible : `python -m mkdocs gh-deploy --force`.

---

## 📊 Monitoring (WIP)

- Résumé **MAPE/MAE** sur 24h (page *Results*).  
- À venir : dérive (PSI/KS), qualité des features, stabilité des seuils, alerte dégradation.

---

## 🌐 Données & limites

- **Vélib’ (Paris Data / Opendatasoft)** — disponibilité temps réel par station, via API **Explore v2.1** (pagination `limit<=100` + `offset`).  
- **Open-Meteo** — historique & prévision horaires (température, précipitation, vent).  
- L’agrégat horaire le plus récent correspond à la **dernière heure complète (UTC)**.

Détails : `DATA_SOURCES.md`.

---

## 🧪 Reproduire les visuels

    # Après aggregate + run_batch
    python tools/make_report.py
    python tools/make_map.py
    python -m mkdocs build

Sorties principales :
- `docs/results.md` (+ images sous `docs/assets/`)  
- `docs/assets/map.html`  
- `exports/velib_hourly.*` & `exports/velib_forecast_24h.*`

---

## 🛣️ Roadmap

- [ ] **Alerting** (Slack/webhook) sur top risques T+3h/T+6h  
- [ ] **Clustering** de stations (profils / quartiers)  
- [ ] **Modèles enrichis** (météo fine, évènements, mobilité)  
- [ ] **Monitoring drift** (PSI/KS, cibles) + budget d’erreurs  
- [ ] **API JSON** (`/forecast?station=`) + Docker  
- [ ] **Cartes** : clusters + heatmap horaire

---

## 📝 Licence & crédits

- MIT — voir `LICENSE`  
- Données : Paris Data (Opendatasoft) & Open-Meteo  
- Auteur : **Adrien Morel** — Paris
