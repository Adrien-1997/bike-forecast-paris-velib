# 🚲 Vélib’ Paris — Forecast & Monitoring

Bienvenue ! Ce projet prédit le **nombre de vélos disponibles par station Vélib’** aux horizons **T+1h**, **T+3h** et **T+6h**, et propose un **monitoring temps réel** du réseau (cartes, KPI, historiques).

---

## 🎯 Objectifs

- **Aider à la décision** : savoir où trouver/poser un vélo dans 1h/3h/6h.
- **Suivre l’état du réseau** : occupation, stations pleines/vides, tendances.
- **Fournir une base reproductible** : pipeline de collecte → features → modèle → rapport → déploiement.

---

## 🧭 Ce que vous allez trouver

- **Carte interactive** (stations, état en temps réel, tooltips, filtres)
- **KPI réseau** (occupation, vélos totaux, stations critiques)
- **Historique & tendances** (séries temporelles, moyennes glissantes)
- **Prévisions** (horizons T+1h/T+3h/T+6h, erreurs de backtest)
- **Rapport statique** publié via **MkDocs Material**
- **Pipeline GitHub Actions** (collecte automatique toutes les 10 min + build & déploiement)

> URL publique : *voir le badge “Docs” dans le README du dépôt.*

---

## 🏗️ Comment ça marche (vue d’ensemble)

```
GBFS Vélib’ + Météo ──► Ingestion (DuckDB snapshots)
                           │
                           ├──► Agrégat horaire (+ météo)
                           │        └─► Exports (Parquet/CSV)
                           │
                           ├──► Features (lags, calendrier, Fourier, météo)
                           │
                           ├──► Modèle (LightGBM global) + Backtest
                           │        └─► Prévisions (par station, par horizon)
                           │
                           └──► Rapport (MkDocs) + Carte interactive
                                    └─► Déploiement GitHub Pages
```

- **Ingestion** : “snapshots” du GBFS + météo dans **DuckDB**
- **Agrégat** : passage en pas **horaire** + enrichissement météo
- **Modèle** : **LightGBM** global (stationcode catégoriel + features numériques)
- **CI/CD** : GitHub Actions **toutes les 10 min** + build du site

---

## ⚡ Quick Start (local)

```bash
# 1) Environnement
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Pipeline data
py -m src.ingest
py -m src.aggregate

# 3) Visuels & pages
py tools/make_map.py
py tools/make_kpi_banner.py
py tools/make_report.py
py tools/make_forecast_page.py    # ou: py tools/make_backtest.py

# 4) Site local
py -m mkdocs serve
```

**Variables utiles :**
- `FORCE_OFFLINE=1` → génère un snapshot synthétique (tests hors-ligne)
- `NO_SSL_VERIFY=1` → désactive la vérif SSL (utile derrière un proxy)

---

## 🤖 Automatisation (GitHub Actions)

Deux workflows recommandés :

1) **`pipeline.yml`** (toutes les **10 min**)  
   - Ingestion → Agrégat → Visuels/Pages → **Build & Deploy**  
   - Commit automatique des fichiers utiles (`warehouse.duckdb`, `exports/`, `docs/assets/…`)

2) **`gh-pages.yml`** (optionnel si `pipeline.yml` deploy déjà)  
   - Build & déploiement du site à chaque **push sur `main`**

> Si vous ne voyez pas de runs planifiés : vérifiez **Actions activées** et le cron en **UTC**.

---

## 🔬 Modèle & évaluation

- **Modèle** : LightGBM global (une seule instance pour toutes les stations)
- **Features** :
  - Lags (1h, 2h, 24h), rolling windows
  - Météo (température, vent, précipitations)
  - Calendrier (jour/semaine, weekend, jours fériés FR)
  - Fourier (24h, 7j)
  - Capacité station
- **Backtest** : **expanding window** (TimeSeriesSplit)
- **Métriques** : MAE / RMSE / MedAE
- **Visuels** : importances, Observé vs Prédit, résidus

> **Fallback** : en cas de données insuffisantes, un **baseline persistance** (dernier niveau connu par station) est utilisé pour ne pas casser le rapport.

---

## 📊 Ce que montrent les pages

- **Monitoring** : carte interactive + bandeau KPI (vélos totaux, occupation, stations critiques)
- **Historique** : évolutions temporelles (réseau & stations), moyennes glissantes
- **Prévisions** : tableaux & graphiques par horizon, erreurs de backtest, top stations (manques/pleins)
- **Méthodo** : description pipeline, variables, limites et pistes d’amélioration

---

## ❓ FAQ express

- **À quelle fréquence les données sont-elles mises à jour ?**  
  Toutes les **10 minutes** via GitHub Actions (UTC). Localement, à la demande.

- **Pourquoi une prédiction est vide ?**  
  Pas assez d’historique pour la station/horizon → baseline → publication tout de même.

- **Je veux plus de signal météo/événementiel**  
  Ajouter des features (pluie cumulée, rafales, événements publics via calendrier…).

- **Je veux tester sans réseau**  
  `FORCE_OFFLINE=1` crée des données synthétiques pour valider le pipeline.

---

## 🗺️ Roadmap (extraits)

- 🔧 Modèle **LightGBM** affiné (tuning, interactions, station clustering)
- 🧠 Passage à des modèles **hiérarchiques** (globaux + spécifiques)
- 🛰️ Enrichissements **météo/évènementiel** (APIs externes)
- 📈 Page **erreurs en production** (drift, résidus)
- 🎛️ **App Streamlit** d’aide à la décision (reco utilisateur)

---

## ⚠️ Limites & bonnes pratiques

- Attention aux **périodes atypiques** (grèves, météo extrême) : prévoir des features dédiées.
- Les **heures creuses/nuit** peuvent biaiser la métrique globale : comparer par **tranche horaire**.
- Surveillez les **stations récentes** (peu d’historique) → baseline.

---

## 🔗 Liens utiles

- README du dépôt (badges, pipelines)  
- Historique & Prévisions (pages du site)  
- Workflow GitHub Actions (onglet **Actions** du repo)
