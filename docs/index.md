# Vélib’ Paris — Forecast (Batch)

Petit portail de **suivi du réseau**, **performances du modèle** et **qualité des données**.  
Tout est recalculé à partir des exports locaux (bin **15 min**, **UTC** côté fichiers ; les pages affichent en **heure de Paris**).

---

## Accès rapide

- **Réseau**
  - 🔎 [Aperçu du réseau](network/overview.md) — KPIs du jour, carte pénurie/saturation, journée vs médiane
  - 🧭 [Stations & profils](network/stations.md) — table filtrable, fiches station, clustering expliqué
  - 🌊 [Dynamiques spatio-temporelles](network/dynamics.md) — heatmaps h×j, tension, saisonnalités

- **Modèle**
  - 📊 [Performance & baseline](model/performance.md) — MAE/RMSE, lift vs persistance, découpages
  - 🛠️ [Pipeline & features](model/pipeline.md) — données d’entrée, lags/rollings, validation temporelle
  - 🔍 [Explicabilité & calibration](model/explainability.md) — importance, PDP/ICE, calibration, incertitudes

- **Monitoring**
  - 🩺 [Santé des données](monitoring/data-health.md) — fraîcheur, complétude, schéma, anomalies
  - 🔀 [Drift des données](monitoring/drift.md) — PSI/K–S, segments, tendances lissées
  - ⚖️ [Santé du modèle](monitoring/model-health.md) — dérives de performance, couverture, règles d’alerte

- **Données**
  - 📦 [Exports](data/exports.md) — fichiers publiés & contrat minimal
  - 📚 [Dictionnaire & schéma](data/dictionary.md) — types, unités, contraintes, validations
  - 📑 [Méthodologie & licences](data/methodology.md) — fabrication, versionnage, usages

---

## À savoir

- **Horodatage** : les fichiers sont en **UTC** (arrondis :00/:15/:30/:45). Les pages affichent en **Europe/Paris**.  
- **Horizon par défaut** : 60 min (persistance comme baseline).  
- **Clés** : `(ts, station_id)` unique dans `events.parquet` et `perf.parquet`.

---

## Sources

- 💻 Code : [repo GitHub](https://github.com/Adrien-1997/bike-forecast-paris-velib)  
- 📄 Exports locaux : `docs/exports/` (non versionnés)
