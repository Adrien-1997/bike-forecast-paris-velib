# V�lib� Paris � Forecast (Batch)

Petit portail de **suivi du r�seau**, **performances du mod�le** et **qualit� des donn�es**.  
Tout est recalcul� � partir des exports locaux (bin **15 min**, **UTC** c�t� fichiers ; les pages affichent en **heure de Paris**).

---

## Acc�s rapide

- **R�seau**
  - ?? [Aper�u du r�seau](network/overview.md) � KPIs du jour, carte p�nurie/saturation, journ�e vs m�diane
  - ?? [Stations & profils](network/stations.md) � table filtrable, fiches station, clustering expliqu�
  - ?? [Dynamiques spatio-temporelles](network/dynamics.md) � heatmaps h�j, tension, saisonnalit�s

- **Mod�le**
  - ?? [Performance & baseline](model/performance.md) � MAE/RMSE, lift vs persistance, d�coupages
  - ??? [Pipeline & features](model/pipeline.md) � donn�es d�entr�e, lags/rollings, validation temporelle
  - ?? [Explicabilit� & calibration](model/explainability.md) � importance, PDP/ICE, calibration, incertitudes

- **Monitoring**
  - ?? [Sant� des donn�es](monitoring/data-health.md) � fra�cheur, compl�tude, sch�ma, anomalies
  - ?? [Drift des donn�es](monitoring/drift.md) � PSI/K�S, segments, tendances liss�es
  - ?? [Sant� du mod�le](monitoring/model-health.md) � d�rives de performance, couverture, r�gles d�alerte

- **Donn�es**
  - ?? [Exports](data/exports.md) � fichiers publi�s & contrat minimal
  - ?? [Dictionnaire & sch�ma](data/dictionary.md) � types, unit�s, contraintes, validations
  - ?? [M�thodologie & licences](data/methodology.md) � fabrication, versionnage, usages

---

## � savoir

- **Horodatage** : les fichiers sont en **UTC** (arrondis :00/:15/:30/:45). Les pages affichent en **Europe/Paris**.  
- **Horizon par d�faut** : 60 min (persistance comme baseline).  
- **Cl�s** : `(ts, station_id)` unique dans `events.parquet` et `perf.parquet`.

---

## Sources

- ?? Code : [repo GitHub](https://github.com/Adrien-1997/bike-forecast-paris-velib)  
- ?? Exports locaux : `docs/exports/` (non versionn�s)
