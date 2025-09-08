# Results

**Snapshots**: 69709  •  **Stations**: 1453  •  **Last (UTC)**: 2025-09-08 14:12:50.085877

**Historique couvert** : 2025-09-08 01:00:00+02:00 → 2025-09-08 16:00:00+02:00  
**Stations** : 1453  
*(Heure affichée : Europe/Paris)*

## Example (historique + forecast 24h)
![sample](assets/sample_forecast.png)

## Corrélation simple
Relation occ_ratio vs. température (échantillon)
![occ vs temp](assets/occ_vs_temp.png)

## Top 10 stations les plus volatiles
|   stationcode | name                           |   std_occ |
|--------------:|:-------------------------------|----------:|
|         21021 | Enfants du Paradis - Peupliers |     0.517 |
|         15056 | Place Balard                   |     0.483 |
|          9023 | Laffitte - Italiens            |     0.468 |
|          8103 | Artois - Berri                 |     0.448 |
|         15058 | Place du Moulin de Javel       |     0.447 |
|          9022 | Rossini - Laffitte             |     0.44  |
|         15125 | Parc Suzanne Lenglen           |     0.44  |
|         33019 | Madeleine Vionnet              |     0.438 |
|         15133 | Saint Lambert - Blomet         |     0.431 |
|         11003 | Keller - La Roquette           |     0.43  |

## Exports
- [Prévision 24h (CSV)](exports/velib_forecast_24h.csv)
- [Occupations horaires (échantillon CSV)](exports/velib_hourly.csv)