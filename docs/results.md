# Results

**Snapshots**: 76974  •  **Stations**: 1453  •  **Last (UTC)**: 2025-09-08 15:48:24.282451

**Historique couvert** : 2025-09-08 01:00:00+02:00 → 2025-09-08 17:00:00+02:00  
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
|         21021 | Enfants du Paradis - Peupliers |     0.52  |
|         15056 | Place Balard                   |     0.472 |
|          9023 | Laffitte - Italiens            |     0.458 |
|          8103 | Artois - Berri                 |     0.442 |
|         15058 | Place du Moulin de Javel       |     0.436 |
|          9022 | Rossini - Laffitte             |     0.433 |
|         15125 | Parc Suzanne Lenglen           |     0.432 |
|         11026 | Chemin Vert - Saint-Maur       |     0.431 |
|         15133 | Saint Lambert - Blomet         |     0.431 |
|         11003 | Keller - La Roquette           |     0.429 |

## Exports
- [Prévision 24h (CSV)](exports/velib_forecast_24h.csv)
- [Occupations horaires (échantillon CSV)](exports/velib_hourly.csv)