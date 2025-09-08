# Results

**Snapshots**: 78427  •  **Stations**: 1453  •  **Last (UTC)**: 2025-09-08 16:00:08.879739

**Historique couvert** : 2025-09-08 01:00:00+02:00 → 2025-09-08 18:00:00+02:00  
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
|          9023 | Laffitte - Italiens            |     0.442 |
|         15058 | Place du Moulin de Javel       |     0.436 |
|          8103 | Artois - Berri                 |     0.429 |
|          1023 | Saint-Honoré - Musée du Louvre |     0.427 |
|         13024 | Bobillot - Tolbiac             |     0.427 |
|         11026 | Chemin Vert - Saint-Maur       |     0.427 |
|         15133 | Saint Lambert - Blomet         |     0.425 |
|          1025 | Oratoire - Rivoli              |     0.423 |

## Exports
- [Prévision 24h (CSV)](exports/velib_forecast_24h.csv)
- [Occupations horaires (échantillon CSV)](exports/velib_hourly.csv)