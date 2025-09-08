# Results

**Snapshots**: 90051  •  **Stations**: 1453  •  **Last (UTC)**: 2025-09-08 17:40:17.415105

**Historique couvert** : 2025-09-08 01:00:00+02:00 → 2025-09-08 19:00:00+02:00  
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
|         21021 | Enfants du Paradis - Peupliers |     0.506 |
|         15056 | Place Balard                   |     0.47  |
|          9023 | Laffitte - Italiens            |     0.441 |
|         33019 | Madeleine Vionnet              |     0.44  |
|          8103 | Artois - Berri                 |     0.421 |
|          9022 | Rossini - Laffitte             |     0.42  |
|         11026 | Chemin Vert - Saint-Maur       |     0.416 |
|         15125 | Parc Suzanne Lenglen           |     0.414 |
|         15058 | Place du Moulin de Javel       |     0.408 |
|         13024 | Bobillot - Tolbiac             |     0.407 |

## Exports
- [Prévision 24h (CSV)](exports/velib_forecast_24h.csv)
- [Occupations horaires (échantillon CSV)](exports/velib_hourly.csv)