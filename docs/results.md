# Results

**Snapshots**: 91504  •  **Stations**: 1453  •  **Last (UTC)**: 2025-09-08 17:48:19.643452

**Historique couvert** : 2025-09-08 01:00:00+02:00 → 2025-09-08 19:00:00+02:00  
**Stations** : 1453  
*(Heure affichée : Europe/Paris)*

## Example (historique + forecast 24h)
![sample](assets/sample_forecast.png)

## Corrélation simple
Relation occ_ratio vs. température (échantillon)
![occ vs temp](assets/occ_vs_temp.png)

## Top 10 stations les plus volatiles
|   stationcode | name                                     |   std_occ |
|--------------:|:-----------------------------------------|----------:|
|         21021 | Enfants du Paradis - Peupliers           |     0.503 |
|         15056 | Place Balard                             |     0.47  |
|          9023 | Laffitte - Italiens                      |     0.442 |
|         33019 | Madeleine Vionnet                        |     0.438 |
|          8103 | Artois - Berri                           |     0.42  |
|          9022 | Rossini - Laffitte                       |     0.42  |
|         11026 | Chemin Vert - Saint-Maur                 |     0.416 |
|         15125 | Parc Suzanne Lenglen                     |     0.412 |
|         15058 | Place du Moulin de Javel                 |     0.408 |
|          2009 | Filles Saint-Thomas - Place de la Bourse |     0.407 |

## Exports
- [Prévision 24h (CSV)](exports/velib_forecast_24h.csv)
- [Occupations horaires (échantillon CSV)](exports/velib_hourly.csv)