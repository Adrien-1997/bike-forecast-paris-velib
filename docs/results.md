# Results

**Snapshots**: 79880  •  **Stations**: 1453  •  **Last (UTC)**: 2025-09-08 16:06:48.004683

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
|         21021 | Enfants du Paradis - Peupliers |     0.509 |
|         15056 | Place Balard                   |     0.459 |
|          9023 | Laffitte - Italiens            |     0.441 |
|          1023 | Saint-Honoré - Musée du Louvre |     0.429 |
|          8103 | Artois - Berri                 |     0.428 |
|         11026 | Chemin Vert - Saint-Maur       |     0.426 |
|         13024 | Bobillot - Tolbiac             |     0.424 |
|         15133 | Saint Lambert - Blomet         |     0.423 |
|         11003 | Keller - La Roquette           |     0.423 |
|         15058 | Place du Moulin de Javel       |     0.421 |

## Exports
- [Prévision 24h (CSV)](exports/velib_forecast_24h.csv)
- [Occupations horaires (échantillon CSV)](exports/velib_hourly.csv)