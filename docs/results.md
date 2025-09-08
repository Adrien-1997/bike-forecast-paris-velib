# Results

**Snapshots**: 68256  •  **Stations**: 1453  •  **Last (UTC)**: 2025-09-08 13:51:01.280329

**Historique couvert** : 2025-09-08 01:00:00+02:00 → 2025-09-08 15:00:00+02:00  
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
|         21021 | Enfants du Paradis - Peupliers |     0.501 |
|         15056 | Place Balard                   |     0.493 |
|          9023 | Laffitte - Italiens            |     0.461 |
|         33019 | Madeleine Vionnet              |     0.457 |
|          8103 | Artois - Berri                 |     0.45  |
|         15058 | Place du Moulin de Javel       |     0.443 |
|         15125 | Parc Suzanne Lenglen           |     0.443 |
|          8049 | Georges V - François 1er       |     0.435 |
|          9022 | Rossini - Laffitte             |     0.433 |
|         16201 | Porte Dauphine                 |     0.433 |

## Exports
- [Prévision 24h (CSV)](exports/velib_forecast_24h.csv)
- [Occupations horaires (échantillon CSV)](exports/velib_hourly.csv)