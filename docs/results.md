# Results

## Exemple (historique + forecast 24h)
<div class="figure">
  <img src="{{ base_url }}/assets/figs/hist_forecast_24h.png" alt="Historique + prévision 24h">
  <div class="caption">Historique agrégé + horizon 24h (échantillon de stations).</div>
</div>

## Corrélation simple
<div class="figure">
  <img src="{{ base_url }}/assets/figs/occ_vs_temp.png" alt="Occupation vs Température">
  <div class="caption">Relation occupation vs température (échantillon horaire).</div>
</div>

## Top 10 stations les plus volatiles

|   stationcode | name                           |   std_occ |
|--------------:|:-------------------------------|----------:|
|         21021 | Enfants du Paradis - Peupliers |     0.501 |
|         15056 | Place Balard                   |     0.46  |

## Carte (dernier snapshot)

<iframe src="{{ base_url }}/assets/map.html" width="100%" height="520" style="border:none;"></iframe>

## Exports
- [Prévision 24h (CSV)]({{ base_url }}/exports/velib_forecast_24h.csv){ target=_blank }
- [Occupations horaires (échantillon CSV)]({{ base_url }}/exports/occ_hourly_sample.csv){ target=_blank }
