# Services de Monitoring -- Documentation Frontend

Ce dossier regroupe l'ensemble des modules TypeScript utilisés par la
section **Monitoring** du frontend Vélib' Forecast.\
Chaque fichier correspond directement à un endpoint backend et fournit :

-   Des interfaces TypeScript strictes pour les schémas JSON\
-   Des helpers d'appel API avec gestion d'ETag\
-   Une séparation nette entre récupération des données et affichage UI

------------------------------------------------------------------------

## Services inclus

### 1. `network_overview.ts`

Expose les indicateurs globaux du réseau : - Couverture\
- Fraîcheur des données\
- Instantanés de distribution\
- Métriques d'équilibre et de stabilité

### 2. `network_stations.ts`

Fournit : - KPI de clusterisation\
- Indicateurs agrégés 7 jours par station\
- Centroïdes de clusters (profils type)\
- Index léger des stations (id → métadonnées)

### 3. `network_dynamics.ts`

Décrit : - Dynamiques temporelles du réseau\
- Séries journalières / horaires agrégées\
- Évolution observée vs attendue

### 4. `model_performance.ts`

Contient : - KPI de performance du modèle\
- Métriques journalières\
- Performances par heure et par jour de semaine\
- Données organisées par horizon

### 5. `model_explainability.ts`

Fournit : - Importance globale des features\
- Résumés SHAP\
- Artefacts d'explicabilité par horizon

### 6. `data_health.ts`

Documente : - Validité\
- Complétude\
- Synthèse hebdomadaire de la santé des données\
- Anomalies détectées par l'ingestion

### 7. `data_freshness.ts`

Expose : - Fraîcheur des données GBFS\
- Timestamps binaires\
- KPI de fraîcheur globale

### 8. `data_drift.ts`

Regroupe : - PSI global\
- Feature la plus en drift\
- PSI, KS, et deltas par feature\
- Série temporelle de PSI EMA

### 9. `weather_freshness.ts`

Décrit : - Latence météo (p95)\
- Métadonnées fournisseurs\
- Timestamps de génération backend

### 10. `intro.ts`

Alimente l'introduction de la section Monitoring avec : - Mini-KPIs\
- Agrégats réseau récents\
- Métadonnées consolidées

------------------------------------------------------------------------

## Architecture

Tous les services de monitoring :

-   utilisent `fetchJsonWithEtag` ou `getJSON`\

-   imposent un typage strict via interfaces TS\

-   sont consommés par les pages React sous `ui/pages/monitoring/...`\

-   reflètent la structure JSON produite dans le bucket GCS :

        gs://.../monitoring/<domaine>/<sous-domaine>/latest

------------------------------------------------------------------------

## Exemple d'utilisation

``` ts
import { getNetworkOverview } from "@/lib/services/monitoring/network_overview";

const data = await getNetworkOverview();
console.log(data.kpis.coverage_pct);
```

------------------------------------------------------------------------

## Maintenance

En cas d'évolution du schéma côté backend :

1.  Mettre à jour les interfaces TypeScript concernées\
2.  Vérifier l'alignement complet avec les endpoints\
3.  Tester la validité ETag / cache navigateur\
4.  Maintenir la cohérence entre UI & artefacts monitoring

------------------------------------------------------------------------

## Licence

Documentation interne -- usage non destiné à la diffusion publique.
