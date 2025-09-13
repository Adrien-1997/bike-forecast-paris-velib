# Architecture

```mermaid
flowchart TD
    A[Ingestion 5 min] --> B[DuckDB snapshots]
    B --> C[Aggregate 15 min]
    C -->|docs/exports/velib.parquet| D[Features (lags, météo, calendrier)]
    D --> E[Forecast LGBM T+1h]
    E --> F[Monitoring (métriques, drift)]
    F --> G[MkDocs static site]
